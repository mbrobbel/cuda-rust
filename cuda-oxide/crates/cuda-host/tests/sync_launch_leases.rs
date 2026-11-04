/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Synchronous `#[cuda_module]` launches take part in allocation tracking.
//!
//! A safe contracted launch must neither start while a tracked access on
//! another stream conflicts with one of its buffers, nor leave its own buffer
//! accesses invisible to tracked copies and launches issued afterwards. A
//! conflict with recorded work is ordered on the GPU, in both directions, so
//! fork/join code keeps working whether the work after the join is a launch,
//! a tracked copy or an async launch; only an access without a recorded
//! completion is rejected. Graph capture leaves no lasting access behind. A
//! host callback gate holds one stream so each conflict is deterministic.

use cuda_core::{
    Context, CudaStream, DeviceBuffer, DriverError, LaunchConfig, LaunchConfig1D,
    LaunchContractError,
};
use cuda_host::{CudaKernel, cuda_module};
use cuda_macros::{kernel, launch_contract};
use std::{
    sync::{Arc, Mutex, OnceLock, mpsc},
    time::Duration,
};

/// The element count every test buffer uses: one block of 64 threads.
const LEN: usize = 64;

/// The driver's "an access on another stream is still pending" answer.
const NOT_READY: DriverError = DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_READY);

#[cuda_module]
mod kernels {
    use super::*;

    /// Copies `input` into `output` element by element under a launch
    /// contract, so its generated sync launcher is a safe function.
    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn copy_one(input: &[u32], output: cuda_device::DisjointSlice<u32>) {
        let _ = (input, output);
    }

    /// The same copy without a contract, reaching the legacy raw launcher.
    #[kernel]
    pub fn copy_raw(input: &[u32], output: &mut [u32]) {
        let _ = (input, output);
    }
}

/// Returns PTX for one `(ptr, len, ptr, len)` entry named `name` whose
/// threads copy `a[tid]` to `c[tid]` while `tid < b`.
fn copy_entry(name: &str) -> String {
    format!(
        ".visible .entry {name}(.param .u64 a, .param .u64 b, .param .u64 c, .param .u64 d) {{\n\
         .reg .b32 %r<3>; .reg .b64 %rd<8>; .reg .pred %p;\n\
         mov.u32 %r1, %tid.x; cvt.u64.u32 %rd1, %r1; ld.param.u64 %rd2, [b];\n\
         setp.ge.u64 %p, %rd1, %rd2; @%p ret;\n\
         ld.param.u64 %rd3, [a]; ld.param.u64 %rd4, [c];\n\
         cvta.to.global.u64 %rd3, %rd3; cvta.to.global.u64 %rd4, %rd4;\n\
         shl.b64 %rd5, %rd1, 2; add.s64 %rd6, %rd3, %rd5; add.s64 %rd7, %rd4, %rd5;\n\
         ld.global.u32 %r2, [%rd6]; st.global.u32 [%rd7], %r2; ret;\n}}\n"
    )
}

/// Returns a PTX module defining both kernels of [`kernels`] under their
/// generated entry names.
fn ptx() -> String {
    format!(
        ".version 7.0\n.target sm_52\n.address_size 64\n{}{}",
        copy_entry(<kernels::__copy_one_CudaKernel as CudaKernel>::PTX_NAME),
        copy_entry(<kernels::__copy_raw_CudaKernel as CudaKernel>::PTX_NAME),
    )
}

/// CUDA's driver callback thread services `cuLaunchHostFunc` callbacks for the
/// primary context sequentially, and driver module/allocation calls can block
/// while a host callback is parked. Serialize these host-callback-gated tests
/// so default multi-threaded `cargo test` runs do not contend on the driver's
/// callback worker and trip the 10-second gate timeout.
static HOST_CALLBACK_GATE_LOCK: Mutex<()> = Mutex::new(());

/// Holds a stream inside a host callback until dropped.
///
/// Work enqueued behind the gate stays pending, so its access leases are
/// observably live for as long as the test keeps the gate.
struct Gate(mpsc::Sender<()>);

impl Drop for Gate {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// Blocks `stream` in a host callback until the returned gate is dropped, or
/// for at most ten seconds if a failing assertion unwinds past it.
fn gate(stream: &CudaStream) -> Gate {
    let (tx, rx) = mpsc::channel();
    stream
        .launch_host_function(move || {
            let _ = rx.recv_timeout(Duration::from_secs(10));
        })
        .unwrap();
    Gate(tx)
}

/// One context with two independent streams and the bound test module.
///
/// It lives for the whole test process. Dropping a context, stream, module or
/// buffer retires driver objects on cuda-core's background worker, and a
/// retirement that overlaps a parked host callback can stall the driver until
/// the gate times out, letting the gated work finish before the assertions.
struct Fixture {
    a: Arc<CudaStream>,
    b: Arc<CudaStream>,
    module: kernels::LoadedModule,
    prepared: cuda_core::PreparedLaunch<kernels::__copy_one_CudaKernel>,
}

/// Returns the process-wide fixture, creating it on first use.
///
/// Creation launches both kernels once on both streams, so lazy module
/// loading has finished before any test parks a stream.
fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let context = Context::new(0).unwrap();
        let a = context.new_stream().unwrap();
        let b = context.new_stream().unwrap();
        let module = context.load_module_from_ptx_src(&ptx()).unwrap();
        // SAFETY: both PTX entries declare exactly the `(ptr, len, ptr, len)`
        // packet of their Rust signatures, read only `input[..len]`, and write
        // only the matching `output` element of each in-range thread.
        let module = unsafe { kernels::from_module(module) }.unwrap();
        let prepared = module
            .prepare_copy_one(LaunchConfig1D::new(1, LEN as u32, 0))
            .unwrap();
        let input = keep(DeviceBuffer::from_host(&a, &[0u32; LEN]).unwrap());
        let output = keep(DeviceBuffer::<u32>::zeroed(&a, LEN).unwrap());
        for stream in [&a, &b] {
            module.copy_one(stream, &prepared, input, output).unwrap();
            // SAFETY: the raw geometry and the PTX entry match the packet.
            unsafe { module.copy_raw(stream, raw_config(), input, output) }.unwrap();
            stream.synchronize().unwrap();
        }
        Fixture {
            a,
            b,
            module,
            prepared,
        }
    })
}

/// Keeps a test buffer allocated until the process exits.
///
/// See [`Fixture`]: freeing it when one test ends could overlap the gate of
/// the next one.
fn keep(buffer: DeviceBuffer<u32>) -> &'static mut DeviceBuffer<u32> {
    Box::leak(Box::new(buffer))
}

/// The raw launch geometry matching the contracted one.
fn raw_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (LEN as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Drops `gate` on another thread after a short delay, for a test thread
/// that is about to block on work queued behind it.
fn open_later(gate: Gate) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        drop(gate);
    })
}

/// Locks [`HOST_CALLBACK_GATE_LOCK`], ignoring poisoning by a failed test.
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    HOST_CALLBACK_GATE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn sync_launch_waits_for_recorded_cross_stream_write() {
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    let input = keep(DeviceBuffer::from_host(a, &[1u32; LEN]).unwrap());
    let output = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    let unchecked_output = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    let raw_output = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    b.synchronize().unwrap();

    let gate_a = gate(a);
    input.zero_async(a).unwrap();
    // The tracker sees the pending write on another stream.
    assert_eq!(
        input.storage().acquire(b, false, false).err(),
        Some(NOT_READY)
    );
    // The write recorded its completion, so every sync launcher is admitted
    // and makes B wait for it on the GPU instead of reading the old ones.
    module.copy_one(b, prepared, input, output).unwrap();
    // SAFETY: the raw geometry and both PTX entries match their declared
    // packets; only the access tracking is under test here.
    unsafe { module.copy_one_unchecked(b, raw_config(), input, unchecked_output) }.unwrap();
    // SAFETY: as above.
    unsafe { module.copy_raw(b, raw_config(), input, raw_output) }.unwrap();
    let marker = b.record_event(None).unwrap();
    assert!(
        !marker.query().unwrap(),
        "a sync launch on B ran before the pending write on A"
    );
    drop(gate_a);
    b.synchronize().unwrap();

    for out in [&*output, &*unchecked_output, &*raw_output] {
        assert_eq!(
            out.to_host_vec(b).unwrap(),
            vec![0; LEN],
            "a sync launch raced a pending write on another stream"
        );
    }
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn sync_launch_rejects_access_without_recorded_completion() {
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    let input = keep(DeviceBuffer::from_host(a, &[1u32; LEN]).unwrap());
    let output = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    let raw_output = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    a.synchronize().unwrap();
    b.synchronize().unwrap();

    // Leases on A that have not recorded a completion stand for submissions
    // other threads are still enqueuing; nothing can wait on them yet. They
    // guard the outputs, so each launch first leases `input` and must then
    // give that lease back.
    let pending = [
        output.storage().acquire(a, false, false).unwrap(),
        raw_output.storage().acquire(a, false, false).unwrap(),
    ];
    match module.copy_one(b, prepared, input, output) {
        Err(LaunchContractError::Driver(error)) => assert_eq!(error, NOT_READY),
        other => panic!("a safe sync launch raced an unrecorded access: {other:?}"),
    }
    // SAFETY: the raw geometry and both PTX entries match their declared
    // packets; only the access tracking is under test here.
    let unchecked = unsafe { module.copy_one_unchecked(b, raw_config(), input, output) };
    assert_eq!(unchecked.err(), Some(NOT_READY));
    // SAFETY: as above.
    let raw = unsafe { module.copy_raw(b, raw_config(), input, raw_output) };
    assert_eq!(raw.err(), Some(NOT_READY));
    drop(pending);
    b.synchronize().unwrap();

    // A rejected launch enqueues nothing, so both outputs keep their zeros.
    assert_eq!(output.to_host_vec(b).unwrap(), vec![0; LEN]);
    assert_eq!(raw_output.to_host_vec(b).unwrap(), vec![0; LEN]);
    // Rejection released the `input` leases it had already taken.
    assert!(input.storage().acquire(a, true, false).is_ok());
    module.copy_one(b, prepared, input, output).unwrap();
    b.synchronize().unwrap();
    assert_eq!(output.to_host_vec(b).unwrap(), vec![1; LEN]);
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn sync_launches_across_fork_join_are_ordered_not_rejected() {
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    let src = keep(DeviceBuffer::from_host(b, &[9u32; LEN]).unwrap());
    let mid = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    let joined = keep(DeviceBuffer::<u32>::zeroed(a, LEN).unwrap());
    a.synchronize().unwrap();
    b.synchronize().unwrap();

    // The book's fork/join pattern: produce on B, join, consume on A while
    // B's kernel is still pending.
    let gate_b = gate(b);
    module.copy_one(b, prepared, src, mid).unwrap();
    a.join(b).unwrap();
    module.copy_one(a, prepared, mid, joined).unwrap();
    drop(gate_b);
    a.synchronize().unwrap();
    assert_eq!(joined.to_host_vec(a).unwrap(), vec![9; LEN]);

    // A tracked copy after the join waits for the launch on B as well. The
    // copy blocks the host until it ran, so another thread opens the gate.
    mid.zero_async(b).unwrap();
    b.synchronize().unwrap();
    let gate_b = gate(b);
    module.copy_one(b, prepared, src, mid).unwrap();
    a.join(b).unwrap();
    let opener = open_later(gate_b);
    assert_eq!(mid.to_host_vec(a).unwrap(), vec![9; LEN]);
    opener.join().unwrap();
    let gate_b = gate(b);
    module.copy_one(b, prepared, joined, mid).unwrap();
    a.join(b).unwrap();
    let opener = open_later(gate_b);
    mid.copy_from_host(a, &[4u32; LEN]).unwrap();
    opener.join().unwrap();
    b.synchronize().unwrap();
    assert_eq!(mid.to_host_vec(a).unwrap(), vec![4; LEN]);
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn sync_launch_after_forgotten_join_waits_for_recorded_write() {
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    let src = keep(DeviceBuffer::from_host(b, &[9u32; LEN]).unwrap());
    let mid = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    let unjoined = keep(DeviceBuffer::<u32>::zeroed(a, LEN).unwrap());
    a.synchronize().unwrap();
    b.synchronize().unwrap();

    // Without a join, the tracker still orders the read on A after the
    // launch on B that writes `mid`.
    let gate_b = gate(b);
    module.copy_one(b, prepared, src, mid).unwrap();
    module.copy_one(a, prepared, mid, unjoined).unwrap();
    drop(gate_b);
    a.synchronize().unwrap();
    assert_eq!(unjoined.to_host_vec(a).unwrap(), vec![9; LEN]);
}

/// Signals its channel when dropped, telling a waiting test that the
/// retirement worker reached it.
struct Signal(mpsc::Sender<()>);

impl Drop for Signal {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// Waits until cuda-core's retirement worker has run everything queued so far
/// and has no completion event left to poll.
///
/// A global-mode capture is invalidated by any other thread's potentially
/// unsafe driver call, and the worker polls and frees on its own thread. The
/// worker runs every ready item in one pass before taking new ones, so once a
/// second marker queued on the idle `stream` has run, so has everything
/// queued before the first.
fn drain_retirement(stream: &CudaStream) {
    for _ in 0..2 {
        let (ran_tx, ran_rx) = mpsc::channel();
        cuda_core::completion::retain_until_complete(Signal(ran_tx), stream).unwrap();
        ran_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the retirement worker did not drain");
    }
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn captured_sync_launch_leaves_no_lasting_access() {
    use cuda_core::sys;
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    for mode in [
        sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
        sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL,
    ] {
        let input = keep(DeviceBuffer::from_host(a, &[3u32; LEN]).unwrap());
        let output = keep(DeviceBuffer::<u32>::zeroed(a, LEN).unwrap());
        a.synchronize().unwrap();
        drain_retirement(a);
        a.context().bind_to_thread().unwrap();
        // SAFETY: the context is current and `a` is idle and not capturing.
        unsafe { a.begin_capture(mode) }.unwrap();
        let launched = module.copy_one(a, prepared, input, output);
        // SAFETY: the capture was begun above on this stream.
        let graph = unsafe { a.end_capture() }.unwrap();
        launched.unwrap();
        let mut exec = std::mem::MaybeUninit::uninit();
        // SAFETY: `graph` is the valid graph just captured, `exec` is a valid
        // out-pointer, and the instance is launched once on `a`, which the
        // test then synchronizes before destroying both handles exactly once.
        unsafe {
            assert_eq!(
                sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 0),
                sys::cudaError_enum_CUDA_SUCCESS
            );
            let exec = exec.assume_init();
            assert_eq!(
                sys::cuGraphLaunch(exec, a.cu_stream()),
                sys::cudaError_enum_CUDA_SUCCESS
            );
            a.synchronize().unwrap();
            assert_eq!(
                sys::cuGraphExecDestroy(exec),
                sys::cudaError_enum_CUDA_SUCCESS
            );
            assert_eq!(sys::cuGraphDestroy(graph), sys::cudaError_enum_CUDA_SUCCESS);
        }
        // Nothing captured stays registered: a lease recorded inside the
        // capture would never complete and exclude other streams forever.
        assert_eq!(output.to_host_vec(b).unwrap(), vec![3; LEN], "{mode}");
        assert!(output.storage().acquire(b, true, false).is_ok(), "{mode}");
        assert!(input.storage().acquire(b, true, false).is_ok(), "{mode}");
    }
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn tracked_copies_on_other_stream_wait_for_pending_sync_launch() {
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    let src = keep(DeviceBuffer::from_host(a, &[7u32; LEN]).unwrap());
    let dst = keep(DeviceBuffer::<u32>::zeroed(a, LEN).unwrap());
    a.synchronize().unwrap();

    let gate_b = gate(b);
    module.copy_one(b, prepared, src, dst).unwrap();
    // The pending launch writes `dst` and reads `src` on B. A tracked read
    // on A waits for it instead of returning the old zeros at once.
    let opener = open_later(gate_b);
    assert_eq!(
        dst.to_host_vec(a).unwrap(),
        vec![7; LEN],
        "a tracked read observed a buffer a pending sync launch is writing"
    );
    opener.join().unwrap();

    // The tracker sees both accesses of a pending launch.
    let gate_b = gate(b);
    module.copy_one(b, prepared, src, dst).unwrap();
    assert_eq!(
        dst.storage().acquire(a, false, false).err(),
        Some(NOT_READY)
    );
    assert!(
        src.storage().acquire(a, false, false).is_ok(),
        "a pending read must still admit concurrent reads"
    );
    assert_eq!(
        src.storage().acquire(a, true, false).err(),
        Some(NOT_READY),
        "a pending sync launch read must exclude writes on another stream"
    );
    // A tracked write on A is ordered after the pending launch's write, so
    // the launch cannot overwrite it afterwards.
    dst.zero_async(a).unwrap();
    let marker = a.record_event(None).unwrap();
    assert!(
        !marker.query().unwrap(),
        "a tracked write on A overtook a pending sync launch on B"
    );
    drop(gate_b);
    b.synchronize().unwrap();
    a.synchronize().unwrap();
    assert_eq!(dst.to_host_vec(a).unwrap(), vec![0; LEN]);

    // The launch recorded its completion, so a synchronized stream releases
    // the exclusion at once instead of waiting for a background retirement.
    assert!(dst.storage().acquire(b, true, false).is_ok());
}

#[test]
#[ignore = "requires a live CUDA GPU"]
fn sync_launch_admits_same_stream_order_and_concurrent_reads() {
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    let input = keep(DeviceBuffer::from_host(a, &[1u32; LEN]).unwrap());
    let first = keep(DeviceBuffer::from_host(a, &[5u32; LEN]).unwrap());
    let second = keep(DeviceBuffer::from_host(b, &[5u32; LEN]).unwrap());

    let gate_a = gate(a);
    // Stream order already serializes a launch behind a tracked write on the
    // same stream, so the launch is admitted and observes the write.
    input.zero_async(a).unwrap();
    module.copy_one(a, prepared, input, first).unwrap();
    drop(gate_a);
    a.synchronize().unwrap();
    assert_eq!(first.to_host_vec(a).unwrap(), vec![0; LEN]);

    // Reads of an ordinary element type share access across streams.
    let gate_a = gate(a);
    module.copy_one(a, prepared, input, first).unwrap();
    module.copy_one(b, prepared, input, second).unwrap();
    drop(gate_a);
    a.synchronize().unwrap();
    b.synchronize().unwrap();
    assert_eq!(second.to_host_vec(b).unwrap(), vec![0; LEN]);
}

/// A pending `cuda-async` launch on B records its completion once it
/// outlives its inline spin, so a sync launch on A after `a.join(b)` is
/// ordered after it instead of failing with NOT_READY.
#[cfg(feature = "async")]
#[test]
#[ignore = "requires a live CUDA GPU"]
fn sync_launch_after_join_with_pending_async_launch_is_ordered() {
    use cuda_host::cuda_async::{
        device_operation::DeviceOperation, scheduling_policies::SingleStream,
    };
    use std::{
        future::Future,
        pin::Pin,
        task::{Context as TaskContext, Waker},
    };
    let _guard = serialize();
    let Fixture {
        a,
        b,
        module,
        prepared,
    } = fixture();
    let src = keep(DeviceBuffer::from_host(b, &[6u32; LEN]).unwrap());
    let mid = keep(DeviceBuffer::<u32>::zeroed(b, LEN).unwrap());
    let out = keep(DeviceBuffer::<u32>::zeroed(a, LEN).unwrap());
    a.synchronize().unwrap();
    b.synchronize().unwrap();

    let gate_b = gate(b);
    let mut future = module
        .copy_one_async(prepared, &*src, &mut *mid)
        .schedule(&SingleStream::from_stream(b.clone()))
        .unwrap();
    let mut task = TaskContext::from_waker(Waker::noop());
    assert!(Pin::new(&mut future).poll(&mut task).is_pending());
    // The submitted launch keeps running and keeps its leases after the
    // future is dropped, which releases the borrows of `src` and `mid`.
    drop(future);
    a.join(b).unwrap();
    module.copy_one(a, prepared, mid, out).unwrap();
    drop(gate_b);
    a.synchronize().unwrap();
    assert_eq!(out.to_host_vec(a).unwrap(), vec![6; LEN]);
}
