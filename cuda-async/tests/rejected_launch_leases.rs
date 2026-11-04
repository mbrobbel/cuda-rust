/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A launch rejected while acquiring its argument accesses must not leave
//! leases behind on the arguments it had already acquired, and an accepted
//! launch must keep all of them until it completes.
//!
//! Each test launches with two buffers, `x` (write) and `m_in` (read), on a
//! stream B that is kept busy by a closed [`Gate`]. In the rejection tests a
//! write access to `m_in` held on stream A makes the second acquisition fail
//! with `NOT_READY`, so nothing is ever enqueued. The lease on `x` taken just
//! before belongs to no submitted work and must be released on the spot.
//! Before the fix it was handed to the stream-completion retirement path with
//! the rest of the failed submission, and `x` stayed blocked for other
//! streams until all the unrelated work queued on B had finished. The
//! acceptance tests pin the other half of all-or-nothing acquisition: the
//! leases of a launch that was submitted stay with it until B drains.
//!
//! Every launch is polled exactly once while B is gated, so a launch that is
//! accepted when it should be rejected (or the reverse) fails an assertion
//! instead of blocking on the closed gate. Requires a GPU.

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_context::{init_device_contexts, load_module_from_ptx, with_device};
use cuda_async::device_operation::DeviceOp;
use cuda_async::error::DeviceError;
use cuda_async::launch::{AsyncKernelLaunch, AsyncKernelLaunchBuilder};
use cuda_async::scheduling_policies::SingleStream;
use cuda_core::{CudaFunction, DeviceBuffer, DriverError, LaunchConfig, Stream};
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Serializes the tests in this binary: each one parks a stream behind a
/// [`Gate`], and the driver calls of one test must not run behind another
/// test's closed gate.
static GATED_TEST_LOCK: Mutex<()> = Mutex::new(());

/// A no-op kernel taking two device pointers. It only runs in the acceptance
/// tests, after their gate opens; the rejection tests never submit it.
const PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64
.visible .entry write_first(.param .u64 dst, .param .u64 src) { ret; }
"#;

/// Runs `f` on a new thread, so the thread-local device context map starts
/// empty and every test initializes its own.
fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

/// Polls `future` exactly once with a no-op waker. The first poll of a
/// `DeviceFuture` executes its operation: a rejected launch resolves right
/// there, an accepted one stays pending behind a closed gate. Never blocking
/// keeps a wrong outcome from hanging the test on the gate.
fn poll_once<F: Future + Unpin>(future: &mut F) -> Poll<F::Output> {
    Pin::new(future).poll(&mut Context::from_waker(Waker::noop()))
}

/// Whether `result` is the `NOT_READY` access conflict that rejects a
/// launch before anything is enqueued.
fn is_not_ready<T>(result: &Result<T, DeviceError>) -> bool {
    matches!(result, Err(DeviceError::Driver(error)) if is_conflict(error))
}

/// Whether `error` is the `NOT_READY` code an access conflict with
/// another stream reports.
fn is_conflict(error: &DriverError) -> bool {
    error.0 == cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY
}

/// A host-controlled barrier at the head of a stream, as in
/// `drop_in_flight.rs`: a `cuStreamWaitValue32` on a word of pinned,
/// device-mapped host memory keeps every later item on the stream queued
/// until the word is set.
struct Gate {
    host: *mut u32,
    stream: Arc<Stream>,
}

impl Gate {
    /// Arms a closed gate at the current tail of `stream`. Requires a current
    /// CUDA context on the calling thread.
    fn arm(stream: &Arc<Stream>) -> Self {
        let mut host = MaybeUninit::uninit();
        let flags =
            cuda_bindings::CU_MEMHOSTALLOC_PORTABLE | cuda_bindings::CU_MEMHOSTALLOC_DEVICEMAP;
        let code = unsafe {
            cuda_bindings::cuMemHostAlloc(host.as_mut_ptr(), std::mem::size_of::<u32>(), flags)
        };
        assert_eq!(code, 0, "cuMemHostAlloc failed: {code}");
        let host = unsafe { host.assume_init() } as *mut u32;
        unsafe { AtomicU32::from_ptr(host) }.store(0, Ordering::SeqCst);
        let gate = Self {
            host,
            stream: stream.clone(),
        };
        let mut dptr = MaybeUninit::uninit();
        let code =
            unsafe { cuda_bindings::cuMemHostGetDevicePointer_v2(dptr.as_mut_ptr(), host as _, 0) };
        assert_eq!(code, 0, "cuMemHostGetDevicePointer failed: {code}");
        // CUDA driver flag bindings have platform-dependent integer types, so
        // the FFI call casts the flag as `_`.
        let code = unsafe {
            cuda_bindings::cuStreamWaitValue32_v2(
                stream.cu_stream(),
                dptr.assume_init(),
                1,
                cuda_bindings::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ as _,
            )
        };
        assert_eq!(code, 0, "cuStreamWaitValue32 failed: {code}");
        gate
    }

    /// Releases the work queued behind the gate. Idempotent; dropping the
    /// gate also opens it.
    fn open(&self) {
        unsafe { AtomicU32::from_ptr(self.host) }.store(1, Ordering::SeqCst);
    }
}

impl Drop for Gate {
    /// Opens the gate and drains the gated stream before freeing the word,
    /// also when an assertion failed with the gate still closed.
    fn drop(&mut self) {
        self.open();
        let synchronized = self.stream.synchronize();
        let code = unsafe { cuda_bindings::cuMemFreeHost(self.host as _) };
        if !std::thread::panicking() {
            synchronized.expect("gated stream failed");
            assert_eq!(code, 0, "cuMemFreeHost failed: {code}");
        }
    }
}

/// The streams, buffers and kernel each test in this file sets up.
struct Fixture {
    /// Holds the conflicting write access to `m_in`.
    a: Arc<Stream>,
    /// The gated stream every launch under test is scheduled on.
    b: Arc<Stream>,
    /// The bystander stream that probes the buffers' leases.
    c: Arc<Stream>,
    /// First launch argument (write); acquired first.
    x: DeviceBuffer<u32>,
    /// Second launch argument (read); the acquisition the rejection tests
    /// make conflict.
    m_in: DeviceBuffer<u32>,
    /// The no-op kernel from [`PTX`].
    func: Arc<CudaFunction>,
}

impl Fixture {
    /// Initializes device 0 on this thread and loads the kernel. Module
    /// loading has to happen here, before any gate is armed: it would block
    /// behind the gated stream otherwise.
    fn new() -> Self {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let device = with_device(0, |device| device.clone()).expect("device context");
        device.bind_to_thread().expect("bind_to_thread failed");
        let a = device.new_stream().expect("stream A");
        let b = device.new_stream().expect("stream B");
        let c = device.new_stream().expect("stream C");
        let x = DeviceBuffer::<u32>::zeroed(&a, 1024).expect("alloc x");
        let m_in = DeviceBuffer::<u32>::zeroed(&a, 1024).expect("alloc m_in");
        a.synchronize().expect("synchronize A");
        let module = load_module_from_ptx(PTX, 0).expect("load PTX");
        let func = Arc::new(module.load_function("write_first").expect("load kernel"));
        Self {
            a,
            b,
            c,
            x,
            m_in,
            func,
        }
    }

    /// A launch writing `x` and then reading `m_in`, in that acquisition
    /// order, so a conflict on `m_in` is found after `x` was acquired.
    fn launch(&self) -> AsyncKernelLaunch<'static> {
        let mut builder = AsyncKernelLaunchBuilder::new(self.func.clone());
        builder
            .push_arg(self.x.cu_deviceptr())
            .push_arg(self.m_in.cu_deviceptr());
        builder.retain_buffer(self.x.storage().clone(), true);
        builder.retain_buffer(self.m_in.storage().clone(), false);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: both arguments are device pointers matching the kernel's
        // two `.u64` parameters, and both buffers are retained.
        unsafe { builder.finalize_unchecked(cfg) }
    }

    /// Runs `op` on the gated stream B while `m_in` is written on A, and
    /// checks that the rejection left `x` free for other streams while B's
    /// unrelated work is still queued.
    fn assert_rejection_releases_x(&self, op: impl DeviceOp<Output = ()>) {
        let a_write = self
            .m_in
            .storage()
            .acquire(&self.a, true, false)
            .expect("write access to m_in on A");
        let gate = Gate::arm(&self.b);

        let policy = SingleStream::from_stream(self.b.clone());
        let mut future = op.schedule(&policy).expect("schedule on B");
        let Poll::Ready(result) = poll_once(&mut future) else {
            panic!("the launch must be rejected by the conflict on m_in, but it was submitted");
        };
        assert!(
            is_not_ready(&result),
            "the launch must be rejected by the conflict on m_in, got {result:?}"
        );

        // Nothing touching `x` was enqueued, so its lease must be gone even
        // though B is still blocked behind the gate.
        let read = self.x.storage().acquire(&self.c, false, false);
        assert!(
            read.is_ok(),
            "x is still leased by the rejected launch: read on C got {:?}",
            read.as_ref().err()
        );
        drop(read);
        self.x
            .storage()
            .acquire(&self.c, true, false)
            .expect("x is still leased by the rejected launch: write on C failed");

        // The genuine conflict is untouched: m_in stays excluded until the
        // write on A is released.
        assert!(
            self.m_in.storage().acquire(&self.c, false, false).is_err(),
            "m_in must stay excluded while A writes it"
        );
        drop(a_write);
        self.m_in
            .storage()
            .acquire(&self.c, true, false)
            .expect("m_in must be free once the write on A is released");

        drop(gate);
    }

    /// Runs `op` on the gated stream B with no conflicting access, and checks
    /// that the submitted launch keeps `x` and `m_in` excluded from other
    /// streams until B drains, then releases them.
    fn assert_acceptance_keeps_leases(&self, op: impl DeviceOp<Output = ()>) {
        let gate = Gate::arm(&self.b);

        let policy = SingleStream::from_stream(self.b.clone());
        let mut future = op.schedule(&policy).expect("schedule on B");
        if let Poll::Ready(result) = poll_once(&mut future) {
            panic!("the launch must be submitted and wait behind the gate, got {result:?}");
        }

        // The launch is queued behind the gate and writes `x`: no other
        // stream may read or write it, nor write `m_in`, until it completes.
        assert!(
            self.x
                .storage()
                .acquire(&self.c, false, false)
                .is_err_and(|error| is_conflict(&error)),
            "x must stay leased by the submitted launch: read on C was not rejected"
        );
        assert!(
            self.x
                .storage()
                .acquire(&self.c, true, false)
                .is_err_and(|error| is_conflict(&error)),
            "x must stay leased by the submitted launch: write on C was not rejected"
        );
        assert!(
            self.m_in
                .storage()
                .acquire(&self.c, true, false)
                .is_err_and(|error| is_conflict(&error)),
            "m_in must stay leased by the submitted launch: write on C was not rejected"
        );

        gate.open();
        futures::executor::block_on(future).expect("launch on B failed");
        self.x
            .storage()
            .acquire(&self.c, true, false)
            .expect("x must be free once the launch completed");
        self.m_in
            .storage()
            .acquire(&self.c, true, false)
            .expect("m_in must be free once the launch completed");

        drop(gate);
    }
}

/// A kernel launch rejected on its second buffer argument releases the lease
/// it already took on its first one, without waiting for its stream.
#[test]
fn launch_rejected_on_second_argument_releases_first_lease() {
    let _serial = GATED_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    on_fresh_thread(|| {
        let fixture = Fixture::new();
        fixture.assert_rejection_releases_x(fixture.launch());
    });
}

/// A graph replay rejected while reacquiring its second captured buffer
/// releases the replay lease on the first one, without waiting for its
/// stream.
#[test]
fn graph_replay_rejected_on_second_buffer_releases_first_lease() {
    let _serial = GATED_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    on_fresh_thread(|| {
        let fixture = Fixture::new();
        let capture_stream = with_device(0, |device| device.new_stream())
            .expect("device context")
            .expect("capture stream");
        let graph = CudaGraph::capture(capture_stream, fixture.launch()).expect("capture launch");
        fixture.assert_rejection_releases_x(graph.launch());
    });
}

/// An accepted kernel launch keeps the leases on all of its buffer arguments
/// until it completes.
#[test]
fn accepted_launch_keeps_argument_leases_until_complete() {
    let _serial = GATED_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    on_fresh_thread(|| {
        let fixture = Fixture::new();
        fixture.assert_acceptance_keeps_leases(fixture.launch());
    });
}

/// An accepted graph replay keeps the leases on all of its captured buffers
/// until it completes.
#[test]
fn accepted_graph_replay_keeps_buffer_leases_until_complete() {
    let _serial = GATED_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    on_fresh_thread(|| {
        let fixture = Fixture::new();
        let capture_stream = with_device(0, |device| device.new_stream())
            .expect("device context")
            .expect("capture stream");
        let graph = CudaGraph::capture(capture_stream, fixture.launch()).expect("capture launch");
        fixture.assert_acceptance_keeps_leases(graph.launch());
    });
}
