/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! An unrelated error pending in the context's sticky error slot must not
//! turn the drop of an in-flight `DeviceFuture` into a fault.
//!
//! [`Context::record_err`] parks driver errors from `Drop` paths for the
//! caller's next [`Context::check_err`] or [`Context::bind_to_thread`]. A
//! dropped submission used to bind its context with `bind_to_thread`, which
//! takes that slot: the unrelated error then failed the bind, was reported
//! as a stream fault, and the submission leaked every owner and access lease
//! (memory never freed, write exclusion held forever) while silently
//! consuming the error it had taken.
//!
//! Each test holds a write lease and a sentinel owner in a submission whose
//! work waits behind a host-controlled [`Gate`], records an unrelated error,
//! drops the future, and checks that the owners are still retired, the
//! buffer is usable from another stream, and the error is still pending.
//! Requires a GPU.
//!
//! The tests share the process-wide device context, and therefore its one
//! error slot, so they live in their own test binary and run one at a time
//! under [`TEST_LOCK`].

use cuda_async::device_context::{global_policy, init_device_contexts, with_device};
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::{Context as CudaContext, DeviceBuffer, DriverError, RawDeviceBuffer, Stream};
use std::future::{Future, IntoFuture};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

/// Serializes the tests in this binary: they inject into, and inspect, the
/// error slot of the one context every test thread shares.
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// How long retirement may take once the device has finished the work. The
/// retirement worker polls every millisecond; this is generous headroom for
/// a loaded machine, and a leak never finishes at all.
const RETIRE_DEADLINE: Duration = Duration::from_secs(5);

/// Bytes the op writes behind its gate.
const BUF: usize = 1 << 20;

/// Takes [`TEST_LOCK`], ignoring poison so one failed test does not mask the
/// result of the other.
fn lock_test() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A waker that does nothing; the tests poll by hand.
fn noop_waker() -> Waker {
    /// Wake, wake-by-ref and drop: there is no task to wake and no state to
    /// release.
    fn noop(_: *const ()) {}
    /// Clones the waker by handing out the same (null) data pointer.
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    /// The vtable shared by every clone of the no-op waker.
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    // SAFETY: every vtable function ignores the data pointer, so the null
    // pointer satisfies the `RawWaker` contract.
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

/// The context the global device registry hands out for device 0, bound to
/// the calling thread.
fn device_context() -> Arc<CudaContext> {
    let ctx = with_device(0, Arc::clone).expect("device context");
    ctx.bind_to_thread().expect("bind_to_thread failed");
    ctx
}

/// A host-controlled barrier at the head of a stream.
///
/// [`arm`](Gate::arm) enqueues a `cuStreamWaitValue32` on a word of pinned,
/// device-mapped host memory; work submitted to the stream after that stays
/// queued until [`open`](Gate::open) stores the release value. This holds a
/// submission provably in flight for as long as a test needs.
struct Gate {
    /// Host address of the pinned word; 0 keeps the gate closed, 1 opens it.
    host: *mut u32,
    /// Device address of the same word, which `cuStreamWaitValue32` polls.
    dptr: cuda_bindings::CUdeviceptr,
}

// SAFETY: `host` is pinned, device-mapped memory owned by the gate for its
// whole lifetime; the host side only touches it through atomics.
unsafe impl Send for Gate {}
unsafe impl Sync for Gate {}

impl Gate {
    /// Allocates a closed gate. Requires a current CUDA context on the
    /// calling thread.
    fn closed() -> Arc<Self> {
        let mut host = MaybeUninit::uninit();
        let flags =
            cuda_bindings::CU_MEMHOSTALLOC_PORTABLE | cuda_bindings::CU_MEMHOSTALLOC_DEVICEMAP;
        // SAFETY: the out-pointer is valid for writes; a context is current.
        let code = unsafe {
            cuda_bindings::cuMemHostAlloc(host.as_mut_ptr(), std::mem::size_of::<u32>(), flags)
        };
        assert_eq!(code, 0, "cuMemHostAlloc failed: {code}");
        // SAFETY: the allocation succeeded, so the driver wrote the pointer.
        let host = unsafe { host.assume_init() } as *mut u32;
        // SAFETY: `host` is a live, `u32`-aligned pinned allocation, and every
        // later host access to it goes through atomics.
        unsafe { AtomicU32::from_ptr(host) }.store(0, Ordering::SeqCst);
        let mut dptr = MaybeUninit::uninit();
        // SAFETY: `host` was allocated with `CU_MEMHOSTALLOC_DEVICEMAP`, and the
        // out-pointer is valid for writes.
        let code =
            unsafe { cuda_bindings::cuMemHostGetDevicePointer_v2(dptr.as_mut_ptr(), host as _, 0) };
        assert_eq!(code, 0, "cuMemHostGetDevicePointer failed: {code}");
        Arc::new(Self {
            host,
            // SAFETY: the lookup succeeded, so the driver wrote the pointer.
            dptr: unsafe { dptr.assume_init() },
        })
    }

    /// Enqueues the wait: everything submitted to `stream` after this call
    /// stays queued until the gate is opened.
    fn arm(&self, stream: cuda_bindings::CUstream) -> Result<(), DeviceError> {
        // CUDA driver flag bindings have platform-dependent integer types, so
        // the FFI call casts the flag as `_`.
        // SAFETY: `dptr` maps the gate's live pinned word, and the caller
        // passes a valid stream on the current context.
        let code = unsafe {
            cuda_bindings::cuStreamWaitValue32_v2(
                stream,
                self.dptr,
                1,
                cuda_bindings::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ as _,
            )
        };
        if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(DeviceError::Internal(format!(
                "cuStreamWaitValue32 failed: {code}"
            )));
        }
        Ok(())
    }

    /// Releases every stream waiting on the gate.
    fn open(&self) {
        // SAFETY: see `closed`; the word lives as long as the gate.
        unsafe { AtomicU32::from_ptr(self.host) }.store(1, Ordering::SeqCst);
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        // The tests keep their own handle until every wait on the gate has
        // passed and the error slot has been inspected, so binding here (which
        // takes the slot) cannot disturb an assertion.
        let _ = with_device(0, |device| device.bind_to_thread());
        // SAFETY: the gate owns `host`, and no stream waits on it any more:
        // every submission that retained the gate has completed.
        let code = unsafe { cuda_bindings::cuMemFreeHost(self.host as _) };
        if !std::thread::panicking() {
            assert_eq!(code, 0, "cuMemFreeHost failed: {code}");
        }
    }
}

/// Retains a sentinel owner and a write lease on `storage` in its
/// submission, then memsets the buffer behind `gate`.
struct LeaseOp {
    /// The buffer the op writes; its write lease is retained in the submission.
    storage: RawDeviceBuffer,
    /// An owner whose release the tests observe through a `Weak`.
    sentinel: Arc<()>,
    /// Holds the memset in flight until the test opens it.
    gate: Arc<Gate>,
    /// Receives the stream the op was scheduled on.
    stream_out: Arc<Mutex<Option<Arc<Stream>>>>,
}

impl DeviceOp for LeaseOp {
    type Output = ();
    unsafe fn execute(self, context: &ExecutionContext) -> Result<(), DeviceError> {
        let stream = context.get_cuda_stream();
        *self.stream_out.lock().unwrap() = Some(stream.clone());
        context.retain(self.sentinel)?;
        context.retain(self.gate.clone())?;
        context.retain_buffer(self.storage.clone(), true)?;
        self.gate.arm(stream.cu_stream())?;
        // SAFETY: the buffer is live and its write lease is retained in this
        // submission until the memset completes; the stream's context is
        // current inside `execute`.
        let code = cuda_bindings::cuMemsetD8Async(
            self.storage.cu_deviceptr(),
            0x5A,
            self.storage.len_bytes(),
            stream.cu_stream(),
        );
        if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(DeviceError::Internal(format!(
                "cuMemsetD8Async failed: {code}"
            )));
        }
        Ok(())
    }
}

impl IntoFuture for LeaseOp {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), LeaseOp>;
    fn into_future(self) -> Self::IntoFuture {
        let policy = global_policy(0).expect("global policy");
        match self.schedule(&policy) {
            Ok(future) => future,
            Err(error) => DeviceFuture::failed(error),
        }
    }
}

/// Whether the submission's sentinel owner was released before `deadline`.
fn released_within(weak: &Weak<()>, deadline: Duration) -> bool {
    let end = Instant::now() + deadline;
    while weak.upgrade().is_some() {
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// Reads `dev` on a fresh stream, retrying while its write exclusion is still
/// held (`CUDA_ERROR_NOT_READY`) until `deadline`.
///
/// Must run after the error slot has been inspected: the read binds the
/// context, which takes any error still pending there.
fn read_on_other_stream(
    ctx: &Arc<CudaContext>,
    dev: &DeviceBuffer<u8>,
    deadline: Duration,
) -> Result<Vec<u8>, DriverError> {
    let end = Instant::now() + deadline;
    loop {
        let stream = ctx.new_stream()?;
        match dev.to_host_vec(&stream) {
            Err(DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY))
                if Instant::now() < end =>
            {
                std::thread::sleep(Duration::from_millis(5));
            }
            other => return other,
        }
    }
}

/// Polls a gated [`LeaseOp`] once, so its submission is in flight, records an
/// unrelated error in the context's error slot, and drops the future. With
/// `busy`, the gate is still closed at the drop and opens right after it;
/// otherwise the work has completed, and the reactor has retired its slot,
/// before the error is recorded.
///
/// Then checks that the submission's owners and write lease were retired and
/// that the unrelated error is still pending for its owner.
fn drop_with_pending_error(busy: bool) {
    let _guard = lock_test();
    init_device_contexts(0, 1).expect("init failed (requires GPU)");
    let ctx = device_context();
    ctx.check_err()
        .expect("the error slot must start empty for this test");

    let alloc_stream = ctx.new_stream().expect("failed to create CUDA stream");
    let dev = DeviceBuffer::<u8>::zeroed(&alloc_stream, BUF).expect("failed to allocate");
    alloc_stream
        .synchronize()
        .expect("allocation stream failed");

    let sentinel = Arc::new(());
    let weak = Arc::downgrade(&sentinel);
    let gate = Gate::closed();
    let stream_out = Arc::new(Mutex::new(None));
    let mut future = LeaseOp {
        storage: dev.storage().clone(),
        sentinel,
        gate: Arc::clone(&gate),
        stream_out: Arc::clone(&stream_out),
    }
    .into_future();

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    match Pin::new(&mut future).poll(&mut cx) {
        Poll::Pending => {}
        Poll::Ready(result) => {
            panic!("the first poll resolved with the gate closed: {result:?}")
        }
    }
    let op_stream = stream_out
        .lock()
        .unwrap()
        .clone()
        .expect("the op did not execute on the first poll");

    if !busy {
        gate.open();
        op_stream.synchronize().expect("gated work failed");
        // Let the reactor retire the completion slot, so the drop below is
        // the only path that can observe the injected error.
        std::thread::sleep(Duration::from_millis(100));
    }

    let injected = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
    ctx.record_err::<()>(Err(injected));
    drop(future);
    if busy {
        gate.open();
    }

    let released = released_within(&weak, RETIRE_DEADLINE);
    // Inspect the slot before anything below binds the context again.
    let pending = ctx.check_err();
    let read = read_on_other_stream(&ctx, &dev, RETIRE_DEADLINE);
    let observed = format!(
        "owners released: {released}, pending error: {pending:?}, read: {:?}",
        read.as_ref().map(|bytes| bytes.first().copied())
    );

    assert!(
        released,
        "the dropped submission leaked its owners: an unrelated pending context error was \
         treated as a stream fault ({observed})"
    );
    assert_eq!(
        pending,
        Err(injected),
        "dropping the submission consumed an unrelated pending context error ({observed})"
    );
    match read {
        Ok(bytes) => assert!(
            bytes.iter().all(|&b| b == 0x5A),
            "the gated memset did not complete before the buffer was read ({observed})"
        ),
        Err(error) => panic!(
            "the buffer is still excluded after the dropped submission retired: {error:?} \
             ({observed})"
        ),
    }
    drop(op_stream);
    drop(gate);
}

/// The submission's work has completed when the future drops: the stream is
/// idle, so its owners must be released on the spot, and the pending error
/// must stay in the slot.
#[test]
fn idle_submission_drop_preserves_leases_and_pending_error() {
    drop_with_pending_error(false);
}

/// The submission's work is still gated when the future drops: its owners
/// must be handed to completion-bound retirement and released once the gate
/// opens, and the pending error must stay in the slot.
#[test]
fn busy_submission_drop_preserves_leases_and_pending_error() {
    drop_with_pending_error(true);
}
