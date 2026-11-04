/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{Context, DeviceAllocation, DeviceBuffer, DriverError, RawDeviceBuffer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

fn metadata(context: usize) -> RawDeviceBuffer {
    // SAFETY: these fake handles are used only for bookkeeping; no CUDA API
    // sees them and borrowed allocation/context destruction performs no calls.
    unsafe {
        RawDeviceBuffer::borrowed_from_raw_parts(
            0x1000,
            128,
            Context::borrow_raw(context as *mut _, 0, 0),
        )
    }
}

#[test]
fn byte_views_share_extent_identity_and_access_exclusion() {
    let raw = metadata(0x2000);
    let view = raw.slice_bytes(16, 32).unwrap();
    assert_eq!(view.cu_deviceptr(), raw.cu_deviceptr() + 16);
    assert_eq!(view.allocation_id(), raw.allocation_id());
    assert!(raw.slice_bytes(120, 9).is_err());
    assert!(raw.slice_bytes(usize::MAX, 1).is_err());
    assert!(!raw.is_unique());
    let read = raw.acquire_on_stream(1, false, false).unwrap();
    let second_read = view.acquire_on_stream(2, false, false).unwrap();
    assert!(view.acquire_on_stream(2, true, false).is_err());
    drop((read, second_read));
    let write = view.acquire_on_stream(2, true, false).unwrap();
    assert!(raw.acquire_on_stream(1, false, false).is_err());
    assert!(raw.acquire_on_stream(2, true, false).is_ok());
    drop(view);
    assert!(raw.is_unique(), "completion leases are not user aliases");
    drop(write);
}

#[test]
fn same_ordinal_does_not_mean_same_context() {
    let raw = metadata(0x2000);
    let other = metadata(0x3000);
    assert_eq!(raw.device_id(), other.device_id());
    assert_ne!(raw.context(), other.context());
    // SAFETY: this fake borrowed stream is never submitted or destroyed: forget
    // it after testing the context check, before Drop could query the driver.
    let stream = unsafe { cuda_core::Stream::borrow_raw(0x4000 as *mut _, other.context()) };
    let result = raw.acquire(&stream, false, false);
    std::mem::forget(stream);
    assert!(result.is_err());
}

/// Serializes the tests in this file that use the GPU, since some of them park
/// the driver's host-callback thread with a [`Gate`].
///
/// A parked callback stalls other tests' callbacks and blocking driver calls
/// (allocation, synchronization, context setup) in this process, so a test
/// running beside a gated one can make it outlast its own gate.
static GATED_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquires [`GATED_TEST_LOCK`], ignoring poison from an earlier failed test.
fn lock_gated_test() -> MutexGuard<'static, ()> {
    GATED_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct Gate(mpsc::Sender<()>);
impl Gate {
    fn arm(stream: &cuda_core::Stream) -> Self {
        let (tx, rx) = mpsc::channel();
        stream
            .launch_host_function(move || {
                let _ = rx.recv_timeout(Duration::from_secs(10));
            })
            .unwrap();
        Self(tx)
    }
}
impl Drop for Gate {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

#[test]
fn pending_copy_excludes_other_stream_and_outlives_dropped_buffer() {
    let _gated_test_guard = lock_gated_test();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let mut buffer = DeviceBuffer::from_host(&stream, &[7u32; 32]).unwrap();
    let storage = buffer.storage().clone();
    let gate = Gate::arm(&stream);
    buffer.zero_async(&stream).unwrap();
    assert!(storage.acquire(&other, false, false).is_err());
    let before = Instant::now();
    drop(buffer);
    assert!(
        before.elapsed() < Duration::from_secs(2),
        "buffer drop waited for GPU work"
    );
    assert!(storage.acquire(&other, false, false).is_err());
    drop(gate);
    stream.synchronize().unwrap();
    // SAFETY: the zero fill completed, u32 accepts every representation, and
    // this test is the only remaining user of the allocation.
    let buffer = unsafe { DeviceBuffer::<u32>::from_storage_copy(storage, 32).unwrap() };
    assert_eq!(buffer.to_host_vec(&other).unwrap(), vec![0; 32]);
}

/// A foreign allocation that reports when cuda-core releases its last owner.
///
/// Leaked completion leases keep the owner alive forever, so the flag is the
/// observable difference between retirement and a leak.
struct FlaggedOwner {
    buffer: DeviceBuffer<u8>,
    context: Arc<Context>,
    dropped: Arc<AtomicBool>,
}

impl Drop for FlaggedOwner {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

// SAFETY: the owned buffer keeps its pointer and extent valid for the owner's
// lifetime, and `context` is the context the buffer was allocated in.
unsafe impl DeviceAllocation for FlaggedOwner {
    fn device_ptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.buffer.cu_deviceptr()
    }
    fn len_bytes(&self) -> usize {
        self.buffer.num_bytes()
    }
    fn context(&self) -> Arc<Context> {
        self.context.clone()
    }
}

/// Polls `done` until it holds or a generous retirement deadline passes.
///
/// This only bounds eventual cleanup by the retirement worker; it does not
/// assert anything about how quickly that cleanup happens.
fn wait_for_retirement(mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    done()
}

/// An unrelated error left pending in the context's error slot (for example
/// by a destructor on another thread) must neither make a lease whose
/// completion is observable leak its memory and exclusion, nor be consumed by
/// retirement: it stays pending for the context's next checked call.
#[test]
fn pending_context_error_does_not_leak_retained_lease() {
    let _gated_test_guard = lock_gated_test();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let owner = FlaggedOwner {
        buffer: DeviceBuffer::zeroed(&stream, 256).unwrap(),
        context: context.clone(),
        dropped: dropped.clone(),
    };
    let raw = RawDeviceBuffer::foreign(Arc::new(owner), 256);
    let gate = Gate::arm(&stream);
    let lease = raw.acquire(&stream, true, false).unwrap();

    let injected = DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
    context.record_err::<()>(Err(injected));
    let retained = lease.retain_until_complete(&stream);
    assert!(
        raw.acquire(&other, false, false).is_err(),
        "the retained write must exclude other streams until completion"
    );

    // Observe completion without a checked call (such as a stream
    // synchronize), so the injected error stays pending throughout retirement.
    drop(gate);
    let mut released: Result<(), DriverError> = Ok(());
    assert!(
        wait_for_retirement(|| {
            released = raw.acquire(&other, true, false).map(drop);
            released.is_ok()
        }),
        "completed write must not exclude other streams: {released:?}"
    );
    drop(raw);
    assert!(
        wait_for_retirement(|| dropped.load(Ordering::SeqCst)),
        "retirement leaked the allocation owner of a completed lease"
    );
    assert_eq!(
        retained,
        Ok(()),
        "an unrelated pending error must not fail retirement"
    );
    assert_eq!(
        context.check_err(),
        Err(injected),
        "retirement consumed an unrelated pending context error"
    );
}

/// Same hazard for owners retained through the public completion helper,
/// which backs `DeviceBuffer::drop_async`, `Stream` drop, and cuda-async.
#[test]
fn pending_context_error_does_not_leak_completion_owner() {
    let _gated_test_guard = lock_gated_test();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let owner = Arc::new(());
    let weak: Weak<()> = Arc::downgrade(&owner);
    let gate = Gate::arm(&stream);

    let injected = DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
    context.record_err::<()>(Err(injected));
    let retained = cuda_core::completion::retain_until_complete(owner, &stream);
    assert!(
        weak.upgrade().is_some(),
        "the owner must outlive the work queued before it"
    );

    drop(gate);
    assert!(
        wait_for_retirement(|| weak.upgrade().is_none()),
        "retirement leaked an owner whose stream completed"
    );
    assert_eq!(
        retained,
        Ok(()),
        "an unrelated pending error must not fail retirement"
    );
    assert_eq!(
        context.check_err(),
        Err(injected),
        "retirement consumed an unrelated pending context error"
    );
}

#[test]
fn stream_drop_is_nonblocking_and_holds_context_until_completion() {
    let _gated_test_guard = lock_gated_test();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let gate = Gate::arm(&stream);
    let weak = Arc::downgrade(&context);
    drop(context);
    let before = Instant::now();
    drop(stream);
    assert!(
        before.elapsed() < Duration::from_secs(2),
        "stream drop waited for GPU work"
    );
    assert!(weak.upgrade().is_some());
    drop(gate);
    let deadline = Instant::now() + Duration::from_secs(5);
    while weak.upgrade().is_some() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        weak.upgrade().is_none(),
        "retirement leaked an idle stream/context"
    );
}

#[test]
fn fresh_thread_allocation_binds_its_context() {
    let _gated_test_guard = lock_gated_test();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    std::thread::spawn(move || {
        let buffer = DeviceBuffer::from_host(&stream, &[11u32, 29]).unwrap();
        assert_eq!(buffer.to_host_vec(&stream).unwrap(), [11, 29]);
    })
    .join()
    .unwrap();
}

#[test]
fn empty_and_peeled_buffers_preserve_shared_read_access() {
    let _gated_test_guard = lock_gated_test();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let empty = DeviceBuffer::<u32>::from_host(&stream, &[]).unwrap();
    assert!(!empty.shared_access_writes());
    let zeroed_empty = DeviceBuffer::<u32>::zeroed(&stream, 0).unwrap();
    assert!(!zeroed_empty.shared_access_writes());

    let words = DeviceBuffer::from_host(&stream, &[1u32, 2, 3]).unwrap();
    let wrapped = words.wrap_elem::<std::num::Wrapping<u32>>();
    assert!(wrapped.shared_access_writes());
    let peeled = wrapped.peel_elem::<u32>();
    assert!(!peeled.shared_access_writes());
}
