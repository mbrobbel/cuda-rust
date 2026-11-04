/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{Context, DeviceBuffer, RawDeviceBuffer};
use std::sync::{mpsc, Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// Serializes the tests that use the GPU.
///
/// Several of them park a stream in a host callback or park the shared
/// retirement worker, and others assert that drops do not block or that
/// retirement catches up. Run concurrently, a parked callback or worker
/// stalls the others' driver calls and retirements until a timeout fires.
static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Locks [`GPU_TEST_LOCK`], ignoring poisoning by a failed test, once the
/// GPU is quiet enough for a test to park a stream.
///
/// The first memset, event or copy in a context can load driver internals
/// lazily, and a synchronous free waits for the whole device. Either blocks
/// other driver calls while a host callback is parked, so a test whose gate
/// is armed would stall inside its own driver call until the gate times out
/// and the gated work has already run. The primary context is therefore
/// warmed up once and kept retained, and the frees that earlier tests queued
/// on the retirement worker are drained before each test starts.
fn serialize() -> MutexGuard<'static, ()> {
    static WARM: OnceLock<Arc<cuda_core::Stream>> = OnceLock::new();
    let guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stream = WARM.get_or_init(|| {
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let other = context.new_stream().unwrap();
        let mut buffer = DeviceBuffer::from_host(&stream, &[0u32; 32]).unwrap();
        buffer.zero_async(&stream).unwrap();
        other.join(&stream).unwrap();
        assert_eq!(buffer.to_host_vec(&other).unwrap(), vec![0; 32]);
        other.synchronize().unwrap();
        stream
    });
    stream.synchronize().unwrap();
    // The worker runs every ready item it holds in one pass before it takes
    // new ones, so once a second marker has run, everything queued before the
    // first one has run too.
    for _ in 0..2 {
        let (ran_tx, ran_rx) = mpsc::channel::<()>();
        cuda_core::completion::retain_until_complete(Signal(ran_tx), stream).unwrap();
        ran_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the retirement worker did not drain");
    }
    guard
}

/// Signals its channel when dropped, telling a waiting test that the
/// retirement worker reached it.
struct Signal(mpsc::Sender<()>);

impl Drop for Signal {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

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
    let _guard = serialize();
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

#[test]
fn leases_retained_together_release_on_one_completion() {
    let _guard = serialize();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let first = DeviceBuffer::from_host(&stream, &[1u32; 32]).unwrap();
    let second = DeviceBuffer::from_host(&stream, &[2u32; 32]).unwrap();
    let leases = vec![
        first.storage().acquire(&stream, true, false).unwrap(),
        second.storage().acquire(&stream, false, false).unwrap(),
    ];
    assert!(first.storage().acquire(&other, false, false).is_err());
    assert!(second.storage().acquire(&other, false, false).is_ok());
    assert!(second.storage().acquire(&other, true, false).is_err());
    cuda_core::AllocationLease::retain_all_until_complete(leases, &stream).unwrap();
    stream.synchronize().unwrap();
    // The shared event marks both accesses complete, so a synchronized stream
    // releases them without waiting for the retirement worker.
    assert!(first.storage().acquire(&other, true, false).is_ok());
    assert!(second.storage().acquire(&other, true, false).is_ok());
    cuda_core::AllocationLease::retain_all_until_complete(Vec::new(), &stream).unwrap();
}

#[test]
fn completed_lease_awaiting_retirement_does_not_block_detach() {
    let _guard = serialize();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let buffer = DeviceBuffer::from_host(&stream, &[3u32; 32]).unwrap();
    // Park the retirement worker inside a cleanup so the lease below stays
    // queued behind it after its work completed, as it would after a
    // `synchronize` that outruns the worker's polling.
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    struct Parked(mpsc::Sender<()>, mpsc::Receiver<()>);
    impl Drop for Parked {
        fn drop(&mut self) {
            let _ = self.0.send(());
            let _ = self.1.recv_timeout(Duration::from_secs(10));
        }
    }
    cuda_core::completion::retain_until_complete(Parked(entered_tx, release_rx), &stream).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let lease = buffer.storage().acquire(&stream, true, false).unwrap();
    lease.retain_until_complete(&stream).unwrap();
    stream.synchronize().unwrap();

    let (ptr, len, owner) = buffer.into_raw_parts();
    drop(release_tx);
    // SAFETY: the raw parts came from a synchronous allocation of `len` u32
    // values whose only access completed above; ownership returns here once,
    // and the retiring lease must not free the pointer a second time.
    let buffer = unsafe { DeviceBuffer::<u32>::from_raw_parts(ptr, len, owner) };
    assert_eq!(buffer.to_host_vec(&stream).unwrap(), vec![3; 32]);
}

#[test]
#[should_panic(expected = "cannot detach a shared or in-flight allocation")]
fn running_access_still_blocks_detach() {
    let _guard = serialize();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let buffer = DeviceBuffer::from_host(&stream, &[3u32; 32]).unwrap();
    let _lease = buffer.storage().acquire(&stream, true, false).unwrap();
    let _ = buffer.into_raw_parts();
}

#[test]
fn ordered_acquire_waits_for_recorded_access_on_other_stream() {
    let _guard = serialize();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let mut buffer = DeviceBuffer::from_host(&stream, &[7u32; 32]).unwrap();
    let storage = buffer.storage().clone();
    let gate = Gate::arm(&stream);
    buffer.zero_async(&stream).unwrap();
    // A plain acquire still rejects the pending write...
    assert!(storage.acquire(&other, false, false).is_err());
    // ...but its completion is recorded, so an ordered acquire makes `other`
    // wait for it on the GPU instead of failing.
    let lease = storage.acquire_ordered(&other, false).unwrap();
    let marker = other.record_event(None).unwrap();
    assert!(
        !marker.query().unwrap(),
        "work after an ordered acquire ran before the gated write"
    );
    drop(gate);
    lease.retain_until_complete(&other).unwrap();
    other.synchronize().unwrap();
    assert!(marker.query().unwrap());
    assert_eq!(buffer.to_host_vec(&other).unwrap(), vec![0; 32]);
}

#[test]
fn ordered_acquire_rejects_access_without_recorded_completion() {
    let _guard = serialize();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let buffer = DeviceBuffer::from_host(&stream, &[7u32; 32]).unwrap();
    // A lease whose submission has not recorded completion yet cannot be
    // waited on, so the conflict is still reported.
    let pending = buffer.storage().acquire(&stream, true, false).unwrap();
    let not_ready = cuda_core::DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_READY);
    assert_eq!(
        buffer.storage().acquire_ordered(&other, false).err(),
        Some(not_ready)
    );
    // The same stream is ordered by CUDA and never conflicts.
    drop(buffer.storage().acquire_ordered(&stream, true).unwrap());
    drop(pending);
    drop(buffer.storage().acquire_ordered(&other, true).unwrap());
}

#[test]
fn tracked_copies_after_join_wait_instead_of_failing() {
    let _guard = serialize();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let mut buffer = DeviceBuffer::from_host(&stream, &[7u32; 32]).unwrap();
    stream.synchronize().unwrap();

    // Fork/join: produce on `stream`, join, then copy on `other` while the
    // producer is still pending. The join already orders the copy; the
    // tracker cannot see it, so the copy must wait rather than fail. The
    // copy blocks the host until it ran, so another thread opens the gate.
    let gate = Gate::arm(&stream);
    buffer.zero_async(&stream).unwrap();
    other.join(&stream).unwrap();
    let opener = open_later(gate);
    buffer.copy_from_host(&other, &[9u32; 32]).unwrap();
    opener.join().unwrap();
    assert_eq!(buffer.to_host_vec(&other).unwrap(), vec![9; 32]);

    // Without the join, the read is still ordered after the pending fill.
    let gate = Gate::arm(&stream);
    buffer.zero_async(&stream).unwrap();
    let opener = open_later(gate);
    assert_eq!(buffer.to_host_vec(&other).unwrap(), vec![0; 32]);
    opener.join().unwrap();
}

/// Drops `gate` on another thread after a short delay, for a test thread
/// that is about to block on work queued behind it.
fn open_later(gate: Gate) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        drop(gate);
    })
}

#[test]
fn stream_reports_graph_capture() {
    let _guard = serialize();
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    assert!(!stream.is_capturing().unwrap());
    context.bind_to_thread().unwrap();
    // SAFETY: the context is current and the fresh stream is not capturing.
    unsafe {
        stream
            .begin_capture(cuda_core::sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
    }
    assert!(stream.is_capturing().unwrap());
    // SAFETY: the capture was begun above on this stream.
    let graph = unsafe { stream.end_capture() }.unwrap();
    assert!(!stream.is_capturing().unwrap());
    // SAFETY: the graph returned by `end_capture` is owned here, never used
    // again, and destroyed once.
    let destroyed = unsafe { cuda_core::sys::cuGraphDestroy(graph) };
    assert_eq!(destroyed, cuda_core::sys::cudaError_enum_CUDA_SUCCESS);
}

#[test]
fn stream_drop_is_nonblocking_and_holds_context_until_completion() {
    let _guard = serialize();
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
    let _guard = serialize();
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
    let _guard = serialize();
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
