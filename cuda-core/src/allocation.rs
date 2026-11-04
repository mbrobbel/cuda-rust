/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared allocation ownership, byte views, and in-flight access tracking.

use crate::{Context, DriverError, Stream};
use cuda_bindings::CUdeviceptr;
use std::sync::{Arc, Mutex};

/// A foreign allocation whose owner keeps both its memory and CUDA context alive.
///
/// # Safety
/// The pointer, byte extent, and context must remain stable and valid for the
/// owner's lifetime. The pointer must address at least `len_bytes` bytes in that
/// context. Per-operation exclusion from foreign accesses remains the obligation
/// of callers constructing typed views over this memory.
pub unsafe trait DeviceAllocation: Send + Sync + 'static {
    fn device_ptr(&self) -> CUdeviceptr;
    fn len_bytes(&self) -> usize;
    fn context(&self) -> Arc<Context>;
    fn device_id(&self) -> usize {
        self.context().ordinal()
    }
}

struct Access {
    id: u64,
    stream: usize,
    write: bool,
    recording: bool,
    completion: Option<Arc<crate::Event>>,
    /// Position of this access in the allocation's completion order: the
    /// value of [`Accesses::marks`] when its completion event was attached,
    /// or 0 while it has none.
    marked: u64,
    /// The value of [`Accesses::marks`] when this access was registered.
    /// Every access marked at or before it had its completion event recorded
    /// before this access's work was enqueued.
    seen: u64,
}

#[derive(Default)]
struct Accesses {
    next_id: u64,
    active: Vec<Access>,
    /// Number of completion events attached so far; orders [`Access::marked`]
    /// against [`Access::seen`].
    marks: u64,
    /// Ownership of the memory moved out through
    /// [`RawDeviceBuffer::into_owned_parts`] while completed leases still
    /// shared the allocation, so its destructor must not free the pointer.
    detached: bool,
}

impl Accesses {
    /// Forgets accesses whose completion event has already fired.
    ///
    /// A synchronous wait on another thread may have completed a retired
    /// operation before the worker got to its lease. Do not report a stale
    /// conflict in that case. Errors leave the exclusion intact.
    fn retire_completed(&mut self) {
        self.active
            .retain(|a| !matches!(a.completion.as_ref().map(|e| e.query()), Some(Ok(true))));
    }

    fn register(
        &mut self,
        stream: usize,
        write: bool,
        recording: bool,
    ) -> Result<u64, DriverError> {
        self.retire_completed();
        if !recording && self.conflicts(stream, write).next().is_some() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY,
            ));
        }
        self.push(stream, write, recording)
    }

    /// Registers an access that work enqueued next on `stream` will make,
    /// collecting into `waits` the completion events of conflicting accesses
    /// on other streams instead of rejecting them.
    ///
    /// Waiting on those events on `stream` orders the new access after them.
    /// A conflicting access whose completion has not been recorded yet (its
    /// work may still be in the middle of being enqueued) cannot be ordered
    /// and still rejects with `CUDA_ERROR_NOT_READY`.
    fn register_ordered(
        &mut self,
        stream: usize,
        write: bool,
        waits: &mut Vec<Arc<crate::Event>>,
    ) -> Result<u64, DriverError> {
        self.retire_completed();
        for access in self.conflicts(stream, write) {
            let Some(event) = &access.completion else {
                return Err(DriverError(
                    cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY,
                ));
            };
            if !waits.iter().any(|known| Arc::ptr_eq(known, event)) {
                waits.push(event.clone());
            }
        }
        self.push(stream, write, false)
    }

    /// Yields the live non-recording accesses on other streams that conflict
    /// with a new access on `stream`, i.e. those where either side writes.
    ///
    /// Recording (graph capture) accesses never conflict: they run only when
    /// the graph is launched, which reacquires its own accesses.
    fn conflicts(&self, stream: usize, write: bool) -> impl Iterator<Item = &Access> {
        self.active
            .iter()
            .filter(move |a| !a.recording && a.stream != stream && (write || a.write))
    }

    /// Records a new access and returns its id.
    ///
    /// Callers have already decided that the access may coexist with every
    /// live one; this only allocates the id and appends the entry.
    fn push(&mut self, stream: usize, write: bool, recording: bool) -> Result<u64, DriverError> {
        let id = self.next_id.checked_add(1).ok_or(DriverError(
            cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
        ))?;
        self.next_id = id;
        self.active.push(Access {
            id,
            stream,
            write,
            recording,
            completion: None,
            marked: 0,
            seen: self.marks,
        });
        Ok(id)
    }

    /// Attaches the completion event of access `id` and drops the accesses
    /// that event already covers.
    ///
    /// `event` was recorded on the access's stream after its work, so when it
    /// completes so has every access whose own completion was recorded
    /// before this access was registered (`marked <= seen`) on the same
    /// stream, and every conflicting one on another stream: it was either
    /// waited for by [`register_ordered`](Self::register_ordered) or would
    /// have rejected the registration. Such an access is redundant once its
    /// exclusion is no wider than this one's (a write covers everything, a
    /// read covers reads), so it is forgotten here. This keeps the list, and
    /// with it the cost of every later acquisition, bounded by the number of
    /// streams using the allocation rather than by the number of queued
    /// operations. Their leases still keep the allocation alive until the
    /// retirement worker drops them.
    fn complete(&mut self, id: u64, event: &Arc<crate::Event>) {
        let Some(access) = self.active.iter_mut().find(|a| a.id == id) else {
            return;
        };
        if access.recording {
            return;
        }
        self.marks += 1;
        access.completion = Some(event.clone());
        access.marked = self.marks;
        let (stream, write, seen) = (access.stream, access.write, access.seen);
        self.active.retain(|o| {
            let covered = o.id != id
                && !o.recording
                && o.marked != 0
                && o.marked <= seen
                && (write || !o.write)
                && (o.stream == stream || write);
            !covered
        });
    }
}

enum Owner {
    Synchronous,
    StreamOrdered(Arc<Stream>),
    Borrowed,
    Foreign(#[allow(dead_code)] Arc<dyn DeviceAllocation>),
}

struct Allocation {
    ptr: CUdeviceptr,
    bytes: usize,
    context: Arc<Context>,
    owner: Owner,
    accesses: Mutex<Accesses>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.ptr == 0 || matches!(self.owner, Owner::Borrowed | Owner::Foreign(_)) {
            return;
        }
        if self
            .accesses
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .detached
        {
            return;
        }
        let ptr = self.ptr;
        let context = self.context.clone();
        let allocation_stream = match &self.owner {
            Owner::StreamOrdered(s) => Some(s.clone()),
            _ => None,
        };
        // All safe users hold an independent access lease until completion.
        // Driver teardown is retired on a worker, never on the dropping thread.
        crate::completion::defer(move || {
            if let Err(error) = context.bind_to_thread() {
                context.record_err(Err::<(), _>(error));
                std::mem::forget((context, allocation_stream));
                return;
            }
            let result = if let Some(stream) = &allocation_stream {
                // SAFETY: the allocation is no longer accessed and the retained
                // stream is the one on which its allocation became valid.
                unsafe { crate::memory::free_async(ptr, stream.cu_stream()) }
            } else {
                // SAFETY: ownership of this synchronous allocation is unique.
                unsafe { crate::memory::free_sync(ptr) }
            };
            if let Err(error) = result {
                context.record_err(Err::<(), _>(error));
                std::mem::forget((context, allocation_stream));
            }
        });
    }
}

/// A byte view sharing an allocation owner and access tracker with every derived
/// typed buffer, tensor, and sliced view. Cloning does not copy device memory.
#[derive(Clone)]
pub struct RawDeviceBuffer {
    allocation: Arc<Allocation>,
    // Only user-visible views share this token. Completion leases retain the
    // allocation directly, so creating/dropping them cannot affect uniqueness.
    views: Arc<()>,
    offset: usize,
    len: usize,
}

impl std::fmt::Debug for RawDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawDeviceBuffer")
            .field("ptr", &self.cu_deviceptr())
            .field("len", &self.len)
            .field("context", &self.allocation.context)
            .finish_non_exhaustive()
    }
}

impl RawDeviceBuffer {
    fn new(ptr: CUdeviceptr, bytes: usize, context: Arc<Context>, owner: Owner) -> Self {
        assert!(
            ptr.checked_add(bytes as u64).is_some(),
            "device allocation extent overflows"
        );
        Self {
            allocation: Arc::new(Allocation {
                ptr,
                bytes,
                context,
                owner,
                accesses: Mutex::default(),
            }),
            views: Arc::new(()),
            offset: 0,
            len: bytes,
        }
    }

    /// Adopts a synchronous CUDA allocation.
    ///
    /// # Safety
    /// `ptr` must uniquely own `bytes` bytes allocated by `cuMemAlloc` in
    /// `context`, with no outstanding untracked accesses.
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, bytes: usize, context: Arc<Context>) -> Self {
        Self::new(ptr, bytes, context, Owner::Synchronous)
    }

    /// Adopts a stream-ordered CUDA allocation.
    ///
    /// # Safety
    /// `ptr` must uniquely own `bytes` bytes from `cuMemAllocAsync` or its pool
    /// variant on `stream`. The caller must establish allocation ordering for
    /// every use and retain the allocation through outstanding work.
    pub unsafe fn from_raw_parts_async(
        ptr: CUdeviceptr,
        bytes: usize,
        stream: Arc<Stream>,
    ) -> Self {
        Self::new(
            ptr,
            bytes,
            stream.context().clone(),
            Owner::StreamOrdered(stream),
        )
    }

    /// Borrows externally managed memory without taking ownership.
    ///
    /// # Safety
    /// Memory and context must outlive every derived view and queued use. No
    /// external access may conflict with accesses through this allocation.
    pub unsafe fn borrowed_from_raw_parts(
        ptr: CUdeviceptr,
        bytes: usize,
        context: Arc<Context>,
    ) -> Self {
        Self::new(ptr, bytes, context, Owner::Borrowed)
    }

    /// Retains an external owner and limits the logical view to `bytes` bytes.
    /// No CUDA calls are made, permitting metadata validation before submission.
    pub fn foreign(owner: Arc<dyn DeviceAllocation>, bytes: usize) -> Self {
        assert!(
            bytes <= owner.len_bytes(),
            "view exceeds foreign allocation"
        );
        Self::new(
            owner.device_ptr(),
            bytes,
            owner.context(),
            Owner::Foreign(owner),
        )
    }

    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.allocation.ptr + self.offset as u64
    }
    pub fn len_bytes(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn context(&self) -> &Arc<Context> {
        &self.allocation.context
    }
    pub(crate) fn allocation_stream(&self) -> Option<&Arc<Stream>> {
        match &self.allocation.owner {
            Owner::StreamOrdered(stream) => Some(stream),
            _ => None,
        }
    }
    pub fn device_id(&self) -> usize {
        self.context().ordinal()
    }
    /// Identity of the owner/access tracker, unchanged across byte views.
    pub fn allocation_id(&self) -> usize {
        Arc::as_ptr(&self.allocation) as usize
    }

    /// Creates a bounds-checked view without changing allocation identity.
    pub fn slice_bytes(&self, offset: usize, bytes: usize) -> Result<Self, DriverError> {
        if offset.checked_add(bytes).is_none_or(|end| end > self.len) {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        Ok(Self {
            allocation: self.allocation.clone(),
            views: self.views.clone(),
            offset: self.offset + offset,
            len: bytes,
        })
    }

    /// Whether only this user-visible view owns the allocation. Completion leases
    /// do not count as user aliases; they still enforce in-flight exclusion.
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.views) == 1
    }

    /// Acquires an allocation-wide access, rejecting cross-stream conflicts.
    /// The returned lease must be retained until the submitted work completes.
    ///
    /// Safe copies and launches use [`acquire_ordered`](Self::acquire_ordered)
    /// instead, which waits for conflicting recorded work; this form remains
    /// for graph capture (`recording`) and for callers that need to probe
    /// whether an access would conflict.
    pub fn acquire(
        &self,
        stream: &Stream,
        write: bool,
        recording: bool,
    ) -> Result<AllocationLease, DriverError> {
        if self.context().as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.acquire_on_stream(stream.cu_stream() as usize, write, recording)
    }

    /// Acquires an allocation-wide access for work enqueued next on `stream`,
    /// ordering it after conflicting accesses on other streams rather than
    /// rejecting them.
    ///
    /// A conflicting access whose submission has recorded its completion event
    /// (every enqueued `DeviceBuffer` copy and synchronous `#[cuda_module]`
    /// launch does, as does a `cuda-async` submission once it outlives its
    /// inline spin) is ordered by enqueuing `cuStreamWaitEvent` for that event
    /// on `stream`, so work enqueued after this call cannot start before it.
    /// The tracker cannot see `Stream::join` or event waits, so this keeps
    /// cross-stream patterns that are already ordered, such as fork/join, from
    /// failing depending on GPU timing; with the join in place the extra wait
    /// is already satisfied. A conflicting access without a recorded
    /// completion still returns `CUDA_ERROR_NOT_READY`, as
    /// [`acquire`](Self::acquire) does.
    ///
    /// On a stream that is capturing a CUDA graph, waiting for an event
    /// recorded outside the capture is a capture error, so capturing callers
    /// use [`acquire`](Self::acquire) with `recording` set instead.
    ///
    /// The returned lease must be retained until the submitted work completes.
    /// If a wait cannot be enqueued, the access is released and the driver
    /// error is returned; waits enqueued before it only add ordering.
    pub fn acquire_ordered(
        &self,
        stream: &Stream,
        write: bool,
    ) -> Result<AllocationLease, DriverError> {
        if self.context().as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        let mut waits = Vec::new();
        let id = self
            .allocation
            .accesses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register_ordered(stream.cu_stream() as usize, write, &mut waits)?;
        let lease = AllocationLease {
            allocation: self.allocation.clone(),
            id,
        };
        for event in &waits {
            stream.wait_event(event)?;
        }
        Ok(lease)
    }

    /// Registers bookkeeping for a known stream handle. This does not submit GPU
    /// work or grant permission to dereference pointers. Runtime adapters must
    /// separately validate the stream's context before submitting work.
    #[doc(hidden)]
    pub fn acquire_on_stream(
        &self,
        stream: usize,
        write: bool,
        recording: bool,
    ) -> Result<AllocationLease, DriverError> {
        let mut accesses = self
            .allocation
            .accesses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let id = accesses.register(stream, write, recording)?;
        Ok(AllocationLease {
            allocation: self.allocation.clone(),
            id,
        })
    }

    /// Moves ownership of the whole allocation out as raw parts.
    ///
    /// Leases whose accesses have all completed may still be waiting for the
    /// retirement worker; they no longer use the memory and do not prevent
    /// detaching. The allocation is then marked detached so dropping the last
    /// lease does not free memory the caller now owns.
    ///
    /// # Panics
    ///
    /// Panics for a sliced, partial, borrowed or foreign view, while another
    /// view shares the allocation, or while any access may still be running.
    pub(crate) fn into_owned_parts(
        self,
    ) -> (CUdeviceptr, usize, Arc<Context>, Option<Arc<Stream>>) {
        assert_eq!(self.offset, 0, "cannot detach a sliced allocation");
        assert_eq!(
            self.len, self.allocation.bytes,
            "cannot detach a partial allocation"
        );
        let shared = match Arc::try_unwrap(self.allocation) {
            Ok(mut allocation) => {
                let stream = Self::detachable_stream(&allocation);
                let ptr = std::mem::replace(&mut allocation.ptr, 0);
                return (ptr, allocation.bytes, allocation.context.clone(), stream);
            }
            Err(shared) => shared,
        };
        // Completion leases keep the allocation shared until the retirement
        // worker drops them, even after their work has finished. Once every
        // access they registered has completed and no other view remains, the
        // memory is no longer used, so ownership can move out now and the
        // shared destructor is told not to free it.
        assert!(
            Arc::strong_count(&self.views) == 1,
            "cannot detach a shared or in-flight allocation"
        );
        let stream = Self::detachable_stream(&shared);
        let mut accesses = shared.accesses.lock().unwrap_or_else(|e| e.into_inner());
        accesses.retire_completed();
        assert!(
            accesses.active.is_empty(),
            "cannot detach a shared or in-flight allocation"
        );
        accesses.detached = true;
        (shared.ptr, shared.bytes, shared.context.clone(), stream)
    }

    /// Returns the deallocation stream of an allocation this crate owns.
    ///
    /// # Panics
    ///
    /// Panics for borrowed and foreign allocations, whose memory belongs to
    /// someone else and cannot be handed out as owned raw parts.
    fn detachable_stream(allocation: &Allocation) -> Option<Arc<Stream>> {
        match &allocation.owner {
            Owner::Synchronous => None,
            Owner::StreamOrdered(stream) => Some(stream.clone()),
            _ => panic!("cannot detach an allocation owned externally"),
        }
    }
}

/// Keeps an allocation alive and excludes conflicting accesses until dropped.
/// A runtime must hold this beyond enqueue, until completion has been proven.
pub struct AllocationLease {
    allocation: Arc<Allocation>,
    id: u64,
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        let mut accesses = self
            .allocation
            .accesses
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = accesses.active.iter().position(|a| a.id == self.id) {
            accesses.active.swap_remove(pos);
        }
    }
}

impl AllocationLease {
    /// Transfers this lease to completion-bound retirement after submission.
    pub fn retain_until_complete(self, stream: &Stream) -> Result<(), DriverError> {
        Self::retain_all_until_complete(vec![self], stream)
    }

    /// Transfers several leases to completion-bound retirement after one
    /// submission on `stream`, behind a single recorded event.
    ///
    /// This is [`retain_until_complete`](Self::retain_until_complete) for a
    /// launch that touches several allocations: each access becomes
    /// reclaimable once the shared event completes, without one event record
    /// per argument. An empty list records nothing. If the event cannot be
    /// recorded, every lease is leaked, keeping its exclusion conservatively.
    pub fn retain_all_until_complete(
        leases: Vec<Self>,
        stream: &Stream,
    ) -> Result<(), DriverError> {
        if leases.is_empty() {
            return Ok(());
        }
        let event = match stream.record_event(None) {
            Ok(event) => Arc::new(event),
            Err(error) => {
                std::mem::forget(leases);
                return Err(error);
            }
        };
        for lease in &leases {
            lease.complete_with(&event);
        }
        crate::completion::retain_until_event(leases, event);
        Ok(())
    }

    /// Attaches `event` to this lease's access as its completion.
    ///
    /// Runtimes that keep leases inside their own submission (instead of
    /// handing them to [`retain_until_complete`](Self::retain_until_complete))
    /// call this once the submission's work is enqueued. Conflicting ordered
    /// acquisitions on other streams then wait for `event` on the GPU rather
    /// than reject, and a completed event releases the exclusion before the
    /// lease itself is dropped. The lease must still be retained until
    /// completion.
    ///
    /// # Safety
    /// `event` must have been recorded on the stream this lease was acquired
    /// for, after every operation using the access was enqueued there, and
    /// no work using the access may be enqueued after it.
    #[doc(hidden)]
    pub unsafe fn complete_after(&self, event: &Arc<crate::Event>) {
        self.complete_with(event);
    }

    /// Marks this lease's access as finished once `event` completes.
    ///
    /// A later conflicting acquisition queries the event and drops the access
    /// itself, so a caller that already synchronized the stream does not see
    /// a stale conflict while the retirement worker catches up.
    fn complete_with(&self, event: &Arc<crate::Event>) {
        self.allocation
            .accesses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .complete(self.id, event);
    }
}

impl std::fmt::Debug for AllocationLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocationLease")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn metadata() -> RawDeviceBuffer {
        // SAFETY: bookkeeping only. These borrowed handles are never passed to
        // CUDA, dereferenced, or freed by their destructors.
        unsafe {
            RawDeviceBuffer::borrowed_from_raw_parts(
                0x1000,
                128,
                Context::borrow_raw(0x2000 as *mut _, 0, 0),
            )
        }
    }

    #[test]
    fn uniqueness_counts_byte_views_but_not_completion_leases() {
        let raw = metadata();
        let view = raw.slice_bytes(16, 32).unwrap();
        let clone = raw.clone();
        let lease = raw.acquire_on_stream(1, true, false).unwrap();
        assert!(!raw.is_unique());
        drop(view);
        assert!(!raw.is_unique());
        drop(clone);
        assert!(raw.is_unique());
        assert!(raw.acquire_on_stream(2, false, false).is_err());
        drop(lease);
        assert!(raw.acquire_on_stream(2, false, false).is_ok());
    }

    /// Ids of the accesses still tracked, in ascending order.
    fn ids(accesses: &Accesses) -> Vec<u64> {
        let mut ids: Vec<u64> = accesses.active.iter().map(|a| a.id).collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn completed_accesses_are_subsumed_by_later_covering_ones() {
        let Ok(context) = Context::new(0) else {
            return;
        };
        // An event behind a parked host callback stays incomplete, so pruning
        // completed accesses cannot hide what subsumption did.
        let stream = context.new_stream().unwrap();
        let (release, parked) = std::sync::mpsc::channel::<()>();
        stream
            .launch_host_function(move || {
                let _ = parked.recv_timeout(std::time::Duration::from_secs(10));
            })
            .unwrap();
        let event = Arc::new(stream.record_event(None).unwrap());
        let mut accesses = Accesses::default();

        // A queue of operations on one stream keeps a single entry: each one
        // whose completion is recorded covers those recorded before it began.
        let mut last = 0;
        for _ in 0..10_000 {
            last = accesses.register(1, true, false).unwrap();
            accesses.complete(last, &event);
        }
        assert_eq!(ids(&accesses), [last]);

        // A read does not cover an earlier write, but covers earlier reads.
        let read = accesses.register(1, false, false).unwrap();
        accesses.complete(read, &event);
        let next_read = accesses.register(1, false, false).unwrap();
        accesses.complete(next_read, &event);
        assert_eq!(ids(&accesses), [last, next_read]);

        // An access registered before another's completion was recorded does
        // not cover it: its work may have been enqueued first.
        let early = accesses.register(1, true, false).unwrap();
        let late = accesses.register(1, true, false).unwrap();
        accesses.complete(late, &event);
        accesses.complete(early, &event);
        assert_eq!(ids(&accesses), [early, late]);

        // Accesses without a recorded completion, and recording ones, are
        // never covered.
        let unrecorded = accesses.register(1, false, false).unwrap();
        let recording = accesses.register(1, true, true).unwrap();
        let write = accesses.register(1, true, false).unwrap();
        accesses.complete(write, &event);
        assert_eq!(ids(&accesses), [unrecorded, recording, write]);

        // A write on another stream was ordered after every conflicting
        // recorded access, so it covers them; a read there covers none.
        accesses.active.retain(|a| a.id != unrecorded);
        let read_elsewhere = accesses.register(2, false, false);
        assert!(read_elsewhere.is_err(), "conflicts with the stream-1 write");
        let mut waits = Vec::new();
        let read_elsewhere = accesses.register_ordered(2, false, &mut waits).unwrap();
        assert_eq!(waits.len(), 1);
        accesses.complete(read_elsewhere, &event);
        assert_eq!(ids(&accesses), [recording, write, read_elsewhere]);
        waits.clear();
        let write_elsewhere = accesses.register_ordered(2, true, &mut waits).unwrap();
        accesses.complete(write_elsewhere, &event);
        assert_eq!(ids(&accesses), [recording, write_elsewhere]);
        drop(release);
        stream.synchronize().unwrap();
    }

    #[test]
    fn concurrent_lease_retirement_cannot_change_user_uniqueness() {
        let raw = metadata();
        let stop = AtomicBool::new(false);
        let mut false_negatives = 0;
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    drop(raw.acquire_on_stream(1, false, false).unwrap());
                }
            });
            for _ in 0..100_000 {
                false_negatives += usize::from(!raw.is_unique());
            }
            stop.store(true, Ordering::Relaxed);
        });
        assert_eq!(false_negatives, 0);
    }
}
