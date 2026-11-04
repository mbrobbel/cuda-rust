/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Retiring device allocations while a stream is being captured.
//!
//! These tests live in their own binary: a global-mode capture makes unsafe
//! driver calls from every other thread in the process fail, so unrelated tests
//! running concurrently in the same binary would be disturbed.

use cuda_core::{sys, CudaContext, CudaStream, DeviceBuffer, DriverError};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Size of every allocation these tests drop during a capture.
///
/// Large enough to come from a real pool or `cuMemAlloc` block rather than a
/// sub-allocation that the address-range query could not tell apart.
const ALLOCATION_BYTES: usize = 1 << 18;

/// How long the capturing thread waits after a drop before ending the capture.
///
/// The retirement worker polls every millisecond, so this gives a buggy worker
/// ample opportunity to issue its free into the capture.
const RETIREMENT_WINDOW: Duration = Duration::from_millis(100);

/// Upper bound for a deferred free to land once the capture has ended.
///
/// Generous so that a loaded shared GPU host does not make the test flaky.
const FREE_TIMEOUT: Duration = Duration::from_secs(5);

/// Capture mode in which other threads' unsafe driver calls invalidate the capture.
const GLOBAL: sys::CUstreamCaptureMode =
    sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL;

/// Capture mode that `cuda-async` uses for its graphs.
const RELAXED: sys::CUstreamCaptureMode =
    sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED;

/// Serializes the tests in this binary.
///
/// A global-mode capture in one test would otherwise break the driver calls of
/// another test running on a different thread.
static CAPTURE_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquires [`CAPTURE_TEST_LOCK`], ignoring poison left by a failed test.
///
/// Every test ends its capture before asserting, so a panic cannot leave a
/// capture running for the next test.
fn lock_capture_test() -> MutexGuard<'static, ()> {
    CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Returns the driver's address-range lookup result for `ptr`.
///
/// `CUDA_SUCCESS` means the allocation is still live and
/// `CUDA_ERROR_NOT_FOUND` means it has been freed. Requires a current context.
fn address_range_status(ptr: sys::CUdeviceptr) -> sys::CUresult {
    let (mut base, mut size) = (0, 0);
    unsafe { sys::cuMemGetAddressRange_v2(&mut base, &mut size, ptr) }
}

/// Polls until the allocation at `ptr` is freed or [`FREE_TIMEOUT`] expires.
///
/// Returns whether the driver reported the address as unmapped in time.
fn wait_until_freed(ptr: sys::CUdeviceptr) -> bool {
    let deadline = Instant::now() + FREE_TIMEOUT;
    while Instant::now() < deadline {
        if address_range_status(ptr) == sys::cudaError_enum_CUDA_ERROR_NOT_FOUND {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

/// Ends the capture on `stream` and returns how many nodes it recorded.
///
/// The captured graph is destroyed before returning.
fn end_capture_node_count(stream: &CudaStream) -> Result<usize, DriverError> {
    let graph = unsafe { stream.end_capture() }?;
    let mut count = 0;
    let status = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) };
    unsafe { sys::cuGraphDestroy(graph) };
    assert_eq!(status, sys::cudaError_enum_CUDA_SUCCESS);
    Ok(count)
}

/// Drops a stream-ordered allocation while its own allocation stream captures.
///
/// The graph did not allocate this memory, so the free must not join the
/// capture, and it must not be lost either: it waits for the capture to end and
/// must not leave an error in the context.
fn async_allocation_dropped_while_allocation_stream_captures(mode: sys::CUstreamCaptureMode) {
    let _guard = lock_capture_test();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let buffer =
        unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, ALLOCATION_BYTES) }.unwrap();
    stream.synchronize().unwrap();
    let ptr = buffer.cu_deviceptr();
    ctx.bind_to_thread().unwrap();

    unsafe { stream.begin_capture(mode) }.unwrap();
    drop(buffer);
    std::thread::sleep(RETIREMENT_WINDOW);
    let status_during_capture = address_range_status(ptr);
    let captured = end_capture_node_count(&stream);

    assert_eq!(
        status_during_capture,
        sys::cudaError_enum_CUDA_SUCCESS,
        "allocation must stay live while its stream captures"
    );
    assert_eq!(
        captured,
        Ok(0),
        "the deferred free must not join the capture"
    );
    assert_eq!(
        ctx.check_err(),
        Ok(()),
        "retirement recorded a driver error"
    );
    assert!(
        wait_until_freed(ptr),
        "allocation was not freed after the capture ended"
    );
}

#[test]
fn async_allocation_dropped_while_allocation_stream_captures_is_freed_after_capture() {
    async_allocation_dropped_while_allocation_stream_captures(RELAXED);
}

#[test]
fn async_allocation_dropped_while_allocation_stream_captures_globally_is_freed_after_capture() {
    async_allocation_dropped_while_allocation_stream_captures(GLOBAL);
}

/// Number of times the in-capture test launches its instantiated graph.
///
/// A graph whose allocation node has no matching free node cannot be launched
/// again while that allocation is live, so more than one launch exposes it.
const GRAPH_LAUNCHES: usize = 3;

/// Allocates and drops a stream-ordered buffer inside one capture of its stream.
///
/// The allocation is a graph allocation, so its free must join the same capture
/// as a free node. Deferring it past the capture would make the driver reject it,
/// record an error, and leave a graph that cannot be relaunched.
fn async_allocation_made_and_dropped_inside_capture(mode: sys::CUstreamCaptureMode) {
    let _guard = lock_capture_test();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    ctx.bind_to_thread().unwrap();

    unsafe { stream.begin_capture(mode) }.unwrap();
    let buffer =
        unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, ALLOCATION_BYTES) }.unwrap();
    drop(buffer);
    std::thread::sleep(RETIREMENT_WINDOW);
    let graph = unsafe { stream.end_capture() }.unwrap();

    let mut nodes = 0;
    let status = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut nodes) };
    assert_eq!(status, sys::cudaError_enum_CUDA_SUCCESS);
    let mut exec = std::ptr::null_mut();
    let instantiated = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
    let launches: Vec<_> = (0..GRAPH_LAUNCHES)
        .map(|_| unsafe {
            let launched = sys::cuGraphLaunch(exec, stream.cu_stream());
            (launched, sys::cuStreamSynchronize(stream.cu_stream()))
        })
        .collect();
    unsafe {
        sys::cuGraphExecDestroy(exec);
        sys::cuGraphDestroy(graph);
    }

    assert_eq!(
        nodes, 2,
        "the free must join the capture next to the allocation"
    );
    assert_eq!(
        ctx.check_err(),
        Ok(()),
        "retirement recorded a driver error"
    );
    assert_eq!(instantiated, sys::cudaError_enum_CUDA_SUCCESS);
    let ok = (
        sys::cudaError_enum_CUDA_SUCCESS,
        sys::cudaError_enum_CUDA_SUCCESS,
    );
    assert_eq!(
        launches,
        vec![ok; GRAPH_LAUNCHES],
        "the captured graph must be relaunchable"
    );
}

#[test]
fn async_allocation_made_and_dropped_inside_capture_is_freed_by_the_graph() {
    async_allocation_made_and_dropped_inside_capture(RELAXED);
}

#[test]
fn async_allocation_made_and_dropped_inside_global_capture_is_freed_by_the_graph() {
    async_allocation_made_and_dropped_inside_capture(GLOBAL);
}

#[test]
fn drops_during_global_capture_of_another_stream_do_not_invalidate_it() {
    let _guard = lock_capture_test();
    let ctx = CudaContext::new(0).unwrap();
    let captured_stream = ctx.new_stream().unwrap();
    let other_stream = ctx.new_stream().unwrap();
    let synchronous = DeviceBuffer::<u8>::zeroed(&other_stream, ALLOCATION_BYTES).unwrap();
    let stream_ordered =
        unsafe { DeviceBuffer::<u8>::uninitialized_async(&other_stream, ALLOCATION_BYTES) }
            .unwrap();
    other_stream.synchronize().unwrap();
    let synchronous_ptr = synchronous.cu_deviceptr();
    let stream_ordered_ptr = stream_ordered.cu_deviceptr();
    ctx.bind_to_thread().unwrap();

    unsafe { captured_stream.begin_capture(GLOBAL) }.unwrap();
    drop(synchronous);
    drop(stream_ordered);
    std::thread::sleep(RETIREMENT_WINDOW);
    let captured = end_capture_node_count(&captured_stream);

    assert_eq!(
        captured,
        Ok(0),
        "retiring allocations of another stream invalidated the capture"
    );
    assert_eq!(
        ctx.check_err(),
        Ok(()),
        "retirement recorded a driver error"
    );
    assert!(
        wait_until_freed(synchronous_ptr),
        "synchronous allocation was not freed"
    );
    assert!(
        wait_until_freed(stream_ordered_ptr),
        "stream-ordered allocation was not freed"
    );
}
