/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Completion-bound retirement without waiting in destructors.

use crate::{DriverError, Event, Stream};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::Duration;

type Cleanup = Box<dyn FnOnce() + Send>;

/// A condition the retirement worker polls before running a cleanup.
///
/// `Ok(true)` means ready, `Ok(false)` means poll again later, and `Err` means
/// readiness can no longer be observed, so the cleanup is leaked.
type ReadyPoll = Box<dyn FnMut() -> Result<bool, DriverError> + Send>;

/// When the retirement worker may run a pending cleanup.
///
/// The worker re-checks every pending entry roughly once per millisecond.
enum Readiness {
    /// The cleanup may run immediately.
    Now,
    /// The cleanup runs once this event has completed.
    Event(Arc<Event>),
    /// The cleanup runs once this poll reports `Ok(true)`.
    Poll(ReadyPoll),
}

impl Readiness {
    /// Checks whether the cleanup guarded by this readiness may run now.
    ///
    /// Errors are passed through so the worker can fail closed by leaking.
    fn poll(&mut self) -> Result<bool, DriverError> {
        match self {
            Self::Now => Ok(true),
            Self::Event(event) => event.query(),
            Self::Poll(poll) => poll(),
        }
    }
}

struct Pending {
    ready: Readiness,
    cleanup: Cleanup,
}

/// Puts the calling retirement thread into relaxed stream-capture mode.
///
/// In the default global mode, every free, query, or destroy the worker issues
/// while another thread holds a global-mode capture fails with
/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` and invalidates that capture. Relaxed
/// mode only lifts that restriction for calls that never join a capture; the
/// worker's only work on a capturing stream is the free of a graph allocation,
/// which is legal in every mode. If the exchange fails, the worker keeps the
/// default mode and such calls fail closed by leaking.
fn relax_stream_capture_mode() {
    let mut mode = cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED;
    // SAFETY: `mode` is a valid capture mode in writable storage; the call only
    // changes this thread's capture interaction mode.
    let _ = unsafe { cuda_bindings::cuThreadExchangeStreamCaptureMode(&mut mode) };
}

fn enqueue(pending: Pending) {
    static QUEUE: OnceLock<Option<mpsc::Sender<Pending>>> = OnceLock::new();
    let queue = QUEUE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Pending>();
        std::thread::Builder::new()
            .name("cuda-resource-retirement".into())
            .spawn(move || {
                relax_stream_capture_mode();
                let mut pending = Vec::<Pending>::new();
                loop {
                    match receiver.recv_timeout(Duration::from_millis(1)) {
                        Ok(item) => pending.push(item),
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    pending.extend(receiver.try_iter());
                    let mut i = 0;
                    while i < pending.len() {
                        match pending[i].ready.poll() {
                            Ok(false) => i += 1,
                            Ok(true) => {
                                let item = pending.swap_remove(i);
                                // Preserve the event's context until cleanup has finished.
                                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                    item.cleanup,
                                ));
                            }
                            Err(_) => {
                                // Completion is unknown: leaking the owner and its access
                                // lease is safer than freeing memory or allowing conflicting work.
                                std::mem::forget(pending.swap_remove(i));
                            }
                        }
                    }
                }
                for item in pending {
                    std::mem::forget(item);
                }
            })
            .ok()
            .map(|_| sender)
    });
    if let Some(queue) = queue {
        if let Err(error) = queue.send(pending) {
            std::mem::forget(error.0);
        }
    } else {
        std::mem::forget(pending);
    }
}

pub(crate) fn defer(cleanup: impl FnOnce() + Send + 'static) {
    enqueue(Pending {
        ready: Readiness::Now,
        cleanup: Box::new(cleanup),
    });
}

/// Runs `cleanup` on the retirement worker once `ready` reports `Ok(true)`.
///
/// `ready` is polled on the worker, never on the caller. If it returns `Err`,
/// `cleanup` and everything it owns are leaked instead of being run.
pub(crate) fn defer_until(
    ready: impl FnMut() -> Result<bool, DriverError> + Send + 'static,
    cleanup: impl FnOnce() + Send + 'static,
) {
    enqueue(Pending {
        ready: Readiness::Poll(Box::new(ready)),
        cleanup: Box::new(cleanup),
    });
}

/// Keeps an owner or access lease alive until work already queued on `stream`
/// completes. Dropping the caller's future cannot shorten this lifetime.
///
/// Failure to record or observe completion intentionally leaks the owner;
/// releasing it could invalidate work that the driver has already accepted.
pub fn retain_until_complete(
    owner: impl Send + 'static,
    stream: &Stream,
) -> Result<(), DriverError> {
    let event = match stream.record_event(None) {
        Ok(event) => event,
        Err(error) => {
            std::mem::forget(owner);
            return Err(error);
        }
    };
    enqueue(Pending {
        ready: Readiness::Event(Arc::new(event)),
        cleanup: Box::new(move || drop(owner)),
    });
    Ok(())
}

pub(crate) fn retain_until_event(owner: impl Send + 'static, event: Arc<Event>) {
    enqueue(Pending {
        ready: Readiness::Event(event),
        cleanup: Box::new(move || drop(owner)),
    });
}
