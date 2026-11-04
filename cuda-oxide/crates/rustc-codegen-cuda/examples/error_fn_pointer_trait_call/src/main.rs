/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a `fn` pointer passed where `F: FnMut(..)` is expected.
//!
//! Inside the generic `apply`, `f(x)` is a call to `<F as FnMut>::call_mut`.
//! With `F = fn(u32) -> u32` that resolves to rustc's fn-pointer shim, whose
//! body calls through the pointer. A device `fn` pointer is a stable
//! identity token, not a code address (see `examples/fn_ptr_naming`), so the
//! shim is never collected. Before the importer refused the call, it was
//! emitted against the shim's symbol and only failed much later, as a
//! module-level verifier error naming neither the caller nor the cause.
//!
//! The importer must reject the call at the call site instead.
//!
//! Usage:
//!   cargo oxide run error_fn_pointer_trait_call
//!
//! Expected: the build FAILS with a diagnostic containing (pinned):
//!
//! ```text
//! calling through a function pointer
//! ```
//!
//! Passing the fn item itself (`apply(inc, x)`) or a closure stays
//! supported: those types name their callee statically (see
//! `examples/function_item_call` and `examples/device_closures`).

use cuda_device::{DisjointSlice, kernel, thread};

/// First candidate callee; selected for even lanes.
fn inc(x: u32) -> u32 {
    x.wrapping_add(1)
}

/// Second candidate callee; selected for odd lanes.
fn dbl(x: u32) -> u32 {
    x.wrapping_mul(2)
}

/// Calls `f` through the callable traits. `#[inline(never)]` keeps the
/// generic body out of the kernel, so the monomorphized call stays a
/// `<fn(u32) -> u32 as FnMut>::call_mut` shim call.
#[inline(never)]
fn apply<F: FnMut(u32) -> u32>(mut f: F, x: u32) -> u32 {
    // BUG UNDER TEST: with `F` a fn pointer this calls through the pointer.
    f(x)
}

/// Picks the callee per lane at run time, so the pointer cannot be
/// devirtualized, and hands it to the generic `apply`.
#[kernel]
pub fn fn_pointer_trait_call(mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let lane = idx.get() as u32;
    let f: fn(u32) -> u32 = if lane % 2 == 0 { inc } else { dbl };
    if let Some(slot) = out.get_mut(idx) {
        *slot = apply(f, lane);
    }
}

/// Host entry point. Only reached if the device build unexpectedly succeeds.
fn main() {
    println!("=== error_fn_pointer_trait_call ===");
    println!("This example is intentionally broken: a fn pointer is called");
    println!("through `F: FnMut`. The build must FAIL at MIR import with a");
    println!("diagnostic saying that calling through a function pointer is not");
    println!("supported.");
    println!();
    println!("If you see this message, the build did NOT fail and the");
    println!("diagnostic regression has returned.");
}
