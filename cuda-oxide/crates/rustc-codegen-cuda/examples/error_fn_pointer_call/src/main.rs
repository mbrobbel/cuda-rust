/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: calling through a `fn` pointer value inside device code.
//!
//! A device `fn` pointer is a stable identity token (a hash of the
//! function's mangled name, see `examples/fn_ptr_naming`), not a code
//! address, so there is nothing to call. Before the importer refused the
//! call, `f(x)` below was emitted as a call to a placeholder symbol
//! `unknown_function` and only failed much later, as a module-level
//! verifier error that named neither the caller nor the cause.
//!
//! The importer must reject the call at the call site instead, naming the
//! caller and the pointer type.
//!
//! Usage:
//!   cargo oxide run error_fn_pointer_call
//!
//! Expected: the build FAILS with a diagnostic containing (pinned):
//!
//! ```text
//! calling through a function pointer
//! ```
//!
//! The direct-call form is covered here; `error_fn_pointer_trait_call`
//! covers the `F: FnMut` form, where the call goes through rustc's
//! fn-pointer shim instead.

use cuda_device::{DisjointSlice, kernel, thread};

/// First candidate callee; selected for even lanes.
fn inc(x: u32) -> u32 {
    x.wrapping_add(1)
}

/// Second candidate callee; selected for odd lanes.
fn dbl(x: u32) -> u32 {
    x.wrapping_mul(2)
}

/// Calls `f` through the pointer. `#[inline(never)]` keeps the call a real
/// call through a non-constant operand instead of letting rustc's MIR
/// inliner fold the pointer away at a call site where it is known.
#[inline(never)]
fn apply(f: fn(u32) -> u32, x: u32) -> u32 {
    // BUG UNDER TEST: an indirect call; device fn pointers are not addresses.
    f(x)
}

/// Picks the callee per lane at run time, so the pointer cannot be
/// devirtualized, and calls it through `apply`.
#[kernel]
pub fn fn_pointer_call(mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let lane = idx.get() as u32;
    let f: fn(u32) -> u32 = if lane % 2 == 0 { inc } else { dbl };
    if let Some(slot) = out.get_mut(idx) {
        *slot = apply(f, lane);
    }
}

/// Host entry point. Only reached if the device build unexpectedly succeeds.
fn main() {
    println!("=== error_fn_pointer_call ===");
    println!("This example is intentionally broken: the kernel calls through a");
    println!("fn pointer. The build must FAIL at MIR import with a diagnostic");
    println!("saying that calling through a function pointer is not supported.");
    println!();
    println!("If you see this message, the build did NOT fail and the");
    println!("diagnostic regression has returned.");
}
