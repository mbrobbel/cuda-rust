// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A non-generic `#[device]` function is emitted with `#[unsafe(no_mangle)]`
// and keeps the user's `#[inline]` attribute, because the backend reads that
// attribute and passes it to LLVM as `inlinehint` or `alwaysinline`. On an
// exported function rustc's `unused_attributes` lint reports the attribute as
// ignored, which only holds for cross-crate inlining on the host. This crate
// denies that lint, so the build fails if the documented `#[device]` plus
// `#[inline]` pattern ever warns again, in either attribute order and inside
// a `#[cuda_module]`.
#![deny(unused_attributes)]

use cuda_device::{DisjointSlice, cuda_module, device, kernel};

#[device]
#[inline]
fn hinted(x: u32) -> u32 {
    x + 1
}

#[device]
#[inline(always)]
fn always(x: u32) -> u32 {
    x * 2
}

#[device]
#[inline(never)]
fn never(x: u32) -> u32 {
    x ^ 3
}

#[inline(always)]
#[device]
fn always_outer(x: u32) -> u32 {
    x - 1
}

#[device]
#[inline(always)]
unsafe fn always_unsafe(x: u32) -> u32 {
    x << 1
}

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    #[inline(always)]
    fn twice(x: u32) -> u32 {
        x * 2
    }

    #[kernel]
    pub fn doubled(mut out: DisjointSlice<u32>) {
        if let Some((slot, index)) = out.get_mut_indexed() {
            *slot = twice(index.get() as u32);
        }
    }
}

#[kernel]
fn combined(value: u32) {
    // SAFETY: `always_unsafe` has no preconditions; it is `unsafe` only to
    // cover the unsafe wrapper path of the macro.
    let shifted = unsafe { always_unsafe(value) };
    let _ = hinted(value) + always(value) + never(value) + always_outer(value) + shifted;
}

fn main() {
    // SAFETY: as above, `always_unsafe` has no preconditions.
    let shifted = unsafe { always_unsafe(1) };
    assert_eq!(
        (hinted(1), always(1), never(1), always_outer(1), shifted),
        (2, 2, 2, 0, 2)
    );
}
