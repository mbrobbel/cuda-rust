// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// rustc does not report `#[inline(never)]` as ignored on an exported
// function, so `#[device]` must not add `#[allow(unused_attributes)]` for it.
// Under `#![forbid(unused_attributes)]` such an allow is itself an error
// (E0453), so this crate fails to build if `#[device]` ever emits it for
// `#[inline(never)]` or for a function without an `inline` attribute.
#![forbid(unused_attributes)]

use cuda_device::{DisjointSlice, cuda_module, device, kernel};

#[device]
#[inline(never)]
fn never(x: u32) -> u32 {
    x ^ 3
}

#[inline(never)]
#[device]
fn never_outer(x: u32) -> u32 {
    x + 5
}

#[device]
fn plain(x: u32) -> u32 {
    x * 3
}

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    #[inline(never)]
    fn halved(x: u32) -> u32 {
        x / 2
    }

    #[kernel]
    pub fn halves(mut out: DisjointSlice<u32>) {
        if let Some((slot, index)) = out.get_mut_indexed() {
            *slot = halved(index.get() as u32);
        }
    }
}

#[kernel]
fn combined(value: u32) {
    let _ = never(value) + never_outer(value) + plain(value);
}

fn main() {
    assert_eq!((never(1), never_outer(1), plain(1)), (2, 6, 3));
}
