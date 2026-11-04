/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: an enum discriminant value that needs more than 64 bits.
//!
//! 128-bit tags are supported when every declared value fits in 64 bits
//! (`Option<i128>`, small `#[repr(u128)]` enums; see `wide_enum_tags`). The
//! dialect stores declared values as `u64`, so `B = 1 << 64` must be
//! rejected instead of being truncated to the same tag as `A = 0`.

use cuda_device::kernel;

/// A `#[repr(u128)]` enum whose second discriminant does not fit in 64 bits.
#[repr(u128)]
#[derive(Clone, Copy)]
pub enum Wide {
    A = 0,
    B = 1 << 64,
}

/// Reads the tag through a `match`, so device MIR needs the enum type.
#[inline(never)]
fn is_b(value: Wide) -> u32 {
    match value {
        Wide::A => 0,
        Wide::B => 1,
    }
}

/// # Safety
///
/// `input` must point to one initialized `Wide`, and `out` must point to one
/// writable `u32`. Neither location may be raced by another thread.
#[kernel]
pub unsafe fn enum_wide_discriminant(input: *const Wide, out: *mut u32) {
    let value = unsafe { input.read() };
    unsafe {
        out.write(is_b(value));
    }
}

fn main() {
    // Keep both variants reachable in the host crate as well as device MIR.
    let values = [Wide::A, Wide::B];
    println!(
        "This negative example should fail during device compilation ({} variants, B = {:#x}).",
        values.len(),
        values[1] as u128
    );
}
