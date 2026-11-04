/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: an enum payload that splits a pointer's bytes across two
//! narrower integers cannot share storage with that pointer.
//!
//! A pointer-sized integer overlaying the pointer exactly shares its slot
//! through explicit `inttoptr`/`ptrtoint` (see `enum_pointer_int_overlap`).
//! Two `u32` halves are no such integer view, so lowering still refuses to
//! erase LLVM pointer provenance here.

use cuda_device::kernel;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum PointerOrBits {
    Bits(u32, u32),
    Pointer(*const u8),
}

#[inline(never)]
fn value_bits(value: PointerOrBits) -> u64 {
    match value {
        PointerOrBits::Bits(low, high) => (u64::from(high) << 32) | u64::from(low),
        PointerOrBits::Pointer(pointer) => pointer as u64,
    }
}

/// # Safety
///
/// `input` must point to one initialized `PointerOrBits`, and `out` must point
/// to one writable `u64`. Neither location may be raced by another thread.
#[kernel]
pub unsafe fn enum_pointer_overlap(input: *const PointerOrBits, out: *mut u64) {
    let value = unsafe { input.read() };
    unsafe {
        out.write(value_bits(value));
    }
}

fn main() {
    // Keep both variants reachable in the host crate as well as device MIR.
    let values = [
        PointerOrBits::Bits(7, 0),
        PointerOrBits::Pointer(core::ptr::null()),
    ];
    println!(
        "This negative example should fail during device compilation ({} variants).",
        values.len()
    );
}
