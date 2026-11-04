/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test: `&[T]` and `&str` constants over anonymous memory.
//!
//! A `const PREFIX: &[u8] = b"abc"`, a `&str` literal, or a sub-slice computed
//! in a `const` is a fat pointer whose data word points into memory rustc
//! allocated for the constant itself, not into a Rust `static`. The importer
//! used to reject every such operand with "Unsupported constant type in
//! translate_constant"; only slices over device statics had a lowering.
//!
//! Each constant is returned from an `#[inline(never)]` helper, so the slice
//! outlives the helper's frame and must point at an immutable device global,
//! not at a stack copy. The kernel indexes every table with the thread index,
//! which keeps rustc from folding the reads away, and reads one slot past the
//! end of each table so the length word is checked as well. The host computes
//! the same lookups from the same constants.
//!
//! Run:
//!     cargo oxide run slice_str_constants
//!
//! Exits 0 on SUCCESS, 1 on FAIL.

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

/// Byte-string literal behind a `&[u8]` constant.
/// Its data word points at an anonymous 10-byte allocation.
const BYTES: &[u8] = b"cuda-oxide";

/// Wider elements, so the global's element stride is four bytes.
/// The last value checks that all 32 bits survive the byte image.
const WORDS: &[u32] = &[10, 20, 30, 40, 0xDEAD_BEEF];

/// String literal behind a `&str` constant.
/// `str` lowers to a `u8` slice, so this takes the same path as `BYTES`.
const GREETING: &str = "hello, gpu";

/// Sub-string at byte offset 7 of `GREETING`'s allocation.
/// The data word carries a non-zero addend into the backing memory.
const WORLD: &str = GREETING.split_at(7).1;

/// Whole table that `TAIL` is cut from.
/// Only `TAIL` is read by the kernel.
const FULL: &[u16] = &[7, 8, 9, 10];

/// `FULL` without its first element: a two-byte addend on `u16` elements.
/// Computed in a `const`, so the kernel only ever sees the fat pointer.
const TAIL: &[u16] = match FULL.split_first() {
    Some((_, rest)) => rest,
    None => &[],
};

/// Float elements, compared by bit pattern.
/// Negative and fractional values make a byte-order slip visible.
const FLOATS: &[f32] = &[0.5, -1.25, 3.0e9];

/// Tuple elements, whose rustc layout has padding after the `u8`.
/// The global's stride must be rustc's tuple size, not the sum of fields.
const PAIRS: &[(u8, u32)] = &[(1, 100), (2, 200), (3, 300)];

/// Empty slice: the global has no bytes and every lookup misses.
/// Its length word must still read back as zero.
const EMPTY: &[u64] = &[];

/// Value a lookup returns when the index is past the end of the table.
/// No table element equals it, so a wrong length is always visible.
const MISSING: u32 = u32::MAX;

/// Number of tables the kernel reads, one per `lookup` arm.
/// Every lane picks its table as `lane / SLOTS`.
const TABLES: usize = 10;

/// Indices tried per table, one past the longest table.
/// Every lane picks its index as `lane % SLOTS`.
const SLOTS: usize = 11;

/// Device side: one helper per constant and a kernel that reads them.
/// The helpers are not inlined, so each returns its slice by value.
#[cuda_module]
mod kernels {
    use super::*;

    /// Return `BYTES` from a separate device function.
    /// The returned slice must still be valid in the caller.
    #[device]
    #[inline(never)]
    fn bytes() -> &'static [u8] {
        BYTES
    }

    /// Return `WORDS` from a separate device function.
    /// The returned slice must still be valid in the caller.
    #[device]
    #[inline(never)]
    fn words() -> &'static [u32] {
        WORDS
    }

    /// Return `GREETING` from a separate device function.
    /// The returned string must still be valid in the caller.
    #[device]
    #[inline(never)]
    fn greeting() -> &'static str {
        GREETING
    }

    /// Return the sub-string `WORLD` from a separate device function.
    /// It must start seven bytes into `GREETING`'s bytes.
    #[device]
    #[inline(never)]
    fn world() -> &'static str {
        WORLD
    }

    /// Return the sub-slice `TAIL` from a separate device function.
    /// It must start one element into `FULL`.
    #[device]
    #[inline(never)]
    fn tail() -> &'static [u16] {
        TAIL
    }

    /// Return `FLOATS` from a separate device function.
    /// The returned slice must still be valid in the caller.
    #[device]
    #[inline(never)]
    fn floats() -> &'static [f32] {
        FLOATS
    }

    /// Return `PAIRS` from a separate device function.
    /// The returned slice must still be valid in the caller.
    #[device]
    #[inline(never)]
    fn pairs() -> &'static [(u8, u32)] {
        PAIRS
    }

    /// Return `EMPTY` from a separate device function.
    /// The returned slice must have length zero.
    #[device]
    #[inline(never)]
    fn empty() -> &'static [u64] {
        EMPTY
    }

    /// Read slot `k` of table `table`, or `MISSING` past its end.
    /// Tables 8 and 9 use `BYTES` and `GREETING` directly, without a helper.
    #[device]
    #[inline(never)]
    fn lookup(table: usize, k: usize) -> u32 {
        match table {
            0 => bytes().get(k).map_or(MISSING, |&b| u32::from(b)),
            1 => words().get(k).copied().unwrap_or(MISSING),
            2 => greeting()
                .as_bytes()
                .get(k)
                .map_or(MISSING, |&b| u32::from(b)),
            3 => world().as_bytes().get(k).map_or(MISSING, |&b| u32::from(b)),
            4 => tail().get(k).map_or(MISSING, |&v| u32::from(v)),
            5 => floats().get(k).map_or(MISSING, |v| v.to_bits()),
            6 => pairs()
                .get(k)
                .map_or(MISSING, |&(tag, value)| (u32::from(tag) << 24) | value),
            7 => empty().get(k).map_or(MISSING, |&v| v as u32),
            8 => BYTES.get(k).map_or(MISSING, |&b| u32::from(b) + 1000),
            9 => GREETING
                .as_bytes()
                .get(k)
                .map_or(MISSING, |&b| u32::from(b) + 2000),
            _ => MISSING,
        }
    }

    /// Each lane reads one (table, slot) pair and writes the value it found.
    /// Lanes write only their own output slot.
    #[kernel]
    pub fn slice_str_constants(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let lane = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = lookup(lane / SLOTS, lane % SLOTS);
        }
    }
}

/// Host oracle: the same lookups as the device `lookup`, run by host rustc.
/// Kept as a separate function so a device bug cannot leak into the oracle.
fn expected(table: usize, k: usize) -> u32 {
    match table {
        0 => BYTES.get(k).map_or(MISSING, |&b| u32::from(b)),
        1 => WORDS.get(k).copied().unwrap_or(MISSING),
        2 => GREETING
            .as_bytes()
            .get(k)
            .map_or(MISSING, |&b| u32::from(b)),
        3 => WORLD.as_bytes().get(k).map_or(MISSING, |&b| u32::from(b)),
        4 => TAIL.get(k).map_or(MISSING, |&v| u32::from(v)),
        5 => FLOATS.get(k).map_or(MISSING, |v| v.to_bits()),
        6 => PAIRS
            .get(k)
            .map_or(MISSING, |&(tag, value)| (u32::from(tag) << 24) | value),
        7 => EMPTY.get(k).map_or(MISSING, |&v| v as u32),
        8 => BYTES.get(k).map_or(MISSING, |&b| u32::from(b) + 1000),
        9 => GREETING
            .as_bytes()
            .get(k)
            .map_or(MISSING, |&b| u32::from(b) + 2000),
        _ => MISSING,
    }
}

/// Launch one lane per (table, slot) pair and compare with the host oracle.
/// Prints SUCCESS only when every lane matches.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== slice_str_constants: &[T] / &str constants over anonymous memory ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let n = TABLES * SLOTS;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n)?;
    // SAFETY: each lane writes only its own element of the n-element buffer.
    unsafe {
        module.slice_str_constants(
            stream.as_ref(),
            LaunchConfig::for_num_elems(n as u32),
            &mut out,
        )
    }?;
    let got = out.to_host_vec(&stream)?;

    let mut failures = 0usize;
    for (lane, &value) in got.iter().enumerate() {
        let (table, k) = (lane / SLOTS, lane % SLOTS);
        let want = expected(table, k);
        if value != want {
            failures += 1;
            eprintln!("table {table} slot {k}: gpu={value:#x} cpu={want:#x}");
        }
    }

    for table in 0..TABLES {
        let row: Vec<u32> = got[table * SLOTS..(table + 1) * SLOTS].to_vec();
        println!("table {table}: {row:?}");
    }

    if failures == 0 {
        println!("\nSUCCESS: {n} lookups over {TABLES} slice/str constants match the host");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures} of {n} lookups differ from the host");
        std::process::exit(1);
    }
}
