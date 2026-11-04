/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Enums whose variants overlay a pointer and a pointer-sized integer.
//!
//! `Result<&[u8], (usize, usize)>` puts the slice's `{ptr, len}` and the
//! error's `(usize, usize)` on the same sixteen bytes, and
//! `enum PointerOrBits { Bits(u64), Pointer(*const u8) }` puts a `u64` and a
//! pointer on the same eight. The enum slot map used to reject both
//! ("overlapping pointer and non-identical storage ... refusing to erase LLVM
//! pointer provenance"). Now the pointer owns the shared slot as a real
//! `ptr`, and the integer variant crosses into and out of it with explicit
//! `inttoptr`/`ptrtoint`, which preserves its value exactly.
//!
//! Three kernels, each checked against a host oracle:
//!
//! - `checked_lengths`: the reported shape. A non-inlined helper returns
//!   `Result<&[u8], (usize, usize)>` by value; `Ok` carries the slice,
//!   `Err` the requested bit count.
//! - `decode`: reads host-written `PointerOrBits` values from global memory
//!   and passes each by value to a non-inlined decoder. `Bits` must come
//!   back bit-exact; `Pointer` must still dereference to the byte it names.
//! - `encode`: builds both variants on the device (the integer variant is
//!   declared first) and stores them for the host to inspect.
//!
//! Run:
//!     cargo oxide run enum_pointer_int_overlap
//!
//! Exits 0 on SUCCESS, 1 on FAIL.

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Bit counts `checked_lengths` asks a four-byte slice for: 0, 16 and 32 fit
/// (`Ok`), 33 and the large value do not (`Err`).
const BIT_REQUESTS: [u32; 5] = [0, 16, 32, 33, 4_000_000_000];

/// Added to a dereferenced `Pointer` byte so the host can tell the two
/// variants apart in one `u64` output.
const POINTER_TAG: u64 = 2_000;

/// Added to the requested bit count of an `Err` so it cannot collide with an
/// `Ok` slice length.
const ERR_TAG: u64 = 1_000;

/// An eight-byte payload that is either raw bits or a pointer into device
/// memory. `repr(u64)` puts the tag in the first word and both payloads in
/// the second, so the two variants share exactly one pointer-sized slot and
/// the type has no padding to copy.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerOrBits {
    /// Raw bits; must survive the shared pointer slot unchanged.
    Bits(u64),
    /// A pointer into a device buffer; must stay dereferenceable.
    Pointer(*const u8),
}

// SAFETY: `PointerOrBits` is two `u64`-sized words without padding, and the
// all-zero pattern is `Bits(0)`. Its pointers are device addresses that the
// host only compares and never dereferences.
unsafe impl DeviceCopy for PointerOrBits {}

/// Return the whole slice when it holds at least `bits` bits, else the
/// request and the slice length as an error.
///
/// `#[inline(never)]` keeps the enum crossing a call boundary by value, so
/// both the construction (`Err` stores two integers into `{ptr, i64}`) and
/// the caller's payload read are exercised.
#[inline(never)]
fn checked(bytes: &[u8], bits: usize) -> Result<&[u8], (usize, usize)> {
    if bits <= bytes.len() * 8 {
        Ok(bytes)
    } else {
        Err((bits, bytes.len()))
    }
}

/// Decode one `PointerOrBits` passed by value: `Bits` as-is, `Pointer` as
/// the byte it points at plus [`POINTER_TAG`].
///
/// # Safety
///
/// A `Pointer` value must point to a readable byte of device memory.
#[inline(never)]
unsafe fn decode_value(value: PointerOrBits) -> u64 {
    match value {
        PointerOrBits::Bits(bits) => bits,
        // SAFETY: forwarded from this function's contract.
        PointerOrBits::Pointer(pointer) => POINTER_TAG + u64::from(unsafe { *pointer }),
    }
}

/// Build `Pointer(bytes + word % len)` for odd words and `Bits(!word)` for
/// even ones, so both variants are constructed by the same function.
#[inline(never)]
fn encode_value(word: u64, bytes: &[u8]) -> PointerOrBits {
    if word % 2 == 1 {
        let offset = (word % bytes.len() as u64) as usize;
        PointerOrBits::Pointer(bytes[offset..].as_ptr())
    } else {
        PointerOrBits::Bits(!word)
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = checked(bytes, bits[i])` folded to one `u64`: the slice
    /// length for `Ok`, `bits + ERR_TAG` for `Err`.
    #[kernel]
    pub fn checked_lengths(bytes: &[u8], bits: &[u32], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < bits.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = match checked(bytes, bits[i] as usize) {
                Ok(slice) => slice.len() as u64,
                Err((requested, _len)) => requested as u64 + ERR_TAG,
            };
        }
    }

    /// `out[i] = decode_value(entries[i])` for host-written entries.
    ///
    /// # Safety
    ///
    /// Every `Pointer` entry must point to a readable byte of device memory.
    #[kernel]
    pub unsafe fn decode(entries: &[PointerOrBits], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < entries.len()
            && let Some(slot) = out.get_mut(idx)
        {
            // SAFETY: forwarded from this kernel's contract.
            *slot = unsafe { decode_value(entries[i]) };
        }
    }

    /// `out[i] = encode_value(words[i], bytes)`, stored for the host.
    #[kernel]
    pub fn encode(words: &[u64], bytes: &[u8], mut out: DisjointSlice<PointerOrBits>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < words.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = encode_value(words[i], bytes);
        }
    }
}

/// Run all three kernels and compare each against its host oracle.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== enum_pointer_int_overlap ===");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut failures = 0usize;

    let host_bytes: [u8; 4] = [11, 22, 33, 44];
    let bytes = DeviceBuffer::from_host(&stream, &host_bytes)?;
    let bytes_base = bytes.cu_deviceptr();

    // checked_lengths: Result<&[u8], (usize, usize)> by value.
    let bits = DeviceBuffer::from_host(&stream, &BIT_REQUESTS)?;
    let mut lengths = DeviceBuffer::<u64>::zeroed(&stream, BIT_REQUESTS.len())?;
    let cfg = LaunchConfig::for_num_elems(BIT_REQUESTS.len() as u32);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_lengths(&stream, cfg, &bytes, &bits, &mut lengths) }?;
    let got = lengths.to_host_vec(&stream)?;
    let want: Vec<u64> = BIT_REQUESTS
        .iter()
        .map(|&b| match checked(&host_bytes, b as usize) {
            Ok(slice) => slice.len() as u64,
            Err((requested, _len)) => requested as u64 + ERR_TAG,
        })
        .collect();
    println!("checked_lengths: {got:?}");
    if got != want {
        println!("FAIL checked_lengths: got {got:?}, want {want:?}");
        failures += 1;
    }

    // decode: host-written values, read back through the shared slot.
    let entries_host: Vec<PointerOrBits> = vec![
        PointerOrBits::Bits(0),
        PointerOrBits::Bits(u64::MAX),
        PointerOrBits::Bits(0x8000_0000_0000_0001),
        PointerOrBits::Pointer(bytes_base as *const u8),
        PointerOrBits::Pointer((bytes_base + 3) as *const u8),
        PointerOrBits::Bits(0x0123_4567_89ab_cdef),
    ];
    let entries = DeviceBuffer::from_host(&stream, &entries_host)?;
    let mut decoded = DeviceBuffer::<u64>::zeroed(&stream, entries_host.len())?;
    let cfg = LaunchConfig::for_num_elems(entries_host.len() as u32);
    // SAFETY: launch shape/resources match the kernel; buffers cover its
    // accesses, and every `Pointer` entry names a byte of `bytes`.
    unsafe { module.decode(&stream, cfg, &entries, &mut decoded) }?;
    let got = decoded.to_host_vec(&stream)?;
    let want: Vec<u64> = entries_host
        .iter()
        .map(|entry| match *entry {
            PointerOrBits::Bits(bits) => bits,
            PointerOrBits::Pointer(pointer) => {
                let offset = (pointer as u64 - bytes_base) as usize;
                POINTER_TAG + u64::from(host_bytes[offset])
            }
        })
        .collect();
    println!("decode: {got:?}");
    if got != want {
        println!("FAIL decode: got {got:?}, want {want:?}");
        failures += 1;
    }

    // encode: both variants built on the device, inspected on the host.
    let words_host: Vec<u64> = vec![0, 1, 2, 7, u64::MAX - 1, u64::MAX];
    let words = DeviceBuffer::from_host(&stream, &words_host)?;
    let mut encoded = DeviceBuffer::<PointerOrBits>::zeroed(&stream, words_host.len())?;
    let cfg = LaunchConfig::for_num_elems(words_host.len() as u32);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.encode(&stream, cfg, &words, &bytes, &mut encoded) }?;
    let got = encoded.to_host_vec(&stream)?;
    let want: Vec<PointerOrBits> = words_host
        .iter()
        .map(|&word| {
            if word % 2 == 1 {
                let offset = word % host_bytes.len() as u64;
                PointerOrBits::Pointer((bytes_base + offset) as *const u8)
            } else {
                PointerOrBits::Bits(!word)
            }
        })
        .collect();
    println!("encode: {got:?}");
    if got != want {
        println!("FAIL encode: got {got:?}, want {want:?}");
        failures += 1;
    }

    if failures == 0 {
        println!("SUCCESS: enum_pointer_int_overlap (all three kernels match the host)");
        Ok(())
    } else {
        println!("FAIL: enum_pointer_int_overlap ({failures} kernel(s) mismatched)");
        std::process::exit(1);
    }
}
