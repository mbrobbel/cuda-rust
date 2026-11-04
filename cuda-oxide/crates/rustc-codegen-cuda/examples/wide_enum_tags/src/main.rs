/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for enums whose rustc tag is 128 bits wide.
//!
//! `Option<i128>` and `Result<i128, u128>` have no niche, so rustc stores an
//! explicit tag and widens it to the payloads' 16-byte alignment: the tag is
//! a 128-bit integer at byte 0 and the payload sits at byte 16. (rustc
//! widens the tag only to the smallest alignment among the variants'
//! leading fields, so `Result<i128, u8>` keeps an 8-bit tag.) The importer
//! used to reject every tag wider than 64 bits ("Enum Option uses a 128-bit
//! discriminant; widths above 64 bits are not yet represented losslessly"),
//! which also took out every `i128`/`u128` `checked_*` method (they return
//! `Option<i128>`) and `i128::pow` (through `overflowing_pow`).
//!
//! The declared discriminants of such enums (0 and 1) still fit in 64 bits,
//! so the dialect stores them as `u64` and lowering zero-extends them to the
//! 128-bit carrier. A discriminant value that really needs more than 64
//! bits (`#[repr(u128)] enum E { A = 1 << 64 }`) is still rejected.
//!
//! Five kernels:
//!
//! - `decode`: matches host-written `Option<i128>` and `Result<i128, u128>`
//!   values, so the device reads the 128-bit tag rustc wrote.
//! - `encode`: builds `Option<i128>` and `Result<i128, u128>` values on the
//!   device into buffers pre-filled with `0xA5` bytes. The host reads them
//!   back as raw `(tag, payload)` words: each tag must be exactly 0 or 1,
//!   so a store that drops any of the 16 tag bytes is caught, and the
//!   decoded value must match the host.
//! - `checked_u128`: `u128::checked_{add,pow,div}` over edge values.
//! - `pow_i128`: `10i128.pow(e)` up to `10^38`.
//! - `repr_u128`: matches a `#[repr(u128)]` enum whose largest discriminant
//!   is `u64::MAX`, the widest value the 64-bit storage admits.
//!
//! Run:
//!     cargo oxide run wide_enum_tags
//!
//! Exits 0 on SUCCESS, 1 on FAIL.

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// `Option<i128>` as a device-buffer element: no niche, 128-bit tag at
/// byte 0, payload at byte 16, 32 bytes in total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct OptI128(pub Option<i128>);

// SAFETY: `Option<i128>` is plain data (no pointers); rustc's layout is the
// same on host and device.
unsafe impl cuda_core::DeviceCopy for OptI128 {}

/// `Result<i128, u128>` as a device-buffer element: the same 128-bit tag
/// shape as `Option<i128>`, with a 16-byte-aligned payload in both variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ResI128(pub Result<i128, u128>);

// SAFETY: `Result<i128, u128>` is plain data (no pointers); rustc's layout is
// the same on host and device.
unsafe impl cuda_core::DeviceCopy for ResI128 {}

/// Fieldless `#[repr(u128)]` enum: a 128-bit tag whose declared values all
/// fit in 64 bits, up to and including `u64::MAX`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u128)]
pub enum WideTag {
    One = 1,
    Five = 5,
    Max64 = u64::MAX as u128,
}

// SAFETY: trivial repr(u128) POD, safe to copy device<->host.
unsafe impl cuda_core::DeviceCopy for WideTag {}

/// Elements per buffer; one thread per element.
const N: usize = 32;

#[cuda_module]
mod kernels {
    use super::*;

    /// Reads host-written wide-tag enums and folds both into one `i128`.
    #[kernel]
    pub fn decode(opts: &[OptI128], results: &[ResI128], mut out: DisjointSlice<i128>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < opts.len()
            && i < results.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = decode_one(opts[i], results[i]);
        }
    }

    /// Builds wide-tag enums on the device from `i128` checked arithmetic.
    #[kernel]
    pub fn encode(
        xs: &[i128],
        mut opts: DisjointSlice<OptI128>,
        mut results: DisjointSlice<ResI128>,
    ) {
        let i = thread::index_1d().get();
        if i >= xs.len() {
            return;
        }
        // `get_mut` consumes the thread index, so take one per slice.
        if let Some(slot) = opts.get_mut(thread::index_1d()) {
            *slot = encode_option(xs[i]);
        }
        if let Some(slot) = results.get_mut(thread::index_1d()) {
            *slot = encode_result(xs[i], i);
        }
    }

    /// Runs `u128::checked_*`, each of which returns an `Option<u128>`.
    #[kernel]
    pub fn checked_u128(ys: &[u128], mut out: DisjointSlice<u128>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = checked_u128_mix(ys[i], i);
        }
    }

    /// Runs `i128::pow`, which goes through `overflowing_pow` and so
    /// through `Option<i128>`.
    #[kernel]
    pub fn pow_i128(exps: &[u32], mut out: DisjointSlice<i128>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < exps.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = 10i128.pow(exps[i]);
        }
    }

    /// Matches and casts a host-written `#[repr(u128)]` enum.
    #[kernel]
    pub fn repr_u128(tags: &[WideTag], mut out: DisjointSlice<u128>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < tags.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = wide_tag_score(tags[i]);
        }
    }
}

/// Folds one `Option<i128>` and one `Result<i128, u128>` into an `i128`.
/// Shared by the `decode` kernel and the host oracle.
fn decode_one(opt: OptI128, result: ResI128) -> i128 {
    let a = match opt.0 {
        Some(v) => v,
        None => -7,
    };
    let b = match result.0 {
        Ok(v) => v,
        Err(e) => (e as i128).wrapping_add(1000),
    };
    a.wrapping_mul(3).wrapping_add(b)
}

/// `x * x - 1` when neither step overflows, else `None`.
fn encode_option(x: i128) -> OptI128 {
    OptI128(x.checked_mul(x).and_then(|v| v.checked_sub(1)))
}

/// `Ok` for an even `x + i`, `Err(i << 64)` for an odd one, `Err(MAX)` when
/// the sum overflows.
fn encode_result(x: i128, i: usize) -> ResI128 {
    ResI128(match x.checked_add(i as i128) {
        Some(v) if v % 2 == 0 => Ok(v),
        Some(_) => Err((i as u128) << 64),
        None => Err(u128::MAX),
    })
}

/// XOR of three `u128` checked operations, each with its own fallback so a
/// `None` taken on the wrong side shows up in the result.
fn checked_u128_mix(y: u128, i: usize) -> u128 {
    y.checked_add(y).unwrap_or(1)
        ^ y.checked_pow(i as u32).unwrap_or(2)
        ^ (y.checked_div(i as u128).unwrap_or(3) << 1)
}

/// Mixes a `match` on the 128-bit tag with an `as u128` cast of it.
fn wide_tag_score(tag: WideTag) -> u128 {
    let arm: u128 = match tag {
        WideTag::One => 100,
        WideTag::Five => 200,
        WideTag::Max64 => 300,
    };
    (tag as u128).wrapping_add(arm << 64)
}

/// `i128` inputs covering small values, values near `MAX` and `MIN`, and
/// values with bits set above bit 64.
fn i128_inputs() -> Vec<i128> {
    (0..N as i128)
        .map(|i| match i % 4 {
            0 => i,
            1 => i128::MAX - i,
            2 => -(i << 70),
            _ => i128::MIN + i,
        })
        .collect()
}

/// `u128` inputs covering values near `MAX` and values above `u64::MAX`.
fn u128_inputs() -> Vec<u128> {
    (0..N as u128)
        .map(|i| {
            if i % 3 == 0 {
                u128::MAX - i
            } else {
                i * 12_345_678_901_234_567_890
            }
        })
        .collect()
}

/// Fill byte for the `encode` outputs: a tag word that keeps any of these
/// bytes is neither 0 nor 1.
const GARBAGE: u128 = u128::from_ne_bytes([0xA5; 16]);

/// One 32-byte wide-tag enum read back as plain words: `[tag, payload]`.
type RawSlot = [u128; 2];

/// Allocates `N` 32-byte enum slots filled with [`GARBAGE`] bytes.
fn garbage_slots<T: DeviceCopy>(
    stream: &Arc<CudaStream>,
) -> Result<DeviceBuffer<T>, Box<dyn std::error::Error>> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<RawSlot>());
    let words = DeviceBuffer::from_host(stream, &vec![GARBAGE; 2 * N])?;
    let (ptr, _, ctx) = words.into_raw_parts();
    // SAFETY: `ptr` is the synchronous `cuMemAlloc` allocation of `2 * N`
    // u128s (`N` 32-byte slots) that `from_host` made in `ctx`; ownership
    // moves to the new buffer. The kernel writes every slot before the host
    // reads them back, and the host reads them only through `raw_slots`.
    Ok(unsafe { DeviceBuffer::from_raw_parts(ptr, N, ctx) })
}

/// Reads `N` 32-byte enum slots back as raw words, so a slot whose tag the
/// device left partly unwritten never becomes an invalid enum on the host.
fn raw_slots<T: DeviceCopy>(
    stream: &Arc<CudaStream>,
    slots: DeviceBuffer<T>,
) -> Result<Vec<RawSlot>, Box<dyn std::error::Error>> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<RawSlot>());
    let (ptr, len, ctx) = slots.into_raw_parts();
    // SAFETY: `ptr` came from `garbage_slots`: a `cuMemAlloc` allocation of
    // `len` 32-byte slots in `ctx`, i.e. `2 * len` u128s; ownership moves to
    // the new buffer, and every u128 bit pattern is valid.
    let words = unsafe { DeviceBuffer::<u128>::from_raw_parts(ptr, 2 * len, ctx) };
    Ok(words
        .to_host_vec(stream)?
        .chunks_exact(2)
        .map(|word| [word[0], word[1]])
        .collect())
}

/// Decodes a raw `Option<i128>` slot; `None` for a tag that is neither
/// variant.
fn decode_option(raw: RawSlot) -> Option<OptI128> {
    match raw[0] {
        0 => Some(OptI128(None)),
        1 => Some(OptI128(Some(raw[1] as i128))),
        _ => None,
    }
}

/// Decodes a raw `Result<i128, u128>` slot; `None` for a tag that is neither
/// variant.
fn decode_result(raw: RawSlot) -> Option<ResI128> {
    match raw[0] {
        0 => Some(ResI128(Ok(raw[1] as i128))),
        1 => Some(ResI128(Err(raw[1]))),
        _ => None,
    }
}

/// Checks the host's own layout matches what [`decode_option`] and
/// [`decode_result`] assume: a 128-bit tag word, then the payload word.
fn check_raw_layout() {
    // SAFETY: `Some` of an i128 defines all 32 bytes (the whole tag and
    // the payload), and every bit pattern is a valid `[u128; 2]`.
    let some: RawSlot = unsafe { std::mem::transmute(OptI128(Some(-2))) };
    // SAFETY: as above, for both `Result` variants.
    let ok: RawSlot = unsafe { std::mem::transmute(ResI128(Ok(-2))) };
    // SAFETY: as above.
    let err: RawSlot = unsafe { std::mem::transmute(ResI128(Err(7))) };
    assert_eq!(some, [1, (-2i128) as u128], "Option<i128> layout");
    assert_eq!(ok, [0, (-2i128) as u128], "Result<i128, u128> layout");
    assert_eq!(err, [1, 7], "Result<i128, u128> layout");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== wide_enum_tags: Option<i128> / Result<i128, u128> on the device ===\n");
    check_raw_layout();

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let xs = i128_inputs();
    let ys = u128_inputs();
    let opts: Vec<OptI128> = xs
        .iter()
        .enumerate()
        .map(|(i, &v)| OptI128(if i % 3 == 0 { None } else { Some(v) }))
        .collect();
    let results: Vec<ResI128> = xs
        .iter()
        .enumerate()
        .map(|(i, &v)| ResI128(if i % 2 == 0 { Err(!(i as u128)) } else { Ok(v) }))
        .collect();
    // 0, 1, ..., 38: 10^38 is the largest power of ten an i128 holds.
    let exps: Vec<u32> = (0..N as u32).map(|e| e * 38 / (N as u32 - 1)).collect();
    let tags: Vec<WideTag> = (0..N)
        .map(|i| [WideTag::One, WideTag::Five, WideTag::Max64][i % 3])
        .collect();

    let d_opts = DeviceBuffer::from_host(&stream, &opts)?;
    let d_results = DeviceBuffer::from_host(&stream, &results)?;
    let d_xs = DeviceBuffer::from_host(&stream, &xs)?;
    let d_ys = DeviceBuffer::from_host(&stream, &ys)?;
    let d_exps = DeviceBuffer::from_host(&stream, &exps)?;
    let d_tags = DeviceBuffer::from_host(&stream, &tags)?;
    let mut d_decoded = DeviceBuffer::<i128>::zeroed(&stream, N)?;
    let mut d_enc_opts = garbage_slots::<OptI128>(&stream)?;
    let mut d_enc_results = garbage_slots::<ResI128>(&stream)?;
    let mut d_checked = DeviceBuffer::<u128>::zeroed(&stream, N)?;
    let mut d_pows = DeviceBuffer::<i128>::zeroed(&stream, N)?;
    let mut d_scores = DeviceBuffer::<u128>::zeroed(&stream, N)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.decode(&stream, cfg, &d_opts, &d_results, &mut d_decoded) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.encode(&stream, cfg, &d_xs, &mut d_enc_opts, &mut d_enc_results) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_u128(&stream, cfg, &d_ys, &mut d_checked) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.pow_i128(&stream, cfg, &d_exps, &mut d_pows) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.repr_u128(&stream, cfg, &d_tags, &mut d_scores) }?;

    let decoded = d_decoded.to_host_vec(&stream)?;
    let enc_opts = raw_slots(&stream, d_enc_opts)?;
    let enc_results = raw_slots(&stream, d_enc_results)?;
    let checked = d_checked.to_host_vec(&stream)?;
    let pows = d_pows.to_host_vec(&stream)?;
    let scores = d_scores.to_host_vec(&stream)?;

    let mut failures = 0usize;
    let mut check = |name: &str, i: usize, ok: bool, detail: String| {
        if !ok {
            failures += 1;
            eprintln!("[{i}] {name}: {detail}");
        }
    };
    for i in 0..N {
        let want = decode_one(opts[i], results[i]);
        check(
            "decode",
            i,
            decoded[i] == want,
            format!("gpu={} cpu={want}", decoded[i]),
        );
        let want = encode_option(xs[i]);
        let [tag, _] = enc_opts[i];
        check(
            "encode Option<i128>",
            i,
            decode_option(enc_opts[i]) == Some(want),
            format!(
                "gpu tag={tag:#x} value={:?} cpu={want:?}",
                decode_option(enc_opts[i])
            ),
        );
        let want = encode_result(xs[i], i);
        let [tag, _] = enc_results[i];
        check(
            "encode Result<i128, u128>",
            i,
            decode_result(enc_results[i]) == Some(want),
            format!(
                "gpu tag={tag:#x} value={:?} cpu={want:?}",
                decode_result(enc_results[i])
            ),
        );
        let want = checked_u128_mix(ys[i], i);
        check(
            "u128 checked_*",
            i,
            checked[i] == want,
            format!("gpu={} cpu={want}", checked[i]),
        );
        let want = 10i128.pow(exps[i]);
        check(
            "i128 pow",
            i,
            pows[i] == want,
            format!("10^{} gpu={} cpu={want}", exps[i], pows[i]),
        );
        let want = wide_tag_score(tags[i]);
        check(
            "repr(u128) enum",
            i,
            scores[i] == want,
            format!("{:?} gpu={:#x} cpu={want:#x}", tags[i], scores[i]),
        );
    }

    println!("decode[0..4]       = {:?}", &decoded[..4]);
    println!(
        "encode Option[0..4] = {:?}",
        enc_opts[..4]
            .iter()
            .map(|&raw| decode_option(raw))
            .collect::<Vec<_>>()
    );
    println!("10^{} on the device = {}", exps[N - 1], pows[N - 1]);

    if failures == 0 {
        println!("\nSUCCESS: {N} elements x 6 checks match the host");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures} mismatches across {N} elements x 6 checks");
        std::process::exit(1);
    }
}
