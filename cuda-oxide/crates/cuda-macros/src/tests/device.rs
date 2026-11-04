/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device::exported_inline_allow;
use syn::{ItemFn, parse_quote};

/// Render the allow that `#[device]` adds to the exported prefixed function,
/// with whitespace removed, or `None` when it adds no allow.
fn allow_for(function: ItemFn) -> Option<String> {
    exported_inline_allow(&function).map(|tokens| tokens.to_string().replace(' ', ""))
}

/// `#[inline]` and `#[inline(always)]` are the forms rustc reports as ignored
/// on `#[no_mangle]` functions, so both get the allow, in either position
/// relative to other attributes.
#[test]
fn exported_inline_allow_covers_inline_and_inline_always() {
    let expected = Some("#[allow(unused_attributes)]".to_string());
    assert_eq!(allow_for(parse_quote! { #[inline] fn f() {} }), expected);
    assert_eq!(
        allow_for(parse_quote! { #[inline(always)] fn f() {} }),
        expected
    );
    assert_eq!(
        allow_for(parse_quote! { #[cold] #[inline(always)] unsafe fn f() {} }),
        expected
    );
}

/// rustc never reports `#[inline(never)]` on an exported function. An allow
/// there would only widen the lint scope and break crates with
/// `#![forbid(unused_attributes)]` (E0453), so it must not get one.
#[test]
fn exported_inline_allow_skips_inline_never() {
    assert_eq!(allow_for(parse_quote! { #[inline(never)] fn f() {} }), None);
    assert_eq!(
        allow_for(parse_quote! { #[cold] #[inline(never)] fn f() {} }),
        None
    );
}

/// Functions without an `inline` attribute keep their `unused_attributes`
/// diagnostics, and malformed or path-qualified `inline` forms are left to
/// rustc instead of being silenced.
#[test]
fn exported_inline_allow_skips_other_attributes() {
    assert_eq!(allow_for(parse_quote! { fn f() {} }), None);
    assert_eq!(
        allow_for(parse_quote! { #[cold] #[must_use] fn f() -> u32 { 0 } }),
        None
    );
    assert_eq!(
        allow_for(parse_quote! { #[inline(sometimes)] fn f() {} }),
        None
    );
    assert_eq!(
        allow_for(parse_quote! { #[inline(always, never)] fn f() {} }),
        None
    );
    assert_eq!(
        allow_for(parse_quote! { #[inline = "always"] fn f() {} }),
        None
    );
    assert_eq!(allow_for(parse_quote! { #[foo::inline] fn f() {} }), None);
}
