# wide_enum_tags

## 128-bit enum tags work on the device

Regression test for enums whose rustc tag is 128 bits wide.

`Option<i128>` has no niche, so rustc stores an explicit tag and widens it
to the payload's 16-byte alignment: a 128-bit tag at byte 0 and the `i128`
at byte 16. `Result<i128, u128>` has the same shape. The importer used to
reject every tag wider than 64 bits:

```text
Unsupported construct: Enum Option uses a 128-bit discriminant; widths above
64 bits are not yet represented losslessly (issue #306)
```

That took out every `i128`/`u128` `checked_*` method (they return
`Option<i128>`/`Option<u128>`) and `i128::pow` (through
`overflowing_pow`).

The MIR dialect stores declared discriminant values as `u64`, and lowering
already zero-extends them to the carrier width. The fix accepts any tag up
to 128 bits as long as every declared value fits in 64 bits, which holds
for every tag rustc widens (the values are 0 and 1) and for small
`#[repr(u128)]` values. A value that needs more than 64 bits still fails
closed; `error_enum_wide_discriminant` pins that.

## What This Example Does

- `decode` matches host-written `Option<i128>` and `Result<i128, u128>`
  values.
- `encode` builds both enums on the device into buffers filled with `0xA5`
  bytes. The host reads each slot back as raw `(tag, payload)` words, so a
  store that misses any of the 16 tag bytes leaves a tag other than 0 or 1
  and fails the check.
- `checked_u128` runs `u128::checked_{add,pow,div}` over edge values.
- `pow_i128` computes `10i128.pow(e)` for `e` in `0..=38`.
- `repr_u128` matches a `#[repr(u128)]` enum whose largest discriminant is
  `u64::MAX`.

Every result is compared against the same computation on the host. Exits 0
on SUCCESS, 1 on FAIL.

## Run

```bash
cargo oxide run wide_enum_tags
```
