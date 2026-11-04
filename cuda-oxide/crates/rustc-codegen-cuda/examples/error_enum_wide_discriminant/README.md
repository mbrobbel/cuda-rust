# `error_enum_wide_discriminant`

This negative test pins the remaining limit on 128-bit enum tags.

```rust
#[repr(u128)]
enum Wide {
    A = 0,
    B = 1 << 64,
}
```

cuda-oxide supports enum tags up to 128 bits wide, which covers the tag
rustc widens for `Option<i128>` and `Result<i128, E>` (see the
`wide_enum_tags` example). The MIR dialect stores declared discriminant
values as `u64`, though, and zero-extends them to the tag width. `B` needs
65 bits, so storing it would truncate it to `A`'s tag. The importer
therefore fails closed for any declared value that does not fit in 64 bits,
which includes every negative `#[repr(i128)]` discriminant.

Run it with:

```bash
cargo oxide build error_enum_wide_discriminant
```

The build must fail with a diagnostic containing:

```text
that does not fit in 64 bits
```
