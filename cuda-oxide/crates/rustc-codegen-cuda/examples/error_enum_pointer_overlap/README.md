# `error_enum_pointer_overlap`

This negative test protects LLVM pointer provenance in enum payload storage.

```rust
#[repr(C)]
enum PointerOrBits {
    Bits(u32, u32),
    Pointer(*const u8),
}
```

Rust lays the pointer over the same eight bytes as the two `u32` halves.
cuda-oxide cannot model that storage without reinterpreting the pointer's
bytes as two integers or building a pointer from two integer halves with no
explicit cast. It therefore fails closed.

A pointer-sized integer that overlays the pointer exactly (`Bits(u64)`, or
`Result<&[u8], (usize, usize)>`) is supported: it shares the pointer's slot
through explicit `inttoptr`/`ptrtoint` (see the `enum_pointer_int_overlap`
example). This fixture keeps the partial overlap, which has no such exact
integer view, as an expected failure.

Run it with:

```bash
cargo oxide build error_enum_pointer_overlap
```

The build must fail with a diagnostic containing:

```text
overlapping pointer and non-identical storage
```
