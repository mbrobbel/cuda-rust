# enum_pointer_int_overlap

## A pointer-sized integer may share a pointer's enum slot

This example checks enums whose variants put a pointer and a pointer-sized
integer on the same bytes:

```rust
Result<&[u8], (usize, usize)>   // {ptr, len} and (usize, usize) at byte 8

#[repr(u64)]
enum PointerOrBits {
    Bits(u64),
    Pointer(*const u8),
}
```

The enum slot map used to reject both with "overlapping pointer and
non-identical storage ... refusing to erase LLVM pointer provenance". The
pointer variant now owns the shared slot as a real `ptr`, so its provenance
is never laundered through an integer. The integer variant shares the slot
and crosses with an explicit `inttoptr` at construction and `ptrtoint` at
payload extraction. `ptrtoint(inttoptr x) == x` at pointer width, so the
integer's value is exact. rustc's own layout represents a scalar
integer/pointer pair the same way.

Overlaps that are not an exact integer view of the pointer (a pointer split
across two `u32`s, a `f64` over a pointer) still fail closed; see
`error_enum_pointer_overlap`.

## What This Example Does

- `checked_lengths`: a non-inlined helper returns
  `Result<&[u8], (usize, usize)>` by value; the kernel folds `Ok` to the
  slice length and `Err` to the requested bit count plus 1000.
- `decode`: reads host-written `PointerOrBits` values (including
  `u64::MAX` and high-bit patterns) from global memory and passes each by
  value to a non-inlined decoder. `Bits` must come back bit-exact and
  `Pointer` must still dereference to the byte it names.
- `encode`: builds both variants on the device and stores them; the host
  checks every word and every device address.

Each kernel is compared against a host oracle. Exits 0 and prints `SUCCESS`
when all three match, 1 otherwise.

## Run

```bash
cargo oxide run enum_pointer_int_overlap
```
