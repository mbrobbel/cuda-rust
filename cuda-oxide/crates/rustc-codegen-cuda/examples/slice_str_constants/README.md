# slice_str_constants

## `&[T]` and `&str` constants in device code

This example checks that slice and string constants work in device code
when their bytes live in memory rustc allocated for the constant itself,
not in a Rust `static`:

```rust
const BYTES: &[u8] = b"cuda-oxide";
const GREETING: &str = "hello, gpu";
const WORLD: &str = GREETING.split_at(7).1;
```

Before the fix, the importer rejected every such operand with
"Unsupported constant type in translate_constant". Only slices over device
statics had a lowering.

## What This Example Does

- Eight `#[inline(never)]` device helpers each return one constant:
  `&[u8]`, `&[u32]`, `&str`, a sub-string, a sub-slice of `u16`, `&[f32]`,
  `&[(u8, u32)]` (padded tuple elements), and an empty `&[u64]`.
- The kernel reads slot `k` of every table for `k` in `0..11`, one lane per
  (table, slot) pair, using the thread index so rustc cannot fold the reads.
  Slots past the end return a marker value, which checks the length word.
- Two tables read `BYTES` and `GREETING` directly in the kernel, without a
  helper.
- The host computes the same lookups and compares all 110 values.

Exits 0 on SUCCESS, 1 on FAIL.

## Pipeline

```text
  const S: &[T] = { data: alloc + addend, len }
       │
       ▼
  @global : [T; len]  (immutable, addrspace 1)
          = alloc bytes [addend .. addend + len * size_of T]
       │
       ▼
  construct_slice(global as *T, len)
```

The viewed bytes become an immutable device global, the same promoted
global a `&[T; N]` constant gets. A sub-slice copies only the bytes it
views. Because the data lives in a global, a helper can return the slice
safely; a stack copy would dangle once the helper returns.

## Run

```bash
cargo oxide run slice_str_constants
```
