# `error_fn_pointer_call`

Negative test for a call through a `fn` pointer value in device code
(`apply(f, x)` calling `f(x)` with `f: fn(u32) -> u32` chosen per lane).

On the device a `fn` pointer is a stable identity token (a hash of the
function's mangled name), not a code address: it can be created, stored and
compared (see `examples/fn_ptr_naming`), but there is nothing to call. Before
the importer refused the call, it was emitted against a placeholder symbol
`unknown_function` and only failed at module verification, with an error that
named neither the caller nor the cause.

The compiler must reject the call at MIR import instead, naming the calling
function and the pointer type:

```bash
cargo oxide build error_fn_pointer_call
```

Expected diagnostic (pinned by `cuda-oxide/scripts/smoketest.sh`):

```text
calling through a function pointer
```

Call the function directly, or pass the fn item or a closure to a generic
`F: Fn(..)` parameter (see `examples/function_item_call` and
`examples/device_closures`). `error_fn_pointer_trait_call` covers the same
pointer reaching a generic `F: FnMut(..)` parameter.
