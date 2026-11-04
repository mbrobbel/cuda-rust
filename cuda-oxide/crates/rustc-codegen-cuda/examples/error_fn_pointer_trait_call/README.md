# `error_fn_pointer_trait_call`

Negative test for a `fn` pointer passed to a generic `F: FnMut(..)`
parameter. Inside `apply`, `f(x)` is a `<F as FnMut>::call_mut` call, which
for `F = fn(u32) -> u32` resolves to rustc's fn-pointer shim. The shim's body
calls through the pointer, so the collector never gathers it. Before the
importer refused the call, it was emitted against the shim's symbol and only
failed at module verification. The `Map<_, fn(&T) -> U>` iterator idiom
calls through the pointer the same way and gets the same diagnostic.

On the device a `fn` pointer is a stable identity token, not a code address
(see `examples/fn_ptr_naming`), so the compiler must reject the call at MIR
import instead:

```bash
cargo oxide build error_fn_pointer_trait_call
```

Expected diagnostic (pinned by `cuda-oxide/scripts/smoketest.sh`):

```text
calling through a function pointer
```

Passing the fn item itself (`apply(inc, x)`) or a closure stays supported,
because those types name their callee statically. A diverging call through a
`fn` pointer still lowers to the device trap.
