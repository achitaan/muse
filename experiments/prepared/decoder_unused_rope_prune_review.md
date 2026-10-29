# Decode: bypass an unread RoPE preparation table

CPU draft `307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072`
is 893,102 bytes. It derives from df68deb6 and is frozen in
`results/decoder-unused-rope-prune-20261029/`. No GPU validation, preparation
measurement, timed speedup or promotion is claimed.

The original base preparation builds 2^21 positions by 32 FP32 frequencies,
then concatenates their sine and cosine into a 2^21 by 64 FP32 table. The
table's tensor payload is exactly536,870,912bytes (512MiB). During the cat
expression, frequencies, separate sine/cosine tensors and output represent
1,342,177,280bytes (1.25GiB), excluding allocator/workspace and other state.
These are representation sizes, not observed peak allocation or traffic.

The current whole-layer runtime receives attention output directly from
`_native_attn_kernel_fn(data)`, whose native and fallback implementations use
their existing data/configuration paths. The old root-state `cos_sin` key is
never read. The draft removes the two frequency construction assignments and
the matching delete, and keeps `cos_sin` as None. Every other module AST node,
all numerical CUDA, attention fallbacks, precision boundaries, state keys,
other norms and runtime functions remain exact.

The independent root-state audit follows the final bindings of kernel_fn and
all five helpers receiving the whole root state. It rejects dynamic root
indexing, aliases and escapes to an unknown function. Every root-state read
has a constant key; none reads cos_sin. Nested states, weights and objects
retain their previous values. This source-bound proof covers all branches of
those helpers, rather than only the scored token shapes.

CPU reproduction:

```text
python experiments/prepared/build_decoder_unused_rope_prune.py
python experiments/prepared/verify_decoder_unused_rope_reachability.py
```

Use this draft only with a justified later Decode runtime integration. A full
13-test/12-benchmark/integrity check of the exact integrated source is required
before promotion. Do not spend a standalone check or public score merely to
retime preparation cleanup. D9 remains the verified/active Decode source.
