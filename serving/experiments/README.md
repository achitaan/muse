# Preserved native-Marlin revisions

`native_v1/` preserves the direct-layout prototype with 2^57 operand rebasing.
Its CUDA SHA256 is `d8c77db6275fd442186af20572fa5e8c61e0b32a1520febc3e153091132ba831`.
`native_v2/` preserves direct encoded-scale accumulation with the original FP32
global atomic fixups; CUDA SHA256 is
`4e62622378fe71d268dea17d77644d90340a90ff7c269e634b222bceb4287a3c`.

These are immutable source snapshots, not independently packaged entrypoints.
Their generator paths assume placement in the repository's `serving/` directory.
The final implementation and builders live there, with a separate dormant patch.
Do not substitute v1's faster hot-route timings for the final implementation.
The numerical evidence and stopping decision are in [the handoff](../PRECOMP_HANDOFF.md).
