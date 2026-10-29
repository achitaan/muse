# Conditional Decode transfer parked

CPU-only source5af9eb556ffbb5d38c7ce5a3e77d063a0e7f0f4b007ffac4ed252a580ca6ac39,
893421bytes,106579bytes below the1MBsource cap. No Decode GPU job or active
candidate change. It is not ready for GPU submission.

The draft applies exactly three identity-rescale arithmetic edits to the
native Attention CUDA in preparation package307dd791. An inverse restores
that CUDA and every parent package byte; all other package Python AST is exact.
The D9 FP16 PV backend and all BF16 boundaries remain. The standalone tied
FP8-PV change is not transferred. Source packaging and preparation cleanup
remain the previously CPU-audited drafts, without measured preparation gains.

Attentioned810c7a passed its full suite but failed the fixed affected/control
performance gate. Therefore the condition for evaluating this transfer was
not met. Do not spend a Decode check on it, enable it or promote it. Preserve
the proof and draft for retrieval; any later justified integration needs exact
resource/execution markers and whole-layer routing/cache/residual/weighted
latency evidence. The verified active Decode remains D9.

Reproduce the source draft with
`python experiments/prepared/build_decoder_skipscale_draft.py`.
