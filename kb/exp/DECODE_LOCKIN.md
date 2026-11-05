# Decode lock-in

Check file: `kb/exp/decode_lockin.py` (983,756 bytes).
Parent: `ed853c1` (`loop/dec-combo`). Build reproducibly with
`python3 kb/exp/build_decode_lockin.py`.

- d1F defaults to KV, v1S defaults to POOL, subject only to correctness.
  d1S/v1F and every other shape always use the incumbent. Compatibility
  constraints from COMBO are unchanged, restricted to the two target shapes.
- Prepare generates one fixed-seed synthetic input per target shape. It calls
  the incumbent once and the candidate once; no graph captures/warmups/replays,
  flush buffers, null controls, timing events or performance admission remain
  in this appended arm selector. Incumbent preparation/tuning is unchanged.
- Ordered expert IDs must equal the incumbent. BF16 hidden, FP32 stream,
  carry-subtracted residual update, and new FP8 K/V rows must meet the inherited
  public-reference judge tolerances AND have relative-L2/worst-row error no
  greater than this single incumbent call. No epsilon or error floor is added.
  All non-destination cache bytes must remain unchanged. Public routing validity
  and guarded workspace checks remain enabled.
- Shape-specific failures print `DEC_LOCKIN shape arm ids_equal err_new err_inc
  used why` and keep that shape incumbent. Unavailable builds disable their own
  arm independently. Global preparation/guard/cleanup failures disable both.
- Incumbent prefix, candidate source transforms, lossless packing, reference
  math, and all load_inline build flags are unchanged (AST/byte verified).
  Builds remain threaded in-process load_inline calls; no process-launch code.
- Joins retain the +230 s cutoff; correctness uses the +275 s soft deadline.
  Actual <285 s preparation cannot be proven without GPU execution: inherited
  prepare, CUDA compilation and asynchronous device operations are not
  preemptible. Caller must verify GPU duration and correctness.

## Local validation

- `python3 kb/exp/build_decode_lockin.py`: AST/size passed.
- `python3 kb/exp/test_decode_lockin_cpu.py -v`: six passed, two skipped
  (system Python lacks torch).
- With the existing combo CPU-torch venv: all eight lock-in tests passed,
  including strict no-worse errors, zero-error caches, ordered IDs, excluded
  shapes, TLS/launch-global restoration, independent build fallbacks, exact
  one-input/two-call reset/reference protocol, and no timing gate.
- Parent KV suite: all nine passed. Parent COMBO suite with CPU torch:
  thirteen passed, public-task-reference comparison skipped (path unset).
  New AST tests prove every inherited reference helper is identical.
- Forbidden process/versioned-graph grep and `git diff --check`: passed.

No local GPU/nvcc, GPU check/submission, push, or main commit. The configured
ledger helper is absent; the claim/result is local to this branch because
caller forbids main writes. Caller owns the GPU check and ledger result.
