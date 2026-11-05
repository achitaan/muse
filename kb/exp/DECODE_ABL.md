# Prepare-only full-layer launch-group ablation

Check file: `kb/exp/decode_layer_abl.py` (999,285 bytes).

Its first 977,700 bytes are exactly the incumbent `decode_attn_fp16.py` used by
`bc0b306ed55449a8`, SHA256
`96757583753818779e0a10f8f0f5ff1ba1d50e9848588521c2e8dca44af038b3`.
The appended prepare wrapper suppresses the old attention admission experiment
(`_AF_KINDS=()` temporarily), retaining the requested incumbent `af_pick={}`.
No production function or backend selection is replaced after prepare.

## Method

- Synthetic contract-shaped data, actual supplied weights, busiest local expert
  rank, all 12 shapes. Priority: d16, d1, v1, d32, v16, v32; S then F.
- Capture real full layer once per shape. Temporary Python hooks attribute graph
  nodes to components and retain intermediate tensors; hooks are then restored.
- Clone the CUDA graph and replace ONE component's kernel nodes with a native
  no-op: original grid, one warp, zero dynamic shared memory, PDL wait/trigger.
  Clone + kernel parameter update preserves graph edges, PDL edge metadata,
  cooperative/cluster attributes. Non-kernel selected nodes fail closed.
- Run the real graph first; save all component outputs and written KV slots.
  Copy these real values back before EVERY replay, then READ-flush 512 MiB,
  then bracket only the graph replay with CUDA events. Route IDs must agree
  exactly on EVERY timed replay, including baseline/null. Changed routing
  discards the arm, rather than inflating upstream savings (N3 correction).
- Twelve paired ABBA/BAAB rounds after two discarded warm rounds, per arm and
  base-vs-base null. Private activation/counter scratch; shared Lt workspace and
  global scratch references restored. No diagnostic output is ever adopted.
- No new shape after 267 s; timing stops at 280 s, leaving cleanup margin to the
  300 s prepare limit. CUDA compilation/allocations cannot be forcibly preempted.

## Log interpretation

`DEC_ABL shape component saved_us_median saved_us_p10 null_us ...`

Saving is **base minus no-op arm**, median/p10 of paired round differences.
`null_us` is the signed median base-vs-base difference. Optional fields include
base/arm medians, 12 rounds, actual launch-group label, exact-route check.

`DEC_ABL_GRAPH` gives graph node/edge counts, component node counts, local rank.
`DEC_ABL_SUMMARY` ranks measured components by median microseconds saved.
`DEC_ABL_WHATIF` linearly applies 25% of each positive measured saving to the
provided board-time anchor, weighting 46 sliding + 16 full per load and taking
the geomean over six loads. It reports multiplicative board gain, gain percent,
conditional score from 2.288, and a conservative p10-minus-absolute-null version.
Unmeasured shapes contribute zero; coverage is printed. This is an estimate,
not a measured 25%-faster implementation, and components are not additive.

**d1 fusion:** o-projection contains part of boundary/norm, while routing contains
the remaining carry. Its `o_boundary` is `nan/not_separable_or_absent`, not a
fabricated independent measurement. d1 o/route and their what-ifs include those
fused parts, explicitly labeled `o_proj+boundary_partial` and
`route+boundary_carry`. Removing them separately requires kernel surgery and is
not launch ablation. A no-op retains nonzero launch/scheduling cost, so measured
savings are conservative launch-group ceilings, not complete mathematical costs.

`DEC_ABL_SKIP/ERROR/ABORT` explain missing measurements. The component diagnostics
require PyTorch `CUDAGraph(keep_graph=True)` and CUDA graph edge-data APIs (CUDA
>=12.3). CUDA 13 uses the unsuffixed edge-data APIs; CUDA 12 uses the suffixed
`cudaStreamGetCaptureInfo_v3`/`cudaGraphGetEdges_v2` names, selected at compile time.
Helper builds explicitly use C++20 and SM90a; incumbent flags remain untouched.
Build failures print the FIRST compiler diagnostics, not a trailing log fragment.
Without the helper, ordinary PyTorch graph replays still measure 12-round
same-graph base-vs-base nulls at each shape. Component arms are explicitly skipped:
they cannot safely retain PDL dependencies without the compiled no-op helper.

## Validation

`python3 -m unittest discover -s kb/exp -p 'test_decode_layer_abl.py' -v`

Eleven CPU tests pass: AST/exact prefix/size, score weights, quantiles, node ownership,
nested hooks, source-verified output argument indices, hook identity restoration,
ABBA/flush/event order and sample counts, null, deadline discard, per-sample
routing rejection, exception depth cleanup, prepare fallback/global restoration,
native dependency-preservation contract, first-error retention and helper-free
null replay/prepare continuation. Direct prefix diff is empty.

Both CUDA 12 and CUDA 13 **signature contracts** pass clang++ syntax checking:

```
python3 kb/exp/decode_abl_api_contract.py | clang++ -std=c++20 -fsyntax-only -x c++ - -DCUDART_VERSION=12080
python3 kb/exp/decode_abl_api_contract.py | clang++ -std=c++20 -fsyntax-only -x c++ - -DCUDART_VERSION=13000
```

These compile the candidate's actual selector/calls against the signatures from
the judge report, exposing only that generation's names. They do not constitute
verification against NVIDIA headers: no CUDA headers are present locally.

No local torch, CUDA GPU, or nvcc: native compilation and GPU timing remain
unverified. No check submitted and nothing pushed/committed to main.

## Check 032e0f0d828a4344 correction

The caller's judge run reached base prepare in 137 s but the helper failed:
CUDA 13 removed the suffixed graph API identifiers. Thus zero component timings
were obtained, not a negative performance result. This revision fixes API naming
without changing the seven-argument capture-info/five-argument edge-data
semantics. No runtime measurements yet for this revision. Forbidden-process
pattern scan of the candidate returns zero matches.

## Ledger handoff (caller must record before checking)

The legacy `.tools/tried.py` path is missing. Read/search used `scripts/tried.py`
and inspected N3 ceiling/correction. Its add command pushes main, forbidden in
this subtask. Suggested claim: **Decode full-layer component no-op graph ablation,
exact PDL edges plus real-intermediate prefill and per-replay route equality**;
number `loop/decode-abl`; source `kb/exp/decode_layer_abl.py`; status `in progress`.
Material difference from N3: all components, true no-op nodes instead of dropped
launches, and real intermediates/strict route equality preventing workload drift.
