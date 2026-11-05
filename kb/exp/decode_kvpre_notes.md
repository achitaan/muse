# Decode tile-zero KV pre-wait candidate

- Branch: `loop/decode-kvpre`, based on `origin/argon/keell-base-20261102`
  (`f50a51d`; no local branch with the requested base name existed).
- Check source: `kb/exp/decode_kvpre.py` (standalone).
- Incumbent attention C++/CUDA strings are byte-identical to `kb/v1/decode_v1.py`.
  `_kp_source` derives the experimental translation unit only; build flag
  `-DOKC_KVPRE=1` and a separate hashed extension name select it.
- All 13 `load_inline` sites use C++20 for host and CUDA and explicit
  `-gencode=arch=compute_90a,code=sm_90a`; incumbent math flags are retained.
- Removed the body of the disabled, non-selecting MFAB write-flush diagnostic
  to reserve prepare time and keep the standalone file below 1,000,000 bytes.

## Kernel invariants

Both L=1 and L=8 B=1 plans use non-WS kernels. The changed source locations in
the original are the prologue at lines 1519-1532 and its first-tile gather at
1590-1627. Flat/WS/two-stream paths are excluded. The existing ONE-copy body is
moved verbatim into a helper, not duplicated. After the page-index group drains,
SPEC length is validated before any cached address is dereferenced. A mismatch
returns with no outstanding copies/global writes and retries the real length.

Tile zero's old rows (positions below seq and end) are issued before PDL wait;
new rows are skipped, and past-end rows are zero-filled without global reads.
Early copies remain uncommitted across the wait and join the existing Q/constants
group. Tile-zero's original later commit is retained, including its future-index
copies; only its duplicate gather is omitted. Thus no extra commit/group, ring
slot, or launch is added, and all wait_group counts and new-row patching stay
unchanged. The existing Q-group wait also drains early KV0.

## Gate

Per d1S/d1F/v1S/v1F: two fresh seeded input draws/passes; 12 alternating ABBA/BAAB
rounds/pass; independent captured incumbent/null/candidate whole-layer graphs;
512 MiB float32 READ reduction before every replay, outside the event interval.
Each pass must have median AND paired candidate/base <=0.992, and both null
median and paired ratios in [0.995,1.005]. No partial trial can be admitted.

Each pass compares QKV, attention, hidden, full stream, residual-carry-removed
update, new FP8 cache rows and ordered expert IDs. Numeric error budgets come
only from 28 incumbent repeats on that input; no fixed tolerance or multiplier.
Candidate IDs must equal an observed incumbent ID tensor. Every untouched cache
byte is checked; stable new rows must match exactly. Correctness replays are
outside timing. Per-pass isolated attention diagnostics use PDL disabled and
the same ABBA/BAAB read-flush timing, but do not determine admission.

Separate compilation overlaps base prepare. Build wait ends at 225 seconds,
and gate rounds stop at 275 seconds from prepare entry (cleanup reserve).
The previous MFAB diagnostic is disabled. These are soft budgets, not proof of
end-to-end prepare time; no GPU is present locally.

## Local validation

`python3 kb/exp/test_decode_kvpre_cpu.py -v`: nine tests pass (gate boundaries,
missing passes/orders, null failures, NaN/Inf, repeatability/no-tolerance-floor,
all build flags, byte-identical incumbent, source ordering/group counts, async
prologue model, exhaustive cached/new/padded row partition).

Python AST and diff whitespace checks pass. CUDA compilation, CUDA correctness,
performance and <300-second prepare remain unverified: no torch, GPU or nvcc.
No check, push, or main-branch commit was performed. The caller must run the GPU
check and record the ledger claim/result before spending it. TRIED search used
`scripts/tried.py search KV wait`; the configured `.tools/tried.py` is absent.
This is distinct from closed side-stream packing/deeper-ring/index-prefetch ideas.
