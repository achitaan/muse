# Top-50 loop progress (Engineer + Optimizer, loop mode)

Start: Oct 31 02:35 EDT. Limits: 8 passes, 90 min (until 04:05), $200 spend (baseline $60.39 at start).
Baseline: published Decode 2.0755x (`452fbd9d`, board_best/decoder_pdli_scored_2.0755.py).
Rule: keep a pass only if all tests pass AND faster (paired in-prepare A/B >= 1% both orders, and check score >= baseline).

| Pass | Item | Result | Kept/reverted | Lane | Spend so far |
|---|---|---|---|---|---|
| 1 | PDLI selector: 1% gate + MoE-variance-tolerant whole-layer/moe_unpool probe, rel_l2<=1e-3 and <=16 elems (`decoder_p1_gate.py`) | valid, 2.0434x (CI 2.013-2.074); tolerance too strict: MoE run-to-run diffs (18-731 elems, rel 1.3e-3) rejected every whole layer, selected={} | reverted | fixer | $61.6 (+$1.2) |
| 2 | Same, tolerance rel_l2<=2e-3, no element cap (`decoder_p1b_tol.py`) | valid, **2.0975x** (CI 2.066-2.129); selects M1 sliding+full mask 15; whole layer -4.0/-4.7 us both orders | kept | direct (2-line constant edit) | $62.0 |
| 3 | + PDL on `_pn`/`_sfn` Triton twins and MoE launch, mask bit 16 (`decoder_p2_pdl.py`) | valid, 2.0813x (CI 2.050-2.113, within noise of p1b); in-prepare whole layer M1 sliding -4.9/-5.1, M1 full -6.4/-6.9, M8 sliding -1.5/-2.8 us | kept (best lineage) | fixer | $62.0 |
| 4 | + early stock-norm PDL trigger (port of pdls1) at M1, bit 32 (`decoder_p3_norm.py`) | valid, 2.0503x (noise); unit -1.9/-2.3 us but whole layer only -0.5/-0.1 us, not selected | reverted | fixer | $63.1 |
| 5 | + early norm at M8 (`decoder_p4_norm8.py`) | valid, 2.0891x (CI 2.058-2.120); selected M1 sliding (layer -1.4/-1.4 us) and M8 full (-2.1/-2.1 us) | kept | fixer | $63.1 |
| 6 | p2 + glue PDL (bit 16) at 16/32-token decode, batch-m probe data (`decoder_p5b_glue16.py`, rebased on p2 to save prepare time) | valid, 2.0900x (CI 2.059-2.121); M16 sliding selected (-2.1/-1.8 us); M16 full / M32 miss the 1% gate | kept | fixer (rebase direct) | $63.4 |
| 7 | Integration: p4 + glue16 (`decoder_p5_glue16.py`) | valid, **2.0960x** (CI 2.065-2.127); this run selected M1 sliding 30, M1 full 31, M8 sliding/full early norm 52 (per-shape gates near 1% flip run to run) | kept | fixer (already built) | $63.4 |
| 8 | Reviewer fixes on p5: bounded PDLS build join (45 s) + plan try/except; tolerant probe rejects nonfinite outputs (`decoder_p6_final.py`) | valid, check 2.0682x (`689fd93e`, prepare 273.7 s); **scored 2.0783x** (`0aa30c92`, CI 2.047-2.110) = new board best (was 2.0755), rank 280/1091 | kept, published to `main` `board_best/decoder_p6_scored_2.0783.py` (`0fb50a0`) | reviewer + fixer | $64.5 (+$4.1) |

## Self-review (03:30 EDT, 55 of 90 min, 8 of 8 passes)

- **Baseline vs final:** published 2.0755x -> scored 2.0783x (+0.13%). Overall rank ~280 / 1091; top 50 needs 2.246x (+8.1%). **Goal not reached.**
- **What the gains really are:** paired in-prepare A/B per selected shape (both orders): M1 sliding -4.0 to -5.1 us, M1 full -4.6 to -6.9 us, M8 -1.4 to -2.8 us, M16 sliding -1.8 to -2.1 us, early norm -1.4 to -2.1 us on top. That is roughly +1-1.5% of Decode in expectation, below check noise (6 checks of this lineage: 2.068-2.098; no-selection control 2.043).
- **Kept:** 1% gate; MoE-variance-tolerant whole-layer probe (rel L2 <= 2e-3; the MoE kernel is nondeterministic, so strict bytes rejected genuine wins); PDL twins of `_pn`/`_sfn` and the MoE launch (1/8/16/32 tokens); early-trigger private norm at 1/8 tokens; reviewer robustness fixes.
- **Failed:** pass 1 (tolerance too tight: lost all selections, 2.043x); pass 4 (early norm at M1 alone: unit -2 us, layer gain < 1%).
- **Per-shape gates near 1% flip from run to run** (e.g. M8 sliding selected in 3 of 6 runs). The scored run's selection is whatever its prepare measured.
- **Escalation:** not triggered; no two consecutive passes failed to give a valid gain. Researcher (planning, before the loop) supplied items #1 and #3; items #2 (top-k fold) and #4 (MoE flags) did not fit the time box.
- **Reviewer (once, at end):** 2 major, 1 minor. Fixed: unbounded PDLS join (major), nonfinite acceptance in the tolerant probe (minor). Not fixed: the MoE PDL scope is a process-global flag (major in principle); safe in practice because no other thread runs MoE during capture or replay (the background threads only compile or warm the norm). A per-call launch argument needs a C++ ABI change to the packed MoE source.
- **Spend:** ~$4.8 on top of the $60.39 baseline (list-price estimate).
- **Next best idea:** the remaining gap is ~13 us per layer everywhere, which PDL alone can't supply. Next steps, in order:
  1. MoE at 16/32 tokens (65/91 us vs ~45-62 floor; first measure real touched-expert counts);
  2. a persistent FFN island at 1/8 tokens (R2f -> norm -> MoE without boundaries);
  3. top-k folded into `sk2` for 16/32 tokens (needs SGLang's routing source from the judge environment).
