# achitaan / muse - Muse Open Kernel Challenge

Branch `argon/decode-20261103`: pushing the Decode board (and its MoE / Dense components) toward 2.37.

**Full handoff: [HANDOFF.md](HANDOFF.md).** Then: **[kb/APPROACH.md](kb/APPROACH.md) (how we choose work) and [kb/STATE.md](kb/STATE.md)** - problem, scoring, where we stand, per-kernel time, gap to the target, active lanes,
closed ideas, how changes ship, and the file map.

Then:
- [kb/exp/JOEY_PLAN.md](kb/exp/JOEY_PLAN.md) - received hints, lanes, loop protocol, iteration log.
- [kb/exp/VMS_BRIEF.md](kb/exp/VMS_BRIEF.md) - small-batch MoE engine design brief.
- [kb/exp/IDEAS_predictor_prompt.md](kb/exp/IDEAS_predictor_prompt.md) - predictor / chase ideas.
- `python kb/muse_score.py --backtest` / `--job 1ca0668a --save-us d16S=10` - exact score replay and what-ifs.
- Ledger of every experiment: `TRIED.md` on `main`; board bests: `board_best/` on `main`.

Kernel code builds on keell0renz's Apache-2.0 files (see kb/README.md); our edits are marked `achitaan`.
Historical docs from the Oct 23 Dense sprint: docs/archive/oct23-sprint/.
