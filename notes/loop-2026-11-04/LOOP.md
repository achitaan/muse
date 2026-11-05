# LOOP (Muse 2, run inside the opencode chat)

Started: Nov 4 2026. Owner: the main chat session (no external ralph/codex controller).

## Loop
pick idea -> `TRIED.md` search + claim -> implement (on a branch) -> local gate -> okc check -> record result -> repeat.

## Scope
User permission: a full redesign/rewrite of any kernel is allowed. Do not limit work to gated tweaks on keell0renz's files. Validity rules below still apply.
Scored problems only: muse_bf16_proj, muse_attention_decode, muse_moe_w4a16, muse_decode_layer. NOT prefill (user: not scored, not part of the board). Do not touch prefill again.
Start with the highest expected gain per check, not round-robin.

## Worktrees (one lane per problem, no shared write scope)
Base: `~/.local/share/opencode/worktree/e72615/`
- muse_bf16_proj -> `loop-bf16-proj`
- muse_attention_decode -> `loop-attention-decode`
- muse_moe_w4a16 -> `loop-moe-w4a16`
- muse_decode_layer -> `loop-decode-layer`
Each lane edits only its own worktree and branch. `TRIED.md` is the only shared file: claim and record on `main` (fast-forward, retry), never from a lane checkout.

## Budget (user cap: $400 TOTAL for this session/channel across all lanes, not per lane; raised from $250 on Nov 4 ~17:30 PT)
- Live spend is not readable from the session. Estimate it: log tokens in/out per delegated call and per turn in `LOOP_LOG.md` using the proxy price list (`tools.opencode.models` cost fields).
- Stop at an estimated $340 (soft stop, leaves margin for the $400 cap), or earlier on any rule below.
- Expensive agents (oracle, hard-fixer, reviewer, council): at most 3 oracle calls total; ask before more.

## Evaluator limits (reset 00:00 UTC; refresh with `.tools/okc problems --json`)
- Per problem per day: 40 checks, 20 scored. Today all unused.
- User confirmed: both checks and scored submissions may be used. Prefer checks to screen ideas; submit scored only for a candidate that already passes checks and looks like a board improvement, so the 20/day scored quota is not wasted.
- Never spend a check on an idea that `TRIED.md` marks closed unless the new version is materially different.

## Stop rules
1. Estimated spend reaches $340.
2. 3 consecutive experiments with no gain over the board_best score on the targeted problem.
3. Substantial progress: a verified scored result beating the current board by >= 3% on any problem.
4. A correctness/validity doubt about a result (stop and report, do not submit).
5. The user interrupts.

## Validity
No output caching, no harness edits, no eval-detail exploitation. Keep full computation and tolerances. Record every result in `TRIED.md` with key number, job id, source file.

## Ledger helper
`.tools/tried.py` is missing on this machine. Edit `TRIED.md` directly with the same rules and commit only `TRIED.md` to `main`.

## Other sessions
Another session is already working on Muse 2. Do not touch its branches, worktrees, in-progress `TRIED.md` rows, queued jobs, or results files. Do not poll or cancel its jobs. Claim only ideas that are not already open/in progress, and keep this session's quota use modest so the other session keeps its checks (refresh `okc problems --json` before each submission).
