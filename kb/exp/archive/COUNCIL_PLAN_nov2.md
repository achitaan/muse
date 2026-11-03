# Council plan, Nov 2 2026 (achitaan)

Source: oracle (GPT-6 Astra), oracle-lite (Gemini Argon), researcher (GPT-6.1 Sol, FlashInfer SM90 MegaMoE TUNING.md et al.),
then council seats Opus / Argon / Sol and the council synthesis. Kernel close: Nov 4 11:59 PM PT = Nov 5 07:59 UTC.

## Ranked builds
1. Decode v3 (repaired: router hi+lo, bundle composition, per-arm screen, equivalence classes) incl. qkv_gate@32 / pooler@32 CURP.
   Go: paired >= +0.4% in 2 checks; score at >= +0.5%. Stop: 3 repair checks without consistent >= +0.3%.
2. MoE dependency-aware B order + doneAs wait counters (check today). Stop at once if waits < ~3% of phase B.
   Go: MoE win -> standalone Decode transfer test, adopt if >= +0.2% paired.
3. Exact T=8 WSI (t2_wsi) port in Decode only. One check; stop if no gain.
4. L2 locality screen (co-schedule a split expert's jobs in route(); keep xA/xB in L2; evict_first only on single-use weights),
   only if counters say phase B is memory-stalled; check tried.py vs HINT first. One screen; stop if bs256 < +1%.
5. Dynamic finalizer claiming, only if finalizer skew >= 3 us. One experiment.

Excluded: Dense CURP expansion (closed 83c46449), Stream hoist (dup decode_chain), OKC_YBF flag-only (needs float2 readers),
gle T=8 toggle (needs 16-row pooler/router), WS_JOIN split / pre-gather weight start (EARLYW overlap), o-weight prefetch,
smaller CTAs, store-server warp, evict_last hints alone, _pn_inputs fusion (unverified in keell's decode), Attention, Prefill.

## Schedule (UTC)
- Nov 3 00:02 Decode v3 check; MoE B-order follow-up only if waits >= 3%; Decode checks ~04/08/10/12; scored ~14:00 if >= +0.5% twice.
- Nov 4: compose winners 00-06; last new arm by 12:00 (freeze on ideas); scored 14:00; lock final composition 20:00.
- Nov 5 (fresh quota): verify 00:05-02:00, final scored ~02:30 (fallback 2.2341 build if lead not robust), buffer to 07:59.

## Expected Decode outcome
Floor 2.2341; likely +0.4..+1.2% (2.243-2.261, central ~2.250); upside 2.268-2.274; +3.3% (#50) ~5%; beating keell 2.2588 ~25-30%.
