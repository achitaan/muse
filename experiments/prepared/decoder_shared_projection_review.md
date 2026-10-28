# Shared small-projection extension trial

Phase7 passed the full decoder check, but its new projection paths never ran:
old preparation took 257 seconds, above the 230-second gate. The retained
09badc9b source checks at 1.994319167x; the newer 1.997916935x observation used
the same timed compute. Neither observation establishes new projection speed.

The revised source ced7a4b9fa1aef6a51549706acf5eccaa44d012b27f83f4ca25e124d5f42b610
is 995148 bytes. It adds the same three fixed public configurations to the
existing fused pooler/router extension. The generic CUDA body and every old
kernel literal remain byte-identical. The factory exports sk2small alongside
the old entrypoints and has a source-specific module name. No second extension
build occurs. Original Apache and MIT notices remain in place.

Three unused setup paths receive empty weight maps: prepare_router,
prepare_router2 and rt_prepare_routertail_candidates. Their actual functions
then return disabled before any GPU work. The whole decoder calls its dense
helper only for QKV, output, pooler and unpooler. Its separate FP32 small router,
fused pooler/router and wide-router path are unchanged; the wide-router bank
in bundle.rw is retained. Dense fallback routes for all four families remain.

The structural audit reverses only those three setup edits, two appended
extension constants and factory name/export edits to recover the entire old
module AST. Its 1208 dispatch-only probes force dense fallthrough and reject
any access to retired router components. They do not establish numerical
correctness. The independent codec audit reconstructs 1,441,792 BF16 values and
233 correction entries exactly, checks stream bijections, store geometry,
fallbacks, global bindings and the shared factory's one-build reuse.

The bounded experiment retains the same three configurations, two independent
synthetic activation patterns, token counts 1 and 8, seven paired cold-L2 graph
replays, 3% combined win threshold and at most 2% slowdown on either pattern.
No timing is cached for scored inputs. State contains static packed weights
and configuration choices only. All BF16/FP32 precision boundaries remain.

Because compilation is now complete before comparison, selection may begin
until 260 seconds, packing must finish by 265 and comparisons stop at 280.
This change accompanies removal of three builds and the extra extension,
rather than relying only on relaxed time gates. Log the base preparation time.
The complete prepare must still fit the official 300-second allowance.

One full official check is planned, reusing the cached baseline. Keep only
with full 13/12/integrity and explicit nonempty SMALL_PROJ_CHOICES. New GPU
kernel correctness, setup duration and runtime speed remain unverified until
that result. No tile sweep or scored intermediate is planned.
