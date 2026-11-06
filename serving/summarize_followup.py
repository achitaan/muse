"""Summarize the bounded follow-up without changing historical evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from summarize_precomp import validate_patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/followup-20261105"


def read(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines()
            if line.startswith("{")]


def main():
    s1 = read("s1-isolation.jsonl")
    assert s1[-1] == dict(event="complete", cases=168)
    cases = [r for r in s1 if r["event"] == "case"]
    nonempty = [r for r in cases if r["tokens"]]
    differing = [r for r in cases if not r["s1"]["bit_equal"]]
    moe = read("ordered-t128.jsonl")
    stages = [r for r in moe if r["event"] in ("stage", "combined")]
    confirmed = [r for r in stages if r["null"]["bit_equal"] and not r["candidate"]["bit_equal"]]
    rules = json.loads((RESULTS / "official-serving-manifest.json").read_text())["rules"]
    paths = [HERE / "s1_postnorm_v2.patch.diff",
             HERE / "experiments/ordered_fixup/moe_ordered.experimental.patch.diff"]
    summary = dict(
        s1=dict(cases=len(cases), stock_repeat_differences=sum(not r["stock_repeat"]["bit_equal"] for r in cases),
                output_difference_cases=len(differing),
                first_differing_boundary=dict(Counter(r["diagnostics"]["True"]["first_different"] for r in differing)),
                trace_matches_actual_helper=all(r["diagnostics"]["True"]["trace_vs_production"]["bit_equal"] for r in nonempty),
                max_relative_l2=max(r["s1"]["rel_l2"] for r in cases),
                no_fusion_still_differs_small_cases=sum(not r["diagnostics"]["False"]["stages"]["output_bf16"]["bit_equal"] for r in nonempty if r["tokens"] < 32),
                source_changed=False, numerical_correction_cycles=0,
                status="unchanged opening-check candidate; byte equivalence unresolved; official answer gate required"),
        moe=dict(prepared=next((r for r in moe if r["event"] == "prepared"), None),
                 tile_passes=sum(r["event"] == "tile_pass" for r in moe),
                 stable_stock_differing_candidate_rows=confirmed,
                 compared_rows=stages,
                 status="rejected for serving equivalence" if confirmed else "unresolved; no acceptance inferred",
                 enabled_shapes=[], performance_measured=False,
                 reason="T128 correctness gate precedes T256 and timing; no tolerance waiver"),
        patches={p.name: validate_patch(p, rules) for p in paths},
        official_checks_used=0, official_submissions_used=0,
    )
    (HERE / "FOLLOWUP_RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    source_paths = [HERE / name for name in ("check_s1_isolation.py", "audit_ordered_fixup.py", "build_ordered_fixup.py", "run_ordered_probe.py", "build_native_patch.py", "followup_prepare.sh", "summarize_followup.py")]
    source_paths += sorted((HERE / "experiments/ordered_fixup").glob("*.*"))
    binding = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths + paths}
    (HERE / "FOLLOWUP_SOURCE_HASHES.json").write_text(json.dumps(binding, indent=2) + "\n", newline="\n")
    print(json.dumps({"s1": summary["s1"], "moe_status": summary["moe"]["status"], "patches": summary["patches"]}, indent=2))


if __name__ == "__main__":
    main()
