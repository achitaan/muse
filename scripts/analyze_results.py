"""Summarize official results; per-shape ratios are descriptive, not new scores."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    seen = {}
    for path in sorted((ROOT / "results").glob("*-result.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        if doc.get("result"):
            seen[doc["id"]] = doc
    lines = ["# Official H200 results", "", "All speedups are relative to SGLang.", ""]
    summary = {}
    for job, doc in seen.items():
        result = doc["result"]
        records = result.get("records", [])
        benches = [x for x in records if x.get("event") == "benchmark"]
        tests = [x for x in records if x.get("event") == "test"]
        row = {k: result.get(k) for k in ("passed", "speedup", "speedup_ci", "status", "error")}
        row["tests_passed"] = sum(x.get("status") == "pass" for x in tests)
        row["benchmarks_passed"] = sum(x.get("status") == "pass" for x in benches)
        row["shapes"] = [{"m": b["case"]["m"], "layer": b["case"]["layer"],
                          "load": b["case"].get("load"), "speedup": b.get("speedup"),
                          "candidate_us": b.get("median_ns", 0) / 1000,
                          "baseline_us": b.get("baseline_ns", 0) / 1000,
                          "status": b.get("status")} for b in benches]
        summary[job] = row
        lines.extend([f"## {job}", "", f"Status: {row['status']}; correctness: {row['passed']}; speedup: {row['speedup']}; 95% interval: {row['speedup_ci']}.",
                      f"Tests passed: {row['tests_passed']}/9; benchmark shapes passed: {row['benchmarks_passed']}/30.", "",
                      "| M | Layer | Candidate µs | SGLang µs | Speedup |", "|---:|---|---:|---:|---:|"])
        for shape in row["shapes"]:
            lines.append(f"| {shape['m']} | {shape['layer']} | {shape['candidate_us']:.3f} | {shape['baseline_us']:.3f} | {shape['speedup']:.3f} |")
        lines.append("")
        print(job, json.dumps({k: v for k, v in row.items() if k != "shapes"}))
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "results" / "official-results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
