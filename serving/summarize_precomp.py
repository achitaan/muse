"""Produce a source-bound summary without replacing any raw measurements."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/precomp-20261105/results"


def read(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines()
            if line.startswith("{")]


def performance(name, candidate):
    rows = [r for r in read(name) if r["event"] == "timing"]
    groups = {}
    for row in rows:
        key = (row["tokens"], row["pattern"], row["cache"])
        groups.setdefault(key, []).append(row)
    result = []
    for (tokens, pattern, cache), group in sorted(groups.items()):
        reductions, savings, controls = [], [], []
        for row in group:
            m = row["medians_us"]
            reductions.append(1 - m[candidate] / m["stock_a"])
            savings.append(m["stock_a"] - m[candidate])
            controls.append(abs(1 - m["stock_b"] / m["stock_a"]))
        result.append(dict(tokens=tokens, pattern=pattern, cache=cache,
                           blocks=len(group), latency_reduction_range=[min(reductions), max(reductions)],
                           saved_us_range=[min(savings), max(savings)],
                           max_abs_null_control=max(controls),
                           performance_threshold_only=(len(group) == 3 and min(reductions) >= .10
                                                       and min(reductions) > max(controls))))
    return result


def validate_patch(path, rules):
    data = path.read_bytes()
    assert len(data) <= rules["max_patch_bytes"]
    text = data.decode()
    assert "\r" not in text
    names = []
    for line in text.splitlines():
        if line.startswith("+++ b/"):
            name = line[6:]
            names.append(name)
            assert name.startswith(rules["allowed_prefix"])
            assert not any(name.startswith(prefix) for prefix in rules["forbidden_prefixes"])
            assert Path(name).suffix in rules["allowed_suffixes"]
        elif line.startswith("+"):
            assert len(line[1:]) <= rules["max_line_chars"], line[:120]
        assert not line.startswith(("old mode ", "new mode ", "rename from ", "rename to ", "GIT binary patch"))
    assert not any(word in text for word in rules["forbidden_words"])
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), files=names)


def main():
    manifest = json.loads((HERE / "results/precomp-20261105/official-serving-manifest.json").read_text())
    summary = dict(
        scope="synthetic single-H200 expert operations; no full-server answer/throughput evidence",
        s1_status="retained first candidate, pending official serving equivalence",
        moe_status="experimental only; no accepted or enabled serving regime",
        s1=performance("s1-paired-v2.jsonl", "s1"),
        moe=performance("native-release-paired.jsonl", "native"),
        patches={p.name: validate_patch(p, manifest["rules"]) for p in
                 (HERE / "s1_postnorm_v2.patch.diff", HERE / "moe_direct_marlin.experimental.patch.diff")},
    )
    for label, filename in (("s1_coverage", "s1-coverage.jsonl"),
                            ("moe_coverage", "native-final-coverage.jsonl")):
        rows = read(filename)
        summary[label] = {event: sum(r["event"] == event for r in rows)
                          for event in ("case_start", "case_done", "empty_pass", "fresh_replay")}
    precision = []
    for filename in ("native-release-paired.jsonl", "native-v3-epsilon.jsonl"):
        rows = read(filename)
        differences = [r["candidate"]["rel_l2"] for r in rows if "candidate" in r]
        precision.append(dict(file=filename, max_recorded_relative_l2=max(differences),
                              acceptance="unresolved; no numerical tolerance substituted for serving acceptance"))
    summary["precision"] = precision
    (HERE / "PRECOMP_RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    source_files = sorted(p for p in HERE.iterdir() if p.suffix in (".py", ".cu", ".cuh", ".inc", ".diff"))
    binding = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    binding["_base_commit"] = "19e09dd694d748d6ef77bd0e09308f56c3b210ca"
    binding["_retained_board_source"] = "f72cdf2b5438c70369b0fb823e171f4022b22b97f0dff66c6170dfe3aaf845fb"
    binding["_coverage_cuda_sha256"] = "694e2b760c58b513f0a477295c94d2ca115771f0e5239e536adf812ccd2b99f1"
    binding["_release_difference"] = "Removed spaces on one blank CUDA source line; final paired measurements rebuilt exact release source."
    (HERE / "PRECOMP_SOURCE_HASHES.json").write_text(json.dumps(binding, indent=2) + "\n", newline="\n")
    print(json.dumps({"patches": summary["patches"], "coverage": [summary["s1_coverage"], summary["moe_coverage"]]}, indent=2))


if __name__ == "__main__":
    main()
