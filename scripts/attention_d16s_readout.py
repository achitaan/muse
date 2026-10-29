import json
import math
import sys
from pathlib import Path

RUNS = [("A1", "results/attention-split-sweep-20261029"), ("A2", "results/attention-split-a2-20261029"),
        ("A3", "results/attention-split-a3-20261029"), ("A4", "results/attention-split-a4-20261029"),
        ("A5", "results/attention-autotune-a5-20261029"), ("A6", "results/attention-control-a6-20261029")]
TARGET = ("decode 16", "sliding")


def shapes(path):
    j = json.load(open(Path(path) / "result.json", encoding="utf-8"))
    r = j.get("result", j)
    if "records" not in r:
        r = r.get("job", r)
    out = {}
    for x in r["records"]:
        if x.get("event") == "benchmark" and isinstance(x.get("case"), dict):
            c = x["case"]
            layer = c["layer"]
            out[(c["load"], layer)] = x["median_ns"] / 1000.0
    return out, r["speedup"]


data = {}
for (name, path) in RUNS:
    f = Path(path) / "result.json"
    if f.exists() and "records" in f.read_text(encoding="utf-8"):
        data[name] = shapes(path)
if "A6" not in data and len(sys.argv) < 2:
    print("A6 not finished; showing available runs")
ref = data.get("A6") or data["A3"]
controls = [k for k in ref[0] if k != TARGET and all(k in d[0] for d in data.values())]
print("controls:", len(controls))
for (name, (s, sp)) in data.items():
    gm = math.exp(sum(math.log(s[k] / ref[0][k]) for k in controls) / len(controls))
    print(f"{name} track {sp:.4f} decode16 sliding {s[TARGET]:.2f} us  controls vs ref GM {gm:.4f}  normalized decode16 sliding {s[TARGET] / gm:.2f} us")
