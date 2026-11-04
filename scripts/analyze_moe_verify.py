"""Stdlib-only public-routing model; not the judge's unobserved RNG draw."""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
FLOPS_PER_ROUTE = 6 * 4096 * 4096  # gate + up + down, two FLOPs per FMA
BYTES_PER_JOB = 48 * 16 * 36864  # 32 GEMM1 units and 16 GEMM2 units, 16 stages


def analyze():
    rng = random.Random(20261104)
    result = {
        "source": "incumbent _moe_routes, model distribution only; Python RNG != CUDA RNG",
        "seed": 20261104,
        "dense_peak_tflops": 989,
        "flops_per_local_route": FLOPS_PER_ROUTE,
        "explanation": "6HI, not 4HI: GEMM1 has 2I output rows; EP8 ignores nonlocal routes",
        "actual_check_route_counts": "not provided; cannot compute exact achieved TFLOPS from latency alone",
        "cases": [],
    }
    for tokens, us in ((128, 183.3), (256, 216.2)):
        draws = []
        for draw in range(32):
            popularity = [.03 * rng.gauss(0, 1) / math.sqrt(-math.log(max(rng.random(), 1e-12)))
                          for _ in range(256)]
            counts = [0] * 256
            for _ in range(tokens):
                scores = [rng.random() + p for p in popularity]
                for e in heapq.nlargest(8, range(256), key=scores.__getitem__):
                    counts[e] += 1
            rank = max(range(8), key=lambda r: sum(counts[32*r:32*r+32]))
            local = counts[32*rank:32*rank+32]
            jobs = [min(64, c - start) for c in local for start in range(0, c, 64)]
            routes, padded = sum(local), sum(8 * math.ceil(c / 8) for c in jobs)
            draws.append({
                "draw": draw, "local_expert_histogram": local, "local_routes": routes,
                "active_experts": sum(c > 0 for c in local), "jobs": len(jobs),
                "padded_routes": padded, "sync8_jobs": sum(c <= 8 for c in jobs),
                "useful_tflops": routes * FLOPS_PER_ROUTE / (us * 1e6),
                "issued_tflops": padded * FLOPS_PER_ROUTE / (us * 1e6),
                "weight_tb_s": len(jobs) * BYTES_PER_JOB / (us * 1e6),
            })
        summary = {key: statistics.median(d[key] for d in draws)
                   for key in ("local_routes", "active_experts", "jobs", "padded_routes",
                               "sync8_jobs", "useful_tflops", "issued_tflops", "weight_tb_s")}
        summary["useful_pct_dense_peak"] = summary["useful_tflops"] / 989 * 100
        summary["issued_pct_dense_peak"] = summary["issued_tflops"] / 989 * 100
        summary["useful_tflops_range"] = [min(d["useful_tflops"] for d in draws), max(d["useful_tflops"] for d in draws)]
        summary["no_skew_uniform_ep8_tflops"] = tokens * FLOPS_PER_ROUTE / (us * 1e6)
        result["cases"].append({"tokens": tokens, "measured_us": us, "model_medians": summary, "draws": draws})
    return result


if __name__ == "__main__":
    result = analyze()
    out = ROOT / "results/moe_verify_sync"
    out.mkdir(exist_ok=True)
    (out / "roofline.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(case["tokens"], json.dumps(case["model_medians"], sort_keys=True))
