"""Reproducible CPU arithmetic models; these do not execute GPU kernels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def errors(output, reference, probes):
    difference = (output.double() - reference.double()).norm(dim=-1)
    norms = reference.double().norm(dim=-1)
    global_error = float(difference.norm() / norms.norm().clamp_min(1e-30))
    row_error = float((difference / norms.clamp_min(float(norms.mean()) * 0.1)).max())
    probe_error = float((difference[probes] / norms[probes].clamp_min(1e-30)).max())
    return {"global": global_error, "row": row_error, "probe": probe_error}


def input_tensor(m, k, generator, outliers):
    x = torch.randn((m, k), generator=generator)
    if outliers:
        x[:, ::97] *= 30
    # Exact powers of two expose premature weight/output rounding.
    x[0] = 0
    x[0, k // 3] = 32
    return x.to(torch.bfloat16), [0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cpu-arithmetic.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = []
    for seed in (20261023, 44491, 80413):
        gen = torch.Generator().manual_seed(seed)
        k, n = 8192, 256
        w = torch.randn((n, k), generator=gen) / math.sqrt(k)
        hi = w.to(torch.bfloat16).float()
        lo = (w - hi).to(torch.bfloat16).float()
        for m in (1, 8, 9, 32, 128, 256):
            x, probes = input_tensor(m, k, gen, outliers=m in (9, 256))
            xf = x.float()
            # Independent FP64 GEMM implements the mathematical task directly.
            reference = (x.double() @ w.double().T).float()
            splits = 16 if m <= 32 else 4
            partials = []
            for split in range(splits):
                start, end = split * (k // splits), (split + 1) * (k // splits)
                ah = torch.zeros((m, n))
                al = torch.zeros((m, n))
                for kk in range(start, end, 64):
                    ah += xf[:, kk:kk+64] @ hi[:, kk:kk+64].T
                    al += xf[:, kk:kk+64] @ lo[:, kk:kk+64].T
                partials.append(ah + al)
            output = torch.stack(partials).sum(0)
            error = errors(output, reference, probes)
            # Bounds from official muse_bf16_proj/TASK.md and checks.py.
            passed = error["global"] <= 1e-4 and error["row"] <= 5e-4 and error["probe"] <= 1e-3
            assert passed, (seed, m, error)
            results.append({"seed": seed, "m": m, "splits": splits, "errors": error, "pass": passed})
        # Negative control validates that this test detects a disallowed shortcut.
        bad = xf @ hi.T
        bad_error = errors(bad, reference, probes)
        assert bad_error["global"] > 1e-4, bad_error
        results.append({"seed": seed, "negative_control": "BF16 router weights alone", "errors": bad_error, "correctly_rejected": True})
    report = {"utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "device": "cpu", "limitation": "Arithmetic model, not GPU execution or timing; BLAS and MMA reduction details differ.",
              "candidate_sha256": hashlib.sha256((ROOT / "candidates/router_only.py").read_bytes()).hexdigest(),
              "results": results}
    (ROOT / args.output).write_text(json.dumps(report, indent=2))
    print(f"PASS: {len(results)} deterministic cases including negative controls; CPU model only")


if __name__ == "__main__":
    main()
