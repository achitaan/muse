"""Offline byte/roofline audit, NOT a submission or a measured speedup.

Input scalars: supplied OKC job 8bf9cae70baa4192 JSON. The reference's floor
uses 0.75 * nominal mean touched experts, NOT this job's actual route histograms.
No code here imports/executes the reference or accesses any pending job.
"""
from __future__ import annotations

import json
import math


H = 4096
HBM = 4.8e12
BF16_FLOPS = 989e12
SHARE = 0.75
BEST_BOARD = 2.3797
CHECK_BOARD = 2.347287161109339
SOURCE_JOB = "8bf9cae70baa4192"
VALUES = 3 * H * H
CODE_BYTES = VALUES // 2
SCALE_BYTES = VALUES // 16
EXPERT_BYTES = CODE_BYTES + SCALE_BYTES
# bs, measured paired time (us), published nominal discounted SOL (ns).
CASES = (
    (1, 25.990149522972588, 11416.507733333334),
    (16, 93.40947128144966, 52519.458133333326),
    (32, 126.9083636284592, 70755.39626666668),
    (8, 65.26500418951957, 37318.92906666667),
    (128, 183.2749141406434, 101341.04746666664),
    (256, 216.24507722497577, 108944.31573333332),
)


def actual_weight_bytes(ids, *, local_start=0, local_count=32):
    """Exact once-read bytes for provided routes, not a routing simulation.

    The supplied profile does not contain such arrays. This helper exists to
    count actual routes if they are supplied later; synthetic tests do not
    establish this job's actual distinct-expert counts.
    """
    counts = [0] * local_count
    for row in ids:
        if len(row) != 8:
            raise ValueError("expected eight routes per token")
        for e in row:
            if not isinstance(e, int) or not 0 <= e < 256:
                raise ValueError("expert ids must be integers in [0,256)")
            if local_start <= e < local_start + local_count:
                counts[e - local_start] += 1
    touched = sum(c > 0 for c in counts)
    return {
        "distinct": touched,
        "routes": sum(counts),
        "counts": counts,
        "code_bytes": touched * CODE_BYTES,
        "scale_bytes": touched * SCALE_BYTES,
        "weight_bytes": touched * EXPERT_BYTES,
    }


def nominal_case(bs, time_us, sol_ns):
    # Invert the PUBLIC bound's byte formula, not a measured routing draw.
    # Even all 8*bs routes' unpadded GEMM work is below the bandwidth term at
    # these two shapes, so the inversion is not accidentally of a compute max.
    io_bytes = bs * H * 4
    nominal_weight = (sol_ns * 1e-9 * HBM - io_bytes) / SHARE
    distinct = nominal_weight / EXPERT_BYTES
    floor_us = (nominal_weight + io_bytes) / HBM * 1e6
    return {
        "bs": bs,
        "incumbent_us": time_us,
        "nominal_mean_distinct": distinct,
        "nominal_code_bytes": distinct * CODE_BYTES,
        "nominal_scale_bytes": distinct * SCALE_BYTES,
        "nominal_weight_bytes": nominal_weight,
        "input_output_bytes": io_bytes,
        "corrected_nominal_hbm_floor_us": floor_us,
        "max_unpadded_math_us": 2 * (8 * bs) * VALUES / BF16_FLOPS * 1e6,
    }


def projected_board(targets):
    factors = [old / targets.get(bs, old) for bs, old, _ in CASES]
    score = CHECK_BOARD * math.prod(factors) ** (1 / len(CASES))
    return {"score": score, "gain_vs_best": score / BEST_BOARD - 1}


def ss_shared_bytes_per_expert():
    # A: 64 physical/64 useful rows. B: 64 physical/32 useful rows.
    # Each valid code/scale byte is read once from HBM and then from shared.
    # Each expanded matrix is stored to shared and read by the tensor cores.
    decoded_bytes = (2 * H * H + 2 * H * H) * 2
    return {
        "tma_raw_writes": EXPERT_BYTES,
        "generic_raw_reads": EXPERT_BYTES,
        "generic_decoded_writes": decoded_bytes,
        "tensorcore_weight_reads": decoded_bytes,
        "generic_reads_and_writes": EXPERT_BYTES + decoded_bytes,
        "all_weight_ports_sum": 2 * EXPERT_BYTES + 2 * decoded_bytes,
    }


def report():
    shapes = [nominal_case(*case) for case in CASES if case[0] in (16, 32)]
    ideals = {s["bs"]: s["corrected_nominal_hbm_floor_us"] for s in shapes}
    # A target, not an achieved performance number. It has effectively no
    # margin over the user's >=3%-above-best build requirement.
    targets = {16: 84.0, 32: 109.0}
    per_port = ss_shared_bytes_per_expert()
    for s in shapes:
        d = s["nominal_mean_distinct"]
        t = targets[s["bs"]]
        s["conditional_target_us"] = t
        s["target_weight_port_TBps"] = {k: d * v / (t * 1e6) for k, v in per_port.items()}
    return {
        "source_job": SOURCE_JOB,
        "code_bytes_per_expert": CODE_BYTES,
        "scale_bytes_per_expert": SCALE_BYTES,
        "weight_bytes_per_expert": EXPERT_BYTES,
        "actual_job_route_histograms_available": False,
        "shape_estimates": shapes,
        "hbm_only_ceiling": projected_board(ideals),
        "conditional_target": projected_board(targets),
        "score_needed_for_3pct": BEST_BOARD * 1.03,
        "required_product_of_two_shape_speedups": (BEST_BOARD * 1.03 / CHECK_BOARD) ** 6,
        "shared_weight_bytes_per_expert_by_port": per_port,
        "shared_ports_capacity_or_concurrency_verified": False,
        "build_gate": "NOT PASSED: actual routed-byte evidence and independent/shared port limits unresolved; target lacks >=3% margin",
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
