"""Offline analysis of the supplied actual headroom histograms, not a benchmark."""
import argparse
import json
import math
from pathlib import Path
import statistics

EXPERT_BYTES = 27 * 2**20


def parse(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("MOE_HEADROOM case="):
            continue
        fields = dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
        fields["hist"] = [int(v) for v in fields["hist"].split(",")]
        for key in ("M", "draw", "D", "R", "dup_jobs", "bytes"):
            fields[key] = int(fields[key])
        for key in ("inc_us", "stream_us", "stream_TBps", "paired_gap_us"):
            fields[key] = float(fields[key])
        cap = 8 if fields["M"] == 16 else 64
        assert sum(fields["hist"]) == fields["R"]
        assert sum(v > 0 for v in fields["hist"]) == fields["D"]
        assert sum((v + cap - 1) // cap for v in fields["hist"]) - fields["D"] == fields["dup_jobs"]
        rows.append(fields)
    return rows


def report(rows):
    result = {}
    for M in (128, 256):
        selected = [r for r in rows if r["M"] == M]
        detail = []
        for r in selected:
            duplicate = sum((c + 63) // 64 for c in r["hist"]) - r["D"]
            saved = duplicate * EXPERT_BYTES
            # Use each draw's measured once-read logical bandwidth, not 4.8
            # TB/s or attribution of the entire paired residual to rereads.
            saved_us = saved / (r["stream_TBps"] * 1e6)
            old_x = sum(((c + 7) // 8) * 8 * 384 * 1024 for c in r["hist"] if c)
            new_x = sum(((c + 7) // 8) * 8 * 384 * 1024 * (4 if c > 128 else 2 if c > 64 else 1)
                        for c in r["hist"] if c)
            detail.append({"draw": r["draw"], "D": r["D"], "R": r["R"], "hist": r["hist"], "heavy": {str(e): c for e, c in enumerate(r["hist"]) if c > 64},
                           "dup_jobs": duplicate, "bytes_saved": saved,
                           "inc_us": r["inc_us"], "stream_us": r["stream_us"],
                           "activation_TMA_old_bytes": old_x, "activation_TMA_new_bytes": new_x,
                           "extra_activation_TMA_bytes": new_x - old_x,
                           "new_activation_TBps_required_at_byte_model_time": new_x / ((r["inc_us"] - saved_us + .02 * r["inc_us"]) * 1e6),
                           "byte_only_saved_us": saved_us,
                           "byte_only_cut_pct": 100 * saved_us / r["inc_us"],
                           "peak_hbm_saved_us": saved / 4.8e6,
                           "peak_hbm_cut_pct": 100 * saved / 4.8e6 / r["inc_us"]})
        cuts = [r["byte_only_cut_pct"] for r in detail]
        peak_cuts = [r["peak_hbm_cut_pct"] for r in detail]
        result[str(M)] = {"draws": detail, "median_byte_only_cut_pct": statistics.median(cuts),
                          "median_peak_hbm_cut_pct": statistics.median(peak_cuts),
                          "normal_stage_penalty_sensitivity_pct": {str(p): statistics.median(cuts) - p for p in (0, 1, 2, 3, 5)},
                          "if_extra_X_were_fully_serial_plus_2pct_stage_penalty": {
                              str(bw): statistics.median(r["byte_only_cut_pct"] - 2 - 100 * r["extra_activation_TMA_bytes"] / (bw * 1e6) / r["inc_us"] for r in detail)
                              for bw in (12, 20, 40, 80)},
                          "target": M == 256 and min(statistics.median(cuts), statistics.median(peak_cuts)) - 2 >= 3}
    result["notes"] = ["Byte-only savings are a bandwidth-equivalent opportunity, NOT measured runtime savings.",
                       "TQ64 repeats dequantization per active token-WG; it does not remove tensor-core work.",
                       "Build hypothesis requires normal-path K128/2-X-slot overhead below about 3%; per-shape ABBA gate must prove it.",
                       "Row/token-axis exchange raises activation TMA traffic 2x/4x for heavy experts; the positive model assumes that L2/shared traffic overlaps. Required aggregate activation bandwidth and pessimistic serial-cost sensitivity are explicitly reported.",
                       "Measured stream rates include fixed graph/ramp costs: the independent 4.8-TB/s byte-cost sensitivity avoids treating all those fixed costs as saved per extra expert.",
                       "M128 observed histograms have no >64-token experts; keep M128 incumbent.",
                       "quads=8 is ABBA repetitions, unrelated to jobs, WGs or token slabs."]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    args = p.parse_args()
    print(json.dumps(report(parse(args.log.read_text())), indent=2))
