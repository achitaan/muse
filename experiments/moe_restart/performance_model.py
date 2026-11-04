"""Static traffic/issue model, NOT a benchmark or an attribution measurement.

Measured values are from judge e61f95e5a91944c8. Hardware ceilings (4.4 TB/s
HBM, 989 dense BF16 TF/s) are optimistic H200 SXM assumptions. Routing
histograms and local-memory instruction counts were NOT in that log.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBSERVED = ((1,26.152,114.976), (8,68.872,420.008),
            (16,102.944,1060.120), (32,121.608,1230.688),
            (128,174.288,2371.072), (256,235.072,3410.208))
WEIGHTS_A = 8192 * 4096 * 9 // 16
WEIGHTS_B = 4096 * 4096 * 9 // 16
WEIGHTS = WEIGHTS_A + WEIGHTS_B


def tile(rows, n, revised):
    groups = (4 if n <= 64 else 2 if n <= 128 else 1) if revised else 1
    stages = 64 // groups
    wgs = rows // 64
    return dict(rows=rows, n=n, stage_k=groups*64, stages=stages,
                weight_bytes_per_k64=wgs*2304,
                weight_bytes_per_full_k=wgs*2304*64,
                activation_bytes_per_stage=n*64*2*groups,
                activation_bytes_per_full_k=n*4096*2,
                ordinary_decode_warp_instructions_per_k64_estimate=64*(rows//16),
                scale_shuffle_warp_instructions_per_k64=2*(rows//16),
                wgmma_instructions_per_full_k=256*wgs,
                wgmma_fence_commit_wait_triples_per_full_k_per_wg=128,
                wgmma_final_drain_per_full_k_per_wg=1,
                activation_full_waits_per_consumer_thread=stages,
                flops_per_full_k=2*rows*n*4096,
                accumulator_registers_per_consumer_thread=n//2,
                register_code_lookahead=(4 if n<=32 else 2 if n<=64 else 1) if revised else (6 if n<=32 else 3),
                empty_arrivals_per_full_k=stages*(rows//16))


def main():
    # Independent scalar count: A+B has 50,331,648 weights/expert; every
    # weight contributes one multiply-add per padded token (2 FLOPs).
    assert WEIGHTS == 28311552
    assert 2 * 12288 * 4096 == 100663296
    layouts = [tile(r,n,new) for new in (False,True) for r in (64,128)
               for n in (8,16,32,64,96,128,192,256)]
    for item in layouts:
        assert item['weight_bytes_per_full_k'] * (12288 // item['rows']) == WEIGHTS
        assert item['wgmma_instructions_per_full_k'] * (12288 // item['rows']) == 49152
    spill = []
    for m,old,new in OBSERVED:
        # With AT MOST 32 active experts this is a LOWER bound on the hot
        # local bytes/thread/K64 needed if HBM bandwidth explained ALL excess.
        # It is an illustrative equivalent, not a claim those bytes occurred.
        equivalent = (new-old)*4.4e6/(32*96*288*64) if m>=16 else None
        spill.append(dict(m=m,old_us=old,new_us=new,ratio_of_medians_not_gate_geomean=new/old,
                          equivalent_extra_bytes_per_thread_k64_at_4_4TBps_32experts=equivalent))
    report = dict(measured_job='e61f95e5a91944c8',
                  assumptions={'HBM_bytes_per_us':4.4e6,'BF16_flops_per_us':989e6,
                               'decode_thread_instructions_per_weight_estimate':2,
                               'local_frame_bytes_is_not_a_dynamic_spill_count':True},
                  expert={'A_weight_bytes':WEIGHTS_A,'B_weight_bytes':WEIGHTS_B,
                          'weight_floor_us':WEIGHTS/4.4e6,
                          'decode_warp_instructions_estimate':3145728,
                          'scale_shuffle_warp_instructions':98304,
                          'wgmma_instructions':49152,
                          'flops_per_padded_token':100663296,
                          'R64_logical_shared_bytes_per_padded_token':3145728,
                          'R128_logical_shared_bytes_per_padded_token':2359296},
                  observed_and_equivalents=spill, tile_models=layouts)
    path = ROOT/'results/moe_restart_performance_model.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tile_models'},indent=2))


if __name__ == '__main__':
    main()
