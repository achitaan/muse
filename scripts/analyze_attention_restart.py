"""Reproduce the v1/v2 slowdown accounting from paired logs and owned source.

No GPU profiler counters are inferred. Instruction counts are source-level
warp/group operations, and pipeline figures are capacity upper bounds, NOT
measurements of outstanding transactions. Clang compiles only host layouts.
"""
import ast
import json
import math
from pathlib import Path
import random
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PROFILES=Path('/private/var/folders/pm/r8xm1mr525v913lct5tdg_5w0000gn/T/opencode/subs/profiles')


def source_string(path,name):
    tree=ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)


def load_log(path):
    found=[]
    pat=r"^RR search (\S+) (\{.*?\}) base=([\d.]+) new=([\d.]+) ratios=([\d.]+)/([\d.]+)"
    for line in path.read_text().splitlines():
        m=re.match(pat,line)
        if m:
            label,plan,base,new,ratio,paired=m.groups()
            found.append(dict(label=label,plan=ast.literal_eval(plan),base_us=float(base),
                              new_us=float(new),median_ratio=float(ratio),paired_ratio=float(paired)))
    return found


def layouts():
    base=source_string(ROOT/'board_best/attention_kb1h2_scored_2.6536.py','CUDA_SRC')
    cfg=base[base.index('template <int L, bool F>\nstruct Cfg'):base.index('constexpr int TILE = 64;')]
    start=base.index('template <int L, bool F>\n    struct Smem')
    smem=base[start:base.index('\n};',start)+3]
    text='#include <cstdint>\n#include <cstdio>\n#define __device__\n#define __forceinline__ inline\nusing __nv_bfloat16=uint16_t;\n'
    text+='namespace incumbent {\n'+cfg+'\nconstexpr int TILE=64,ROWB=128,STAGE_BYTES=8192,MAXB=512,MAXSEG=64;\n'+smem+'\n}\n'
    for version in (2,3):
        src=source_string(ROOT/f'.tools/attention_restart_v{version}_tail.py','RR_CUDA')
        start=src.index('template<int L, int TILE> struct Shared')
        structure=src[start:src.index('\n};',start)+3]
        text+=f'namespace v{version} {{ constexpr int ROW=128;\n'+structure+'\n}\n'
    text+='int main(){\n'
    for L in (1,8):
        text+=f'printf("incumbent_vj_L{L} %zu\\n",sizeof(incumbent::Smem<{L},true>));\n'
    for version,geometries in ((2,((1,64),(1,128),(8,64),(8,256))),
                               (3,((1,64),(1,128),(8,64),(8,128)))):
        for L,T in geometries:
            text+=f'printf("v{version}_L{L}_T{T} %zu\\n",sizeof(v{version}::Shared<{L},{T}>));\n'
    text+='}\n'
    target=ROOT/'.tools/attn_restart_layouts.cpp'
    exe=ROOT/'.tools/attn_restart_layouts'
    target.write_text(text)
    subprocess.run(['clang++','-std=c++20',str(target),'-o',str(exe)],check=True)
    run=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    data={k:int(v) for k,v in (l.split() for l in run.stdout.splitlines())}
    # Actual v16F dispatch is WST, not the generic vj engine: NST=4, NPW=4.
    # Direct formula from launch_ct14b: four raw 128-key K/V slots, four
    # transposed V images, barriers/prefix, and the fold copy staging buffer.
    data['incumbent_v16F_WST']=4*32768+4*9216+2*4*8+64+((4*(16+1)+15)&~15)+36864
    return data


def main():
    logs={v:PROFILES/f'{name}.{job}.log' for v,name,job in
          ((1,'attention_restart','2847448242174b71'),(2,'attention_restart_v2','c36537ede3ff45c1'))}
    records={v:load_log(p) for v,p in logs.items()}
    report=dict(sources={v:str(p) for v,p in logs.items()},log_tail_warning='v2 log starts part-way through v1F; sliding lines are absent',
                little_law=dict(peak_bytes_per_s=4.8e12,latency_s=1e-6,SMs=132,bytes_per_SM=4.8e12*1e-6/132),
                shared_bytes=layouts(),summaries={})
    for version,rows in records.items():
        best=[]
        for label in sorted({r['label'] for r in rows}):
            row=min((r for r in rows if r['label']==label),key=lambda r:r['paired_ratio']).copy()
            B=int(label[1:-1]);L=8 if label.startswith('v') else 1
            if label.endswith('F'):
                # Exact lengths of these independent _at_data probes, NOT
                # an assumption about the judge's fresh scored input lengths.
                rng=random.Random(0xAC417AA+101*B+13*L)
                lengths=[80000] if B==1 else [rng.randrange(74000,86001) for _ in range(B)]
                kv=256*sum(s+L for s in lengths)
                tiles=sum((s+L+63)//64 for s in lengths)
                row.update(probe_seq_lens=lengths,unique_KV_bytes=kv,
                           gross_useful_TB_s=kv/row['new_us']/1e6,
                           equivalent_64key_tile_period_us_per_SM=row['new_us']*132/tiles,
                           base_tile_period_us_per_SM=row['base_us']*132/tiles)
            best.append(row)
        report['summaries'][version]=best
    report['capacity_upper_bounds']=[
        dict(name='v2 verify T64',tile=64,stages=3,planned_CTA_per_SM=2,max_future_bytes_SM=2*2*64*256),
        dict(name='v2 verify T256',tile=256,stages=2,planned_CTA_per_SM=1,max_future_bytes_SM=1*1*256*256),
        dict(name='incumbent v32F vj',tile=64,stages=4,planned_CTA_per_SM=2,max_future_bytes_SM=2*3*64*256),
        dict(name='incumbent v16F WST',tile=128,stages=4,planned_CTA_per_SM=1,max_future_bytes_SM=1*3*128*256),
        dict(name='v3 verify T64',tile=64,stages=4,planned_CTA_per_SM=2,max_future_bytes_SM=2*3*64*256),
        dict(name='v3 verify T128',tile=128,stages=3,planned_CTA_per_SM=2,max_future_bytes_SM=2*2*128*256),
    ]
    report['source_level_verify_ops_per_64keys']=dict(
        restart_warp_mma_QK=8*8*2,restart_warp_mma_PV=8*8*4,
        restart_K_shared_scalar_warp_loads=8*8*2*2,
        restart_V_shared_uint2_warp_loads=8*4*4,
        restart_V_byte_permutation_warp_ops=8*4*8,
        restart_V_FP8_to_half2_warp_conversions=8*4*16,
        restart_K_shared_bytes=32768,restart_V_shared_bytes=32768,
        incumbent_v32F_vj_warpgroup_mma=8,
        incumbent_v16F_WST_warpgroup_mma_equivalent=6,
        limitation='PTX/source operation counts, NOT SASS counts or cycle attribution')
    report['v2_matched_half_control']=[
        dict(label=r['label'],plan=r['plan'],half_us=r['new_us'],
             float_us=next(x['new_us'] for x in records[2] if x['label']==r['label'] and
                           {k:v for k,v in x['plan'].items() if k!='half'}==
                           {k:v for k,v in r['plan'].items() if k!='half'} and x['plan']['half']==0))
        for r in records[2] if r['plan']['half']==1 and r['label'] in ('v16F','v32F')]
    report['compile_tools']='nvcc/ptxas/NVRTC not found in PATH, Spotlight or bounded common-path search; broad user-tree find timed out after 120s without a match'
    report['register_spills']='unknown: GPU logs suppress ptxas output; v3 adds verbose ptxas and persistent runtime function-attribute/occupancy logs'
    (ROOT/'results/attn_restart_slowdown.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('v2_matched_half_control',)},indent=2))


if __name__=='__main__': main()
