"""One L1-flat cold-finalizer outlining hypothesis, without GPU evaluation."""

import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'results/decode-moe-plan-20261030/attention-finoutline'
PARENT_SHA='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'


def derive():
    raw=(ROOT/'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    text=raw.decode();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')=='CUDA_SRC')
    parent=ast.literal_eval(node.value)
    first=parent.index('template <int L, int NT>\n__device__ __forceinline__ void fin_nc1(')
    last=parent.index('// the q_len 8 flat loop',first)
    body=parent[first:last]
    outlined=body.replace('__forceinline__ void fin_nc1(','__noinline__ void fin_nc1_outlined(',1)
    assert outlined.replace('__noinline__ void fin_nc1_outlined(','__forceinline__ void fin_nc1(',1)==body
    # Keep the old helper for every unaffected template. No split/coop change.
    src=parent[:first]+outlined+parent[first:]
    old='    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);'
    new='    if constexpr (L == 1 && FLAT) fin_nc1_outlined<L, NT>(p, b, pair, bp, vscale, 0);\n    else fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);'
    assert src.count(old)==1;src=src.replace(old,new,1)
    assert src.replace(new,old,1).replace(outlined,'',1)==parent
    return raw,text,tree,node,parent,src


def main():
    OUT.mkdir(exist_ok=True);raw,text,tree,node,parent,src=derive()
    old=ast.get_source_segment(text,node);assert text.count(old)==1
    updated=text.replace(old,'CUDA_SRC = '+repr(src),1)
    old='okc_attn_v298{tag}_90a';new='okc_attn_finoutline_v1{tag}_90a'
    assert updated.count(old)==1;updated=updated.replace(old,new,1)
    after=ast.parse(updated)
    for a,b in zip(tree.body,after.body):
        if a is node:continue
        if isinstance(a,ast.FunctionDef)and a.name=='_build':
            assert ast.dump(a).replace('okc_attn_v298','okc_attn_finoutline_v1')==ast.dump(b)
        else:assert ast.dump(a)==ast.dump(b)
    blob=updated.encode();assert len(blob)<1_000_000
    (HERE/'attention_finoutline.py').write_bytes(blob)
    (OUT/'parent.cu').write_bytes(parent.encode());(OUT/'candidate.cu').write_bytes(src.encode())
    record=dict(parent_sha256=PARENT_SHA,parent_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
        source_sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),
        exact_inverse=True,scope='L1 flat finalizer call boundary only. All arithmetic,masks,cache writes,workspace ownership and launch plans retained.',
        hypothesis='Move the cold finalizer body out of the hot L1-flat kernel instruction footprint.',
        observed_hot_path='Retained Decode profiles attribute88.1/171.2us to full L1 attention at B16/B32; instruction-cache exposure is unmeasured.',
        not_repeated='No split/coop flag,wide tile,FP8 math change,external launch or score-bank pipeline.',
        CPU_admission='Reject added local/stack/spills or negligible hot-entry shrink before spending a GPU check.',
        GPU_jobs=0,admitted_for_GPU=False,Decode_checks_used=0)
    (OUT/'source-binding.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
