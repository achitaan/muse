"""Apply the authorized TR8 numerical gate and prepare diagnostic deadline."""

import ast
import hashlib
import json

from tinyroute_source import HERE, ROOT, BASE_SHA
from build_norm_pair_diag import extract

OUT=ROOT/'results/decode-moe-plan-20261030/tinyroute-bounded'
OLD_SHA='958335479991016b3cebae2ad9517ef8f75c036fc1d8ac63a0b8459395219498'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=(HERE/'moe_tinyroute.py').read_bytes();assert hashlib.sha256(old).hexdigest()==OLD_SHA
    source=old.decode();tree=ast.parse(source);lines=source.splitlines(keepends=True)
    probe=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_tr_probe')
    first=sum(map(len,lines[:probe.lineno-1]));last=sum(map(len,lines[:probe.end_lineno]))
    snapshot=extract((HERE/'moe_repeat_v2_runtime.py').read_text(encoding='utf-8'),['_mr_snapshot'])
    snapshot=snapshot.replace('def _mr_snapshot(','def _trb_snapshot(',1)
    replacement=snapshot+'\n'+(HERE/'tinyroute_bounded_probe.py').read_text(encoding='utf-8')+'\n'
    updated=source[:first]+replacement+source[last:]
    old_label="print('TR8_BITWISE', m, ok";new_label="print('TR8_BOUNDED', m, ok"
    assert updated.count(old_label)==1;updated=updated.replace(old_label,new_label,1)
    # New shape diagnostics must start before274s; timing retains its278s cutoff.
    old_deadline='not _okh_budget(start, 278.)'
    new_deadline='not _okh_budget(start, 274.)'
    assert updated.count(old_deadline)==3
    updated=updated.replace(old_deadline,new_deadline,2)
    assert updated.replace(replacement,source[first:last],1).replace(new_label,old_label,1).replace(new_deadline,old_deadline,2)==source
    raw=updated.encode();ast.parse(raw);assert len(raw)<1_000_000
    (HERE/'moe_tinyroute_bounded.py').write_bytes(raw)
    before={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)}
    after={n.targets[0].id:n.value for n in ast.parse(updated).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)}
    assert ast.dump(before['CUDA_SRC'])==ast.dump(after['CUDA_SRC'])
    for name in ('kernel_fn','_tr_launch','_ext','_make_scratch','prepare'):
        aa=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name]
        bb=[n for n in ast.parse(updated).body if isinstance(n,ast.FunctionDef)and n.name==name]
        assert len(aa)==len(bb)
        for a,b in zip(aa,bb):
            if name!='prepare':assert ast.dump(a)==ast.dump(b)
            else:
                original=ast.unparse(a);changed=ast.unparse(b)
                expected=original.replace("'TR8_BITWISE'","'TR8_BOUNDED'")
                if "'TR8_BITWISE'"in original:
                    expected=expected.replace('_okh_budget(start, 278.0)','_okh_budget(start, 274.0)',2)
                assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(changed))
    record=dict(source_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),retained_MoE_sha256=BASE_SHA,
        prior_TR8_source_sha256=OLD_SHA,CUDA_AST_exact=True,timed_dispatch_ext_and_scratch_AST_exact=True,
        parent_prepare_change='Diagnostic label and274s new-diagnostic cutoff only;278s timing cutoff and original >3% both orders on one/skew populations retained.',
        gate_change='Human authorizes MoE ss/output variance only. Exact finite inter/ybuf,ss relativeL2<=1e-5,out relativeL2<=.001,worstrow<=.005 with official floor; nonlocal outputs exactly zero.',
        reopened_because='Prior TR8 check selected no timed new path due the now-superseded byte gate;75 H200 diagnostic fixtures localize variation to ss/out and preserve inter/ybuf bytes.',
        actual_path_marker='okc::moe_tr8_fused',unchanged='Every precision boundary,route job,GEMM,norm implementation,completion,generation,both banks,M>8 fallback.',
        GPU_jobs=0,admitted_for_GPU=False,Decode_or_Attention_gate_relaxed=False)
    (OUT/'source-binding.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
