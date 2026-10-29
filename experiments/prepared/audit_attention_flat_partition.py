"""Prove existing full-context partition behavior before proposing load balancing."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())


def extract(path,variable,plan):
    raw=path.read_bytes();tree=ast.parse(raw)
    cuda=next(n.value.value for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id==variable for t in n.targets))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==plan)
    if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):
        fn.body=fn.body[1:]  # Embedded source omits the standalone explanatory docstring.
    text=ast.unparse(fn).replace('_native_attn_','')
    assert 'if not window and 1 < B <= MAX_BATCH:' in text
    assert 'flat=1' in text and 'target = 2 * sms' in text
    markers=['constexpr bool fpart = L == 1;', 'const int seq = __ldg(p.seq_lens + b);',
             'S.cum[b + 1] = run + incl;',
             'const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31)',
             'int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);',
             'const bool in = lane_f < p.B && incl_l > csf && excl < cef;']
    assert all(s in cuda for s in markers)
    start=cuda.index('      const int64_t Tf =')
    stop=cuda.index('        S.fce = cef;',start)+len('        S.fce = cef;')
    segment=cuda[start:stop]
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(),CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                normalized_plan=text,partition_excerpt=segment,markers=markers)


def main():
    a=extract(ROOT/'best/attention_checked.py','CUDA_SRC','_plan')
    d=extract(ROOT/'best/decoder_verified.py','_native_attn_CUDA_SRC','_native_attn__plan')
    assert a['source_sha256']=='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
    assert d['source_sha256']=='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
    assert a['normalized_plan']==d['normalized_plan'] and a['partition_excerpt']==d['partition_excerpt']
    report=dict(sources={'attention':a,'Decode':d},existing_mechanism='Actual-sequence-length prefix sums; global virtual-key interval partition across requests; per-CTA segment list',
                applies_to='Retained L1 full context with1<B<=MAX_BATCH; not a claim for every fallback or L8 path',
                plan_and_partition_text_identical_between_retained_sources=True,
                metadata_stays_on_device=True,host_read_of_lengths=False,
                candidate_created=False,GPU_jobs=0,
                decision='A generic adaptive-length balancing proposal duplicates the incumbent. Require a distinct measured segment/pipeline bottleneck before a new trial.',
                limitations=['Source behavior and identity audit, not profiler critical-path evidence',
                             'No new performance or numerical claim'])
    out=ROOT/'results/attention-flat-partition-audit-20261029';out.mkdir(exist_ok=True)
    p=out/'audit.json';data=(json.dumps(report,indent=2)+'\n').encode()
    assert not p.exists() or p.read_bytes()==data,'Frozen audit collision'
    p.write_bytes(data)
    print(json.dumps(dict(source_hashes={k:v['source_sha256'] for k,v in report['sources'].items()},existing_global_length_balancing=True,candidate_created=False,GPU_jobs=0)))


if __name__=='__main__':main()
