"""Three-CTA L1 flat specialization, with complete bounded segment storage."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_attention_fp8q import ROOT,literal

BASE='a310b988ce0eaccb7ff5db7016dc041ec900019528bc8c136e34a278c3fd9372'

def parent_source():
    paths=[ROOT/'.attention-measurements'/BASE/'submission.py',
           ROOT/'results/resumed-attention-fp8pv-20261028/submission.py',ROOT/'best/attention_checked.py']
    raw=next(p.read_bytes() for p in paths if p.is_file())
    assert hashlib.sha256(raw).hexdigest()==BASE
    return raw

def transform(old):
    edits=[]
    def change(a,b,count=1):
        nonlocal old
        assert old.count(a)==count,(a[:100],old.count(a))
        old=old.replace(a,b);edits.append(dict(before=a,after=b,count=count))
    change('#include <type_traits>','#include <type_traits>\n#include <cstdio>')
    change('static constexpr int MINB = 2;', 'static constexpr int MINB = (L == 1 && F) ? 3 : 2;')
    change('(65536 / (2 * NTX))','(65536 / (MINB * NTX))',2)
    change('int seg[F ? MAXSEG : 1][3];','int seg[F ? ((L == 1) ? MAXB : MAXSEG) : 1][3];')
    change('S.cum[b] < ceg && n < MAXSEG','S.cum[b] < ceg && n < MAXB')
    change('attn_kernel_fp8pv','attn_kernel_pv3',old.count('attn_kernel_fp8pv'))
    marker='    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);'
    change(marker,marker+'''
    if constexpr (L == 1) {
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, attn_kernel_pv3<L, true>);
      int regs_sm = 0, shared_sm = 0;
      cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
      cudaDeviceGetAttribute(&shared_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
      printf("ATTN_PV3_RESOURCES %d %zu %d %d %zu %d %d %d %d %d %d\\n",
             attr.numRegs, attr.localSizeBytes, occ_f, sm_f, sizeof(Smem<L, true>),
             Cfg<L, true>::MINB, Cfg<L, true>::BASEREGS, Cfg<L, true>::CREGS,
             nsm, regs_sm, shared_sm);
    }''')
    return old,edits

def build():
    raw=parent_source();text=raw.decode();tree=ast.parse(text);node,old=literal(text)
    new,edits=transform(old)
    inverse=new
    for e in reversed(edits):
        assert inverse.count(e['after'])==e['count']
        inverse=inverse.replace(e['after'],e['before'])
    assert inverse==old
    plan=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_plan')
    fits=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fits')
    plan2=copy.deepcopy(plan)
    # Only the first, full/multirequest branch changes its quota.
    ret=plan2.body[3].body[0]
    assert isinstance(ret,ast.Return) and isinstance(ret.value,ast.Call)
    split=next(k for k in ret.value.keywords if k.arg=='n_splits')
    assert ast.unparse(split.value)=='max(1, target // pairs)'
    split.value=ast.parse('max(1, ((3 if L == 1 else 2) * sms) // pairs)',mode='eval').body
    fits2=copy.deepcopy(fits)
    bound=ast.parse("cfg['batch'] * (data['page_table'].shape[1] * cfg['page_size'] + cfg['q_len'] + 600) < INDEX_LIMIT",mode='eval').body
    fits2.body[-1].value=ast.BoolOp(op=ast.And(),values=[fits2.body[-1].value,bound])
    lines=text.splitlines(keepends=True)
    def offsets(n):return sum(map(len,lines[:n.lineno-1]))+n.col_offset,sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
    changes=[(*offsets(node.value),repr(new)),(*offsets(plan),ast.unparse(plan2)),(*offsets(fits),ast.unparse(fits2))]
    result=text
    for start,end,replacement in sorted(changes,reverse=True):result=result[:start]+replacement+result[end:]
    updated=ast.parse(result)
    for original in (node,plan,fits):
        name=original.targets[0].id if isinstance(original,ast.Assign) else original.name
        current=next(n for n in updated.body if
                     (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)) or
                     (isinstance(n,ast.FunctionDef) and n.name==name))
        updated.body[updated.body.index(current)]=original
    assert ast.dump(updated)==ast.dump(tree)
    out=ROOT/'.tools/attention-three-cta-variant.py';out.write_bytes(result.encode())
    (ROOT/'.tools/attention-three-cta.cu').write_bytes(new.encode())
    report=dict(parent_sha256=BASE,source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,
                inverse_cuda_exact=True,all_other_ast_unchanged=True,changed_functions=['_plan','_fits'],
                math_unchanged=True,edits=edits,
                python_edits=[dict(before=ast.unparse(a),after=ast.unparse(b)) for a,b in ((plan,plan2),(fits,fits2))])
    (ROOT/'.tools/attention-three-cta-source-audit.json').write_bytes(json.dumps(report,indent=2).encode())
    print(json.dumps({k:v for k,v in report.items() if k not in ('edits','python_edits')},indent=2))

if __name__=='__main__':build()
