"""Unscored synthetic routing proxy; not measurements of challenge activations."""
import ast
import functools
import json
import math
from pathlib import Path
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
torch.set_num_threads(2)
tree=ast.parse((ROOT/'muse_prefill_layer/reference.py').read_bytes())
names={'HIDDEN','N_EXPERTS','LOCAL_EXPERTS','STREAM_SEED','STREAM_RANK','STREAM_MEAN','STREAM_SHARED','ROUTER_STD','ROUTER_ALIGN','ROUTER_BIAS_STD','TOP_K','ROUTER_MULTIPLIER'}
scope={'torch':torch,'math':math,'functools':functools}
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for target in n.targets for t in ast.walk(target)):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<public-constants>','exec'),scope)
fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_stream_basis','_router')]
exec(compile(ast.Module(body=fns,type_ignores=[]),'<public-router>','exec'),scope)
assert scope['N_EXPERTS']==256 and scope['LOCAL_EXPERTS']==32
mean,shared=scope['_stream_basis']('cpu');records=[]
T=753
for i in range(8):
    gen=torch.Generator().manual_seed(20261028+i)
    coef=torch.randn(T,scope['STREAM_RANK'],generator=gen)
    stream=scope['STREAM_MEAN']*mean+scope['STREAM_SHARED']*(coef@shared)+torch.randn(T,scope['HIDDEN'],generator=gen)
    # Proxy omits the attention/residual update and the private pre-FFN gain.
    h=stream/torch.sqrt((stream*stream).mean(-1,keepdim=True)+1e-5)
    router=scope['_router'](gen,'cpu',None)
    score=(scope['ROUTER_MULTIPLIER']*(h@router['router'].T)).sigmoid()+router['router_bias']
    ids=score.topk(scope['TOP_K'],-1).indices
    bucket=ids//scope['LOCAL_EXPERTS']
    busiest=int(torch.bincount(bucket.reshape(-1),minlength=8).argmax())
    active=(bucket==busiest).any(-1)
    records.append(dict(seed=20261028+i,tokens=T,busiest_bucket=busiest,zero_fraction=float((~active).float().mean()),routes_per_token=float((bucket==busiest).float().sum()/T)))
report=dict(kind='CPU synthetic proxy, not actual challenge zero fractions',
            public_router_constants=True,total_experts=256,local_experts=32,
            uniform_range_prior=(224/256)**8,busiest_range_selected=True,
            omitted=['attention/residual update','private pre-FFN gain','run router-load calibration'],
            records=records,mean_zero_fraction=sum(x['zero_fraction'] for x in records)/len(records),
            min_zero_fraction=min(x['zero_fraction'] for x in records),max_zero_fraction=max(x['zero_fraction'] for x in records))
(ROOT/'.tools/prefill-zero-row-proxy.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report,indent=2))
