"""Read-only inspection snapshots; never write into the alygg reference tree."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT.parent/'alygg/kernels/muse_moe_w4a16.py'
inputs={'base':ROOT/'experiments/boards/moe/moe_b15.py','reference':REF}
manifest={}
for name,p in inputs.items():
    raw=p.read_bytes();t=raw.decode();a=ast.parse(t)
    src=ast.literal_eval(next(n.value for n in a.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='CUDA_SRC' for v in n.targets)))
    (ROOT/'.kperm_offline'/(name+'.cu')).write_text(src)
    manifest[name]=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),cuda_sha256=hashlib.sha256(src.encode()).hexdigest())
    print(name,[(n.name,n.lineno,n.end_lineno) for n in a.body if isinstance(n,ast.FunctionDef)])
(ROOT/'.kperm_offline/inputs.json').write_text(json.dumps(manifest,indent=2))
