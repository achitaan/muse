"""Freeze the one FP32 cache source; no GPU/API use."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PARENT='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
CUDA='403bd12392a617b28ef14f5f92c0a8892c465b8ed851773b2ffe4d036d233733'


def main():
    parent=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==PARENT
    out=ROOT/'results/moe-fp32-final-cache-preflight-20261029'
    cuda=(out/'full.cu').read_bytes()
    assert hashlib.sha256(cuda).hexdigest()==CUDA
    addon=(ROOT/'experiments/prepared/moe_fp32_final_cache_addon.py').read_bytes()
    raw=parent+b'\n\n_F32CACHE_CUDA = '+repr(cuda.decode()).encode()+b'\n'+addon
    tree=ast.parse(raw);old=ast.parse(parent)
    assert [ast.dump(n) for n in tree.body[:len(old.body)]]==[ast.dump(n) for n in old.body]
    assert raw.startswith(parent)
    binding=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),parent_sha256=PARENT,
                 CUDA_sha256=CUDA,parent_is_exact_prefix=True,parent_AST_unchanged=True,
                 addon_sha256=hashlib.sha256(addon).hexdigest(),
                 affected_official_indices=[2,4,5],controls=[0,1,3],
                 namespace='okc_f32cache',untouched='M<=16 and non-PRMT fallback',GPU_jobs=0)
    for name,data in [('submission.py',raw),('source-binding.json',(json.dumps(binding,indent=2)+'\n').encode())]:
        path=out/name
        assert not path.exists() or path.read_bytes()==data,('Frozen collision',name)
        path.write_bytes(data)
    print(json.dumps(binding))


if __name__=='__main__':main()
