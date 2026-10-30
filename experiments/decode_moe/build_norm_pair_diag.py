"""Freeze a Dense-container diagnostic of the exact proposed Decode handoff."""

import ast
import base64
import hashlib
import json
import zlib

from pdl16_sources import HERE, ROOT, BASE_SHA
from pdli_sources import pdli_pack
from harness import append_harness

OUT = ROOT / 'results/decode-moe-plan-20261030/norm-pair-diag'
DENSE_SHA = 'edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b'


def extract(source, names):
    tree = ast.parse(source)
    found = {n.name:ast.get_source_segment(source,n) for n in tree.body
             if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    assert set(found)==set(names),set(names)-set(found)
    return '\n\n'.join(found[n] for n in names)+'\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parent=(ROOT/'best/verified.py').read_bytes()
    decode=(HERE/'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==DENSE_SHA
    assert hashlib.sha256(decode).hexdigest()==BASE_SHA
    tree=ast.parse(decode)
    expected=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_PDLI_EXPECTED')
    binding=json.loads((OUT.parent/'pdli-cpu/source-binding.json').read_text())
    sources={k:(OUT.parent/'pdli-cpu'/f'{k}.cu').read_text() for k in ('r1','r8')}
    assert all(hashlib.sha256(s.encode()).hexdigest()==expected[k] for k,s in sources.items())
    specs={k:binding['specs'][k]for k in sources}
    cuda,cpp,functions=pdli_pack(sources,specs)
    sha=hashlib.sha256(cuda.encode()).hexdigest()
    (OUT/'r2f-combined.cu').write_bytes(cuda.encode())
    (OUT/'r2f-combined.cpp').write_bytes(cpp.encode())
    small=(HERE/'decoder_pdls_draft.py').read_text()
    small_binding=json.loads((OUT.parent/'pdl-norm-small/candidate-binding.json').read_text())
    assert hashlib.sha256((HERE/'decoder_pdls_draft.py').read_bytes()).hexdigest()==small_binding['candidate_sha256']
    app=append_harness('')+'\nimport hashlib\n'
    app+='def _np_zsrc(s):\n import base64,zlib\n return zlib.decompress(base64.b64decode(s)).decode()\n'
    app+=extract(small,['pdln_derive','_pdls_build','_pdls_boundary'])
    for key in ['_PDLS_HEADERS','_PDLS_QUERY','_PDLS_NOTICE','_PDLS_SOURCE_SHA']:
        node=next(n for n in ast.parse(small).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name)and n.targets[0].id==key)
        app+=ast.get_source_segment(small,node)+'\n'
    app+=extract(decode.decode(),['_PdliFns','_pdli_rc','_pdli_route','_pdli_equal','_pdli_probe'])
    app+='_NP_R_CUDA = _np_zsrc('+repr(base64.b64encode(zlib.compress(cuda.encode(),9)).decode())+')\n'
    for key,value in {'_NP_R_CPP':cpp,'_NP_R_FUNCTIONS':functions,'_NP_R_SHA':sha,
        '_NP_PUBLIC_SHA':json.loads((OUT.parent/'pdl-chain/stock-norm-pinned/origin.json').read_text())['files']['norm_residual.py']['sha256']}.items():
        app+=key+' = '+repr(value)+'\n'
    app+=(HERE/'norm_pair_diag_runtime.py').read_text()
    raw=parent+app.encode();ast.parse(raw)
    assert raw.startswith(parent)and len(raw)<1_000_000
    (HERE/'dense_norm_pair_diag.py').write_bytes(raw)
    result=dict(source_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
        Dense_parent_sha256=DENSE_SHA,Decode_parent_sha256=BASE_SHA,
        norm_candidate_sha256=small_binding['candidate_sha256'],
        norm_CUDA_sha256=small_binding['private_CUDA_sha256'],R2f_CUDA_sha256=sha,
        R2f_individual_sha256={k:expected[k]for k in sources},R2f_specs=specs,
        timed_kernel_unchanged=True,enables_nothing=True,GPU_jobs=0,
        question='Exact SDK/H200 norm/router bytes and causal early norm signal with identical incoming-PDL R2f; >3% and >=1.5us both orders.',
        limits='No whole-layer or aggregate gain follows from this standalone diagnostic.')
    (OUT/'source-binding.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
