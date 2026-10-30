"""Package the single grouped-statistic tiny norm hypothesis with retained gates."""

import ast
import hashlib
import json

from norm_group_source import derive, OUT
from tinyroute_source import HERE, ROOT, BASE_SHA
from harness import append_harness

GATE_SHA='5d691341d127fdf6f866be9b7e99d06502655665b19d1d229b1ca1cff1941550'


def main():
    parent=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==BASE_SHA
    gate=(HERE/'moe_tinyroute_bounded.py').read_bytes()
    assert hashlib.sha256(gate).hexdigest()==GATE_SHA
    original,cuda,cpp,vals=derive()
    assert (OUT/'packaged.cu').read_bytes()==cuda.encode()
    audit=json.loads((OUT/'payload-compiler-binding.json').read_text(encoding='utf-8'))
    assert audit['candidate_CUDA_sha256']==hashlib.sha256(cuda.encode()).hexdigest()
    for c in audit['compiles'].values():
        assert c['rc']==0
        assert all(k['stack_frame']==k['spill_stores']==k['spill_loads']==0 for k in c['kernels'])
    base=parent.decode();tree=ast.parse(base)
    ext=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_ext')
    loader=ast.get_source_segment(base,ext)
    for old,new in [('_ext(', '_tn_ext('),('_EXTS','_TN_EXTS'),('CUDA_SRC','CUDA_TN_SRC'),
                    ('CPP_SRC','CPP_TN_SRC'),('okc_moe_w4a16_v216x_pair1dq8_','okc_moe_norm_group_v1_resources_'),
                    ('functions=["okc_run", "okc_set_pdl"]','functions=["okc_tnorm_run", "okc_tnorm_set_pdl", "okc_tnorm_resources"]')]:
        assert old in loader;loader=loader.replace(old,new)
    marker='        _TN_EXTS[defines].okc_set_pdl(1 if PDL else 0)'
    assert marker in loader
    # Adapt our own object; never assign attributes on a loaded extension module.
    aliases='''        metadata = _TN_EXTS[defines].okc_tnorm_resources()
        print('TNORM_RESOURCE', tag, dict(zip(('registers', 'local_bytes', 'static_shared_bytes', 'blocks_per_SM', 'dynamic_shared_bytes', 'threads'), metadata)), flush=True)
        if not (metadata[1] == metadata[2] == 0 and metadata[3] >= 1 and metadata[4:] == [217600, 288]):
            raise RuntimeError('tnorm resource guard rejected before probe: ' + repr(metadata))
        _TN_EXTS[defines] = _TNormExtension(_TN_EXTS[defines])
'''
    loader=loader.replace(marker,aliases+marker,1)
    marker='    defines = tuple(MAIN_DEFINES if defines is None else defines)'
    assert marker in loader
    loader=loader.replace(marker,marker+'\n    assert "-DOKC_TINY=1" in defines and "-DOKC_WS=1" in defines and all(not d.startswith("@") for d in defines)',1)
    gtext=gate.decode();gtree=ast.parse(gtext)
    wanted=('_tr_launch','_tr_prepare_build','_tr_probe_data','_tr_mutate','_trb_snapshot','_trb_relative','_trb_gate','_tr_probe','prepare','kernel_fn')
    parts=[]
    for name in wanted:
        matches=[n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name==name]
        assert matches;part=ast.get_source_segment(gtext,matches[-1])
        if name=='_tr_prepare_build':
            old="_ext(tuple(defs) + tuple(sfx) + ('-DOKC_ROUTE_WARP=1',))"
            assert part.count(old)==1;part=part.replace(old,'_tn_ext(tuple(defs) + tuple(sfx))',1)
        part=part.replace('_trb_','_tnb_').replace('_tr_','_tn_').replace('tr8','tnorm').replace('TR8','TNORM')
        parts.append(part)
    prefix=append_harness(base)
    src=prefix+'\n\n# Private tiny grouped-statistic norm variant; original parent definitions remain exact.\n'
    adapter='class _TNormExtension:\n    def __init__(self, module):\n        self._module = module\n        self.okc_run = module.okc_tnorm_run\n        self.okc_set_pdl = module.okc_tnorm_set_pdl\n\n'
    src+='CUDA_TN_SRC = '+repr(cuda)+'\nCPP_TN_SRC = '+repr(cpp)+'\n_TN_EXTS = {}\n\n'+adapter+loader+'\n\n'
    src+='_tn_parent_prepare = prepare\n_tn_parent_kernel_fn = kernel_fn\n\n'+'\n\n'.join(parts)+'\n'
    raw=src.encode();ast.parse(raw);assert len(raw)<1_000_000 and raw.startswith(parent.rstrip(b'\n'))
    (HERE/'moe_norm_group.py').write_bytes(raw)
    record=dict(source_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),parent_sha256=BASE_SHA,
        CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),gate_source_sha256=GATE_SHA,
        actual_path_marker='norm_group::moe_group_fused',baseline_definitions_exact=True,
        scope='Only prepare-selected M1..8; original routing,GEMM/inter/ybuf arithmetic and all M>8 fallbacks retained. Their byte gates remain mandatory.',
        gate='Human-approved MoE norm/output bounds; inter/ybuf bytes;15 changing fixtures per shape;>3% on both one/skew populations in both arm orders.',
        prepare_cutoffs=dict(new_diagnostics_seconds=274,timing_seconds=278),
        CUDA_compiles='Four tiny/PRMT/LUT variants;zero stack/spills',
        GPU_jobs=0,admitted_for_GPU=False,
        next_action='TR8 is closed; fresh global-job/quota/identity/spacing admission required before one paced MoE check.',
        runtime_resource_gate='Before any new-path probe: query function attributes and occupancy;require zero local/static shared bytes,at least one block/SM,217600 dynamic bytes and288 threads.')
    (OUT/'payload-binding.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record,indent=2))


if __name__=='__main__':main()
