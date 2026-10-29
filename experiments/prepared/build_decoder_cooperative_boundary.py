"""Build a frozen, losslessly packed whole-layer candidate; no GPU execution."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

from preflight_decoder_cooperative_boundary import HELPERS
from preflight_decoder_projection_pdl import CONDITIONAL

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PACKAGED='307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072'
D9='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
OLD=''' attn_output = _achita_dense_project(o, 'o', state)
 split_router = tokens > ROUTER_FP32_MAX_TOKENS
 (ffn_input, stream, router_input) = state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
'''
NEW=''' split_router = tokens > ROUTER_FP32_MAX_TOKENS
 _cb_fused = _cb_apply(o, data['stream'], state)
 if _cb_fused is None:
  attn_output = _achita_dense_project(o, 'o', state)
  (ffn_input, stream, router_input) = state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
 else:
  (ffn_input, stream, router_input) = _cb_fused
'''


def digest(raw): return hashlib.sha256(raw).hexdigest()


def freeze(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==raw, 'Frozen evidence collision: '+str(path)
    path.write_bytes(raw)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='v1');args=parser.parse_args()
    assert args.revision.replace('-','').isalnum()
    packaged=(ROOT/'results/decoder-unused-rope-prune-20261029/submission.py').read_bytes()
    assert digest(packaged)==PACKAGED
    baseline=(ROOT/'best/decoder_verified.py').read_bytes();assert digest(baseline)==D9
    prefix_sources=[];bindings=[]
    for family in ('o1','o8'):
        path=ROOT/f'results/decoder-cooperative-boundary-screen-20261029/compiler/{family}-cooperative.cu'
        proxy=path.read_text(encoding='utf-8')
        assert proxy.count(CONDITIONAL)==1 and proxy.count(HELPERS)==1
        native=proxy.replace(CONDITIONAL,'#include <type_traits>').replace(HELPERS,
            '#ifndef OKC_COOP_BOUNDARY_HELPERS\n#define OKC_COOP_BOUNDARY_HELPERS\n'+HELPERS+'\n#endif\n')
        symbol='smallm2' if family=='o1' else 'v18s'
        anchor=f'{symbol}_gemv('
        assert native.count(anchor)==1
        native=native.replace(anchor,f'cb_{symbol}_gemv(',1)
        prefix_sources.append(native)
        bindings.append(dict(family=family,CPU_proxy_sha256=digest(proxy.encode()),native_device_prefix_sha256=digest(native.encode())))
    host=(ROOT/'experiments/prepared/decoder_cooperative_boundary_host.cu').read_bytes()
    cuda=('\n'.join(prefix_sources)+'\n').encode()+host
    text=host.decode();decl=text[text.index('int64_t cb_launch('):text.index(' {\n#define CB_ARGS')]+ ';\n'
    cpp='#include <pybind11/pybind11.h>\n#include <cstdint>\n'+decl
    assert len(ast.parse(packaged).body)>100
    original=packaged.decode()
    assert original.count(OLD)==1
    amended=original.replace(OLD,NEW,1)
    addon=(ROOT/'experiments/prepared/decoder_cooperative_boundary_addon.py').read_bytes()
    section=('\n# Cooperative Decode boundary candidate: bounded O1/O8 integration.\n'
             '_CB_CUDA = _zsrc('+repr(base64.b64encode(zlib.compress(cuda,9)).decode())+')\n'
             '_CB_CPP = '+repr(cpp)+'\n').encode()+addon+b'\n'
    candidate=amended.encode()+section
    assert len(candidate)<1_000_000
    assert candidate[:-len(section)].decode().replace(NEW,OLD,1).encode()==packaged
    old_nodes=ast.parse(packaged).body;new_nodes=ast.parse(candidate).body
    assert len(new_nodes)>len(old_nodes)
    changed=[]
    for i,(a,b) in enumerate(zip(old_nodes,new_nodes)):
        if ast.dump(a)!=ast.dump(b):
            assert isinstance(a,ast.FunctionDef) and a.name=='kernel_fn'
            changed.append(i)
    assert len(changed)==1
    # New names cannot shadow an existing numerical implementation by accident.
    before={n.name for n in old_nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    added=[n.name for n in new_nodes[len(old_nodes):] if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    assert set(added)&before=={'prepare'}
    assert len(added)==len(set(added))
    output=ROOT/'results/decoder-cooperative-boundary-preflight-20261029'/args.revision
    freeze(output/'submission.py',candidate);freeze(output/'kernel.cu',cuda);freeze(output/'declaration.cpp',cpp.encode());freeze(output/'addon.py',addon)
    report=dict(source_sha256=digest(candidate),bytes=len(candidate),packaged_parent_sha256=PACKAGED,
        verified_baseline_sha256=D9,CUDA_sha256=digest(cuda),addon_sha256=digest(addon),
        host_sha256=digest(host),CPP_sha256=digest(cpp.encode()),device_prefix_bindings=bindings,
        modified_original_functions=['kernel_fn'],exact_inverse_restores_packaged_parent=True,
        all_other_original_module_nodes_unchanged=True,scope_tokens=[1,8],
        host_pybind_header_explicit=True,
        distinct_profile_markers=['cb_smallm2_gemv','cb_v18s_gemv'],
        unsupported_shapes_keep_original_paths=True,GPU_jobs=0,
        component_gate=dict(minimum_geometric_paired_gain=1.08,minimum_winning_pairs=12,total_pairs=16),
        whole_layer_gate=dict(minimum_affected_case_GM=1.005,minimum_control_normalized_weighted_track_gain=1.003,
                              minimum_each_affected_load_gain=.995,required_selected_families=[1,8]),
        probes='Both packed/BF16 formats; original BF16 projection output exact; fresh graph replay; nearby M2..7',
        remaining='Exact host/evaluator identity/full-suite/resource/new-path/performance admission')
    freeze(output/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
