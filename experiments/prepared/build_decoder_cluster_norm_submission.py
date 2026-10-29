"""Bind the fixed cluster-norm prototype to the source-preserving Decode pack."""
import ast
import argparse
import base64
import hashlib
import json
import zlib

from preflight_decoder_cluster_norm import ROOT,BASE,freeze

PACKAGED='307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='v3');args=parser.parse_args()
    assert args.revision in ('v1','v2','v3')
    raw=(ROOT/'results/decoder-unused-rope-prune-20261029/submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PACKAGED
    assert hashlib.sha256((ROOT/'best/decoder_verified.py').read_bytes()).hexdigest()==BASE
    native=(ROOT/'experiments/prepared/decoder_cluster_norm.cu').read_bytes()
    assert hashlib.sha256(native).hexdigest()=='49e61e6a225b9a9ec245aeeb41554ff8b717c06cf8053d387dc1ddd6eaa72653'
    host=(ROOT/'experiments/prepared/decoder_cluster_norm_host.cu').read_bytes()
    cuda=native+b'\n'+host
    text=host.decode()
    a=text.index('void cn_launch(');b=text.index(' {\n#define CN_ARGS',a)
    cpp='#include <pybind11/pybind11.h>\n#include <cstdint>\n#include <vector>\nstd::vector<int64_t> cn_info(int kind);\n'+text[a:b]+';\n'
    addon=(ROOT/'experiments/prepared/decoder_cluster_norm_addon.py').read_bytes()
    sha=hashlib.sha256(cuda).hexdigest()
    section=('\n# Fixed cluster4 residual/norm candidate; all retained sources above stay intact.\n'
        '_CN_CUDA = _zsrc('+repr(base64.b64encode(zlib.compress(cuda,9)).decode())+')\n'
        '_CN_CPP = '+repr(cpp)+'\n_CN_SHA = '+repr(sha)+'\n').encode()+addon+b'\n'
    candidate=raw+section
    oldtree,newtree=ast.parse(raw),ast.parse(candidate)
    assert ast.dump(ast.Module(body=newtree.body[:len(oldtree.body)],type_ignores=[]))==ast.dump(oldtree)
    assert candidate.startswith(raw)
    out=ROOT/'results/decoder-cluster-norm-preflight-20261029'/('submission-'+args.revision)
    report=dict(parent_source_sha256=BASE,packaged_parent_sha256=PACKAGED,
        source_sha256=hashlib.sha256(candidate).hexdigest(),CUDA_sha256=sha,bytes=len(candidate),
        standalone_device_CUDA_sha256=hashlib.sha256(native).hexdigest(),
        addon_sha256=hashlib.sha256(addon).hexdigest(),packaged_parent_exact_prefix=True,
        original_projections_and_dense_geometry_unchanged=True,
        candidate_scope='Only attention/FFN boundaries at M1..32; sliced FFN only M1..8; one shared admission decision for the whole family',
        causal_prepare_gate=dict(each_combined_boundary_speedup_at_least=1.03,each_combined_net_saving_ns_at_least=1000,
            no_individual_boundary_ratio_above=1.02,canonical_tokens=[1,8,16,32]),
        native_requirements=dict(cluster_CTAs=4,threads_per_CTA=256,local_bytes=0,static_shared_bytes=72,
            register_ceiling=64,active_clusters_at_least=32,SMs_at_least=132),
        numerical_probes=20,changed_input_graph_replays=8,
        all_untouched_fallbacks_retained=True,GPU_jobs=0,
        full_layer_gate='13tests/12benchmarks/integrity, real new paths in all eight small cases, controls unchanged, material affected/control weighted gain',
        not_admitted_for_GPU_until='independent source/precision/packet proof, source-bound fault-injected result guard, current evaluator identity and clean cached baseline controller')
    freeze(out/'submission.py',candidate);freeze(out/'kernel.cu',cuda);freeze(out/'addon.py',addon)
    freeze(out/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
