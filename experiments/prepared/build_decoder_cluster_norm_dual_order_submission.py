"""Bind the one dual-order repair to unchanged probes and payoff policy."""
import ast
import base64
import hashlib
import json
import zlib
from preflight_decoder_cluster_norm import ROOT,BASE,freeze

PACKAGED='307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072'
OLD=ROOT/'results/decoder-cluster-norm-ordered-preflight-20261029/submission-v1'
OUT=ROOT/'results/decoder-cluster-norm-dual-order-preflight-20261029/submission-v1'


def main():
    prefix=(ROOT/'results/decoder-unused-rope-prune-20261029/submission.py').read_bytes()
    assert hashlib.sha256(prefix).hexdigest()==PACKAGED
    assert hashlib.sha256((ROOT/'best/decoder_verified.py').read_bytes()).hexdigest()==BASE
    native=(ROOT/'experiments/prepared/decoder_cluster_norm_dual_order.cu').read_bytes()
    assert hashlib.sha256(native).hexdigest()=='5d189e9a2f7c5b1396c82cf60dd893886299dff5f857c23c0d93da316e1a5347'
    host=(OLD/'host.cu').read_bytes();assert host.count(b'okc_cluster_norm_ordered::')==3
    host=host.replace(b'okc_cluster_norm_ordered::',b'okc_cluster_norm_dual_order::')
    addon=(OLD/'addon.py').read_bytes()
    original=ast.parse((ROOT/'experiments/prepared/decoder_cluster_norm_addon.py').read_bytes());updated=ast.parse(addon)
    for name in ('_cn_error','_cn_paired'):
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in updated.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
    cuda=native+b'\n'+host;cuda_sha=hashlib.sha256(cuda).hexdigest()
    constants={n.targets[0].id:n.value for n in ast.parse((OLD/'submission.py').read_bytes()).body
               if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    cpp=ast.literal_eval(constants['_CN_CPP'])
    a=host.decode().index('void cn_launch(');b=host.decode().index(' {\n#define CN_ARGS',a)
    assert host.decode()[a:b]+';' in cpp
    section=('\n# Dual scalar/Triton reduction-order repair; all four modes remain in scope.\n'
        '_CN_CUDA = _zsrc('+repr(base64.b64encode(zlib.compress(cuda,9)).decode())+')\n'
        '_CN_CPP = '+repr(cpp)+'\n_CN_SHA = '+repr(cuda_sha)+'\n').encode()+addon+b'\n'
    candidate=prefix+section
    old_tree=ast.parse(prefix);tree=ast.parse(candidate)
    assert ast.dump(ast.Module(body=tree.body[:len(old_tree.body)],type_ignores=[]))==ast.dump(old_tree)
    report=dict(parent_source_sha256=BASE,packaged_parent_sha256=PACKAGED,
        source_sha256=hashlib.sha256(candidate).hexdigest(),CUDA_sha256=cuda_sha,
        native_CUDA_sha256=hashlib.sha256(native).hexdigest(),bytes=len(candidate),
        rejected_GPU_source_sha256='f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf',
        retry_reason='Source/finite evidence establishes changed reduction order. Repair scalar order and separately generated tiny-Triton order; preserve fixed gates and original complete M1..32 boundary scope.',
        exact_failed_GPU_input_unavailable=True,evaluator_Triton_version_unknown=True,
        addon_exact_from_ordered_trial=True,numerical_and_paired_gate_AST_exact=True,
        numerical_probes=36,changed_input_graph_replays=8,packaged_parent_exact_prefix=True,
        all_four_semantic_modes_preserved=True,CPU_only=True,GPU_jobs=0,
        required_native=dict(registers_at_most=64,local_bytes=0,static_shared_bytes=16,
            active_clusters_at_least=32,SMs_at_least=132,CTAs_per_cluster=4,threads_per_CTA=64))
    for name,data in (('submission.py',candidate),('kernel.cu',cuda),('host.cu',host),('addon.py',addon),
                      ('binding.json',(json.dumps(report,indent=2)+'\n').encode())):freeze(OUT/name,data)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
