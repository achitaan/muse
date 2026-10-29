"""Bind one diagnosed reduction-order repair; never weaken numerical gates."""
import ast
import base64
import hashlib
import json
import zlib

from preflight_decoder_cluster_norm import ROOT,BASE,freeze

PACKAGED='307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072'
OLD_ADDON='25ea5a6a75b50dd5eaea3d0595d92913621636240d319de9c018d0a98f3ea1cd'
OUT=ROOT/'results/decoder-cluster-norm-ordered-preflight-20261029/submission-v1'


def main():
    packaged=(ROOT/'results/decoder-unused-rope-prune-20261029/submission.py').read_bytes()
    assert hashlib.sha256(packaged).hexdigest()==PACKAGED
    assert hashlib.sha256((ROOT/'best/decoder_verified.py').read_bytes()).hexdigest()==BASE
    native=(ROOT/'experiments/prepared/decoder_cluster_norm_ordered.cu').read_bytes()
    assert hashlib.sha256(native).hexdigest()=='49bf73e68a8713a431f11d03d6f2f20e361562d4a105140962f8b6a11f269634'
    host_old=(ROOT/'experiments/prepared/decoder_cluster_norm_host.cu').read_text()
    host=host_old;host_edits=[]
    for old,new,count in (('okc_cluster_norm::','okc_cluster_norm_ordered::',3),
                          ('dim3(256,1,1)','dim3(64,1,1)',2),
                          ('supported,4,256','supported,4,64',1)):
        assert host.count(old)==count,(old,host.count(old));host=host.replace(old,new)
        host_edits.append(dict(old=old,new=new,count=count))
    addon_old=(ROOT/'experiments/prepared/decoder_cluster_norm_addon.py').read_bytes()
    assert hashlib.sha256(addon_old).hexdigest()==OLD_ADDON
    addon=addon_old.decode();edits=[]
    def edit(old,new,count=1):
        nonlocal addon
        assert addon.count(old)==count,(old[:80],addon.count(old),count)
        addon=addon.replace(old,new);edits.append(dict(old=old,new=new,count=count))
    edit('r[3]==72','r[3]==16')
    edit('r[6:]==[1,4,256]','r[6:]==[1,4,64]')
    # Record exact failing context. This does not change comparison tolerances.
    anchor='def _cn_paired(old,new,dev,seed):'
    comparison='''def _cn_compare(new,old,m,kind,fixture):
    try:
        return _cn_error(new(),old())
    except RuntimeError as error:
        print('CN_NUMERICAL_FAILURE '+_cn_json.dumps(dict(m=m,kind=kind,fixture=fixture,error=str(error))),flush=True)
        raise RuntimeError(f'cluster norm comparison failed M={m},kind={kind},fixture={fixture}: {error}') from error


'''
    edit(anchor,comparison+anchor)
    old="            errors=_cn_error(new(),old());probes+=1\n"
    new="""            errors=_cn_compare(new,old,m,kind,'normal');probes+=1
            if m in (1,8,16,32):
                saved_branch=b.clone();saved_residual=residual.clone()
                b.copy_((saved_branch.float()*1e-4).bfloat16())
                residual.copy_(saved_residual*1e-4)
                _cn_compare(new,old,m,kind,'small_values');probes+=1
                b.zero_();residual.copy_(saved_residual)
                _cn_compare(new,old,m,kind,'zero_branch');probes+=1
                b.copy_(saved_branch);residual.copy_(saved_residual)
"""
    edit(old,new)
    edit("graph.replay();stream.synchronize();_cn_error(captured,old());replays+=1",
         "graph.replay();stream.synchronize();_cn_compare(lambda:captured,old,m,kind,'graph_replay_'+str(repeat));replays+=1")
    # Prove the complete numerical and performance gate functions are identical.
    before=ast.parse(addon_old);after=ast.parse(addon)
    for name in ('_cn_error','_cn_paired'):
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    recovered=addon
    for e in reversed(edits):
        assert recovered.count(e['new'])==e['count'];recovered=recovered.replace(e['new'],e['old'])
    assert recovered.encode()==addon_old
    recovered_host=host
    for e in reversed(host_edits):
        assert recovered_host.count(e['new'])==e['count'];recovered_host=recovered_host.replace(e['new'],e['old'])
    assert recovered_host==host_old
    cuda=native+b'\n'+host.encode();cuda_sha=hashlib.sha256(cuda).hexdigest()
    a=host.index('void cn_launch(');b=host.index(' {\n#define CN_ARGS',a)
    cpp='#include <pybind11/pybind11.h>\n#include <cstdint>\n#include <vector>\nstd::vector<int64_t> cn_info(int kind);\n'+host[a:b]+';\n'
    section=('\n# Source-order repair of the rejected cluster boundary; fixed gates unchanged.\n'
        '_CN_CUDA = _zsrc('+repr(base64.b64encode(zlib.compress(cuda,9)).decode())+')\n'
        '_CN_CPP = '+repr(cpp)+'\n_CN_SHA = '+repr(cuda_sha)+'\n').encode()+addon.encode()+b'\n'
    candidate=packaged+section
    tree,updated=ast.parse(packaged),ast.parse(candidate)
    assert ast.dump(ast.Module(body=updated.body[:len(tree.body)],type_ignores=[]))==ast.dump(tree)
    report=dict(parent_source_sha256=BASE,packaged_parent_sha256=PACKAGED,
        source_sha256=hashlib.sha256(candidate).hexdigest(),CUDA_sha256=cuda_sha,
        native_CUDA_sha256=hashlib.sha256(native).hexdigest(),bytes=len(candidate),
        repaired_source_sha256='f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf',
        retry_reason='Pinned scalar source and finite CPU reproductions establish a different FP32 reduction order. Preserve its original256logical-thread order rather than relaxing the failed gate.',
        actual_failed_GPU_input_unavailable=True,numerical_and_paired_gate_AST_exact=True,
        numerical_probes=36,changed_input_graph_replays=8,host_edits=host_edits,addon_edits=edits,
        exact_inverse_restores_prior_host_and_addon=True,packaged_parent_exact_prefix=True,
        original_projection_and_all_other_timed_kernels_unchanged=True,CPU_only=True,GPU_jobs=0,
        required_native=dict(registers_at_most=64,local_bytes=0,static_shared_bytes=16,active_clusters_at_least=32,SMs_at_least=132,CTAs_per_cluster=4,threads_per_CTA=64),
        readiness='Requires independent source/model/compiler/host/guard admission and current evaluator identity before one paced check')
    freeze(OUT/'submission.py',candidate);freeze(OUT/'kernel.cu',cuda)
    freeze(OUT/'host.cu',host.encode());freeze(OUT/'addon.py',addon.encode())
    freeze(OUT/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k not in ('addon_edits','host_edits')},indent=2))


if __name__=='__main__':main()
