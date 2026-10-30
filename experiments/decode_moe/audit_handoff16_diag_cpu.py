"""Exact source binding and prepare-protocol audit; no GPU operations."""

import ast
import base64
import copy
import contextlib
import hashlib
import io
import json
import zlib

import torch

from build_handoff16_diag import HERE, ROOT, OUT, SPIN, QUERY
from audit_pdln_interface_cpu import extract


def funcs(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def main():
    raw=(HERE/'dense_handoff16_diag.py').read_bytes();text=raw.decode();parent=(ROOT/'best/verified.py').read_bytes()
    binding=json.loads((OUT/'source-binding.json').read_text());assert hashlib.sha256(raw).hexdigest()==binding['source_sha256']and raw.startswith(parent)
    a,b=funcs(parent),funcs(text)
    assert ast.dump(a['kernel_fn'],include_attributes=False)==ast.dump(b['kernel_fn'],include_attributes=False)
    assert set(a)&(set(b)-set(a)|{'prepare'})=={'prepare'}
    # Check every appended definition for a collision, including redefinitions.
    appended=ast.parse(raw[len(parent):].decode())
    added_names={n.name for n in appended.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert set(a)&added_names=={'prepare'},set(a)&added_names
    refs=[('decoder_pdli.py',['_fpr_bfp_sk2_tileunits','_fpr_bfp_sk2_compress','_fpr_bfp_sk2_stream',
             '_fpr_bfp_sk2_table','_fpr_bfp__sk2_tokr','_fpr_bfp_fused_pooler_router','_fpr_prepare','_pdli_equal','_pdli_probe']),
          ('decoder_pdl_chain_draft.py',['_p16_project','_p16_unpool','_p16_pool_route','_p16_plans']),
          ('decoder_pdln_draft.py',['pdln_derive','_pdln_build','_pdln_boundary'])]
    for path,names in refs:
        f=funcs((HERE/path).read_text(encoding='utf-8'))
        for name in names:assert ast.dump(f[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
    # The extra FPR query must not change its compilation flags or launcher.
    getter=copy.deepcopy(b['_fpr_get_ext'])
    call=next(n for n in ast.walk(getter)if isinstance(n,ast.Call)and getattr(n.func,'id','')=='_fpr_load_inline')
    keys={n.arg:n for n in call.keywords}
    assert keys['name'].value.left.value.startswith('okc_d16_fpr_')
    keys['name'].value.left.value='okc_fpr_f68f77c0d29cc56b_'
    assert ast.literal_eval(keys['functions'].value)==['sk2f','sk2_set_pdl','d16_fpr_query']
    keys['functions'].value.elts.pop()
    original_getter=funcs((HERE/'decoder_pdli.py').read_text(encoding='utf-8'))['_fpr_get_ext']
    assert ast.dump(getter,include_attributes=False)==ast.dump(original_getter,include_attributes=False)
    original=(OUT/'fpr-original-prefix.cu').read_bytes();queried=(OUT/'fpr-with-query.cu').read_bytes()
    assert queried==original+QUERY.encode()
    assert b'bool PK, bool FUSE' in original and b'int* __restrict__ rcnt, int pdl)' in original
    assert QUERY.count('__nv_bfloat16,false,true>')==2 and '__nv_bfloat16,false,false>'not in QUERY
    assert (OUT/'projection-with-spin.cu').read_bytes()==(OUT/'projection-original-prefix.cu').read_bytes()+SPIN.encode()
    protocol=[]
    for layer,m in [('o',16),('unpooler',32)]:
        def product(x):return x.float().sum(1,keepdim=True).to(torch.bfloat16).expand(m,8192).clone()
        def spin(src,dst):
            dst.copy_(src);dst[:,0].neg_()
            assert torch.equal(dst[:,0].view(torch.int16),torch.bitwise_xor(src[:,0].view(torch.int16),-32768))
        mutations=[]
        def probe(fns,mutate,**kw):
            for seed in range(3):
                mutate(seed);results=[fn()for fn in fns.values()]
                assert all(torch.equal(x,y)for x,y in zip(*results))
                if seed==1:assert torch.count_nonzero(results[0][0])==0
                mutations.append(seed)
            return True
        def timing(fns,dev,reps):
            assert reps==25
            for fn in fns.values():assert torch.count_nonzero(fn()[0])>0
            return dict(parent=10.,early=8.)
        env=dict(torch=torch,_okh_budget=lambda *a:True,_pdli_probe=probe,_okh_time=timing,
            _d16_spin=spin,_p16_project=lambda x,d:product(x),_p16_unpool=lambda x,d:product(x),
            _d16_reference=lambda x,l,d:product(x))
        extract(HERE/'handoff16_diag_runtime.py',{'_d16_times','_d16_projection_probe'},env)
        key=m if layer=='o'else'u'+str(m);diag={'p16':{'plans':{key:{}}},'achita_dense':{}}
        report=dict(projections=[],skipped=[],budget_stop=False)
        env['_d16_projection_probe'](diag,m,layer,torch.Generator().manual_seed(m),'cpu',0,report)
        row=report['projections'][0];assert row['byte_equal']and row['handoff_gate']and mutations==[0,1,2]
        assert [x['order']for x in row['orders']]==[['parent','early'],['early','parent']]
        protocol.append(dict(layer=layer,m=m,fresh_copy_late_sign_and_opposite_orders=True))
    # The real-chain protocol has four tensor outputs (None router excluded) and
    # fixed parent/late/early arms. All work is mocked CPU metadata/data flow.
    def norm(branch,stream,*a,**kw):return branch.clone(),stream.clone(),None
    def fpr(x,state):return x[:,:4096].clone(),x[:,:256].float().clone()
    def probe3(fns,mutate,**kw):
        assert list(fns)==['parent','late','early']
        for seed in range(3):
            mutate(seed);outputs=[fn()for fn in fns.values()];assert all(len(o)==4for o in outputs)
            assert all(torch.equal(x,y)for o in outputs[1:]for x,y in zip(outputs[0],o))
        return True
    def time3(fns,dev,reps):
        assert reps==25
        return dict(parent=20.,late=19.,early=17.)
    from types import SimpleNamespace
    env=dict(torch=torch,_okh_budget=lambda *a:True,_D16_PUBLIC=SimpleNamespace(fused_norm_residual=norm),
        _pdln_boundary=lambda s,b,d:norm(b,s),_fpr_bfp_fused_pooler_router=fpr,
        _p16_pool_route=lambda x,d:fpr(x,d['fpr']),_pdli_probe=probe3,_okh_time=time3)
    extract(HERE/'handoff16_diag_runtime.py',{'_d16_times','_d16_norm_fpr_probe'},env)
    report=dict(norm_fpr=[],skipped=[],budget_stop=False)
    env['_d16_norm_fpr_probe']({'fpr':{}},16,torch.Generator().manual_seed(4),'cpu',0,report)
    row=report['norm_fpr'][0];assert row['byte_equal']and row['handoff_gate']
    assert [x['order']for x in row['orders']]==[['parent','late','early'],['early','late','parent']]
    # A parent win caused mostly by an old incoming-PDL flag does not admit the
    # new norm signal when its own contribution misses the absolute floor.
    env['_okh_time']=lambda *a,**kw:dict(parent=20.,late=18.,early=17.)
    rejected={}
    env['_d16_times']('causal-floor',{'parent':lambda:None,'late':lambda:None,'early':lambda:None},'cpu',0,rejected)
    assert not rejected['handoff_gate'] and all(not x['causal_signal_pass']for x in rejected['orders'])
    # Bound a worst-case terminal report with six rows and diagnostic errors.
    stress=dict(report,projections=[dict(protocol=protocol,orders=row['orders'])for _ in range(4)],errors=['x'*2200])
    payload=base64.b64encode(zlib.compress(json.dumps(stress).encode(),9)).decode();assert len(payload)<9000
    tail=('x'*50000+'\nD16_SUMMARY_Z64 '+payload+'\nD16_DONE 250\n'+'p'*6000)[-16000:]
    line=next(x.split(' ',1)[1]for x in tail.splitlines()if x.startswith('D16_SUMMARY_Z64 '))
    assert json.loads(zlib.decompress(base64.b64decode(line)))==stress
    # A failed independent compiler must not discard the other unit evidence.
    readiness=[]
    for missing in ('norm','projection'):
        state_marker={}
        fake_cuda=SimpleNamespace(get_device_properties=lambda d:SimpleNamespace(major=9,multi_processor_count=132),
                                  synchronize=lambda:None)
        fake_torch=SimpleNamespace(cuda=fake_cuda,Generator=torch.Generator)
        def plans(diag):diag['p16']['plans'].update({16:{},32:{},'u16':{},'u32':{}})
        def projection(diag,m,layer,g,d,t,report):
            key=m if layer=='o'else'u'+str(m)
            if key in diag['p16']['plans']:report['projections'].append(dict(m=m,layer=layer))
        def pair(diag,m,g,d,t,report):
            if diag['fpr']is not None:report['norm_fpr'].append(dict(m=m))
        norm_ext=SimpleNamespace(query=lambda i:[175,0,64,1][i])
        fpr_ext=SimpleNamespace(d16_fpr_query=lambda m,i:[85 if m==16 else 126,0,1024,416 if m==16 else 384][i])
        env=dict(torch=fake_torch,_okh_clock=SimpleNamespace(perf_counter=lambda:0.),
            _d16_parent_prepare=lambda w:state_marker,_d16_threads=[SimpleNamespace(join=lambda:None)]*4,
            _P16_EXT=None if missing=='projection'else object(),_D16_BUILD_ERROR='mock missing projection',
            _D16_PUBLIC=None if missing=='norm'else object(),_D16_PUBLIC_ERROR='mock missing norm',
            _PDLN_EXT=None if missing=='norm'else norm_ext,_PDLN_ERROR=None,_D16_FPR_ERROR=None,
            _okh_budget=lambda *a:True,_fpr_prepare=lambda w:{},_p16_plans=plans,
            _fpr_get_ext=lambda:fpr_ext,_d16_projection_probe=projection,_d16_norm_fpr_probe=pair)
        extract(HERE/'handoff16_diag_runtime.py',{'prepare'},env)
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):returned=env['prepare']({'router':torch.empty(1)})
        assert returned is state_marker
        encoded=next(x.split(' ',1)[1]for x in capture.getvalue().splitlines()if x.startswith('D16_SUMMARY_Z64 '))
        observed=json.loads(zlib.decompress(base64.b64decode(encoded)))
        assert observed['errors']
        assert len(observed['projections'])==(4 if missing=='norm'else 0)
        assert len(observed['norm_fpr'])==(0 if missing=='norm'else 2)
        readiness.append(dict(missing=missing,independent_units_preserved=True,returns_original_state=True))
    result=dict(GPU_jobs=0,source_sha256=binding['source_sha256'],Dense_prefix_and_public_kernel_unchanged=True,
        no_appended_function_collisions_except_prepare=True,exact_component_AST=True,
        FPR_original_kernel_prefix_exact=True,FPR_host_builder_inverse_exact=True,
        FPR_query_points_to_FUSE_true_runtime_PDL_kernel=True,
        projection_prefix_exact=True,projection_protocol=protocol,norm_FPR_three_arms_pass=True,
        old_flag_only_gain_rejected_by_causal_floor=True,
        None_router_not_returned_to_byte_probe=True,terminal_report_retention_pass=True,
        independent_build_readiness_cases=readiness,
        limitation='CPU source and mocked protocol audit. No SDK/host compilation, H200 correctness, timing or whole-layer result.')
    (OUT/'cpu-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
