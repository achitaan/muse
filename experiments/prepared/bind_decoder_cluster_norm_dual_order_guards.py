"""Keep complete-layer gates fixed while binding the dual-order source."""
from preflight_decoder_cluster_norm import ROOT


def main():
    source=(ROOT/'scripts/decoder_cluster_norm_ordered_evidence.py').read_text()
    replacements={
        '5acf30a5af1f8a868c8f5fb1edd6073b8f5f7e8bfe0f2faaa4749fb29e56c09b':'1c23cdc6dbcf3583cfcae023bef6281059fd1db04347f00cfcf28b95268dcfff',
        '2a6336df622148216006121e1e60ff7d4e9b66c10938d28dcfae839406b96997':'65b393c4885037546cacc0eacdd42a505879ff2efcf62fdf43ee788e887c3a83',
        '49bf73e68a8713a431f11d03d6f2f20e361562d4a105140962f8b6a11f269634':'5d189e9a2f7c5b1396c82cf60dd893886299dff5f857c23c0d93da316e1a5347',
        'decoder-cluster-norm-ordered':'decoder-cluster-norm-dual-order',
        'decoder_cluster_norm_ordered.cu':'decoder_cluster_norm_dual_order.cu',
        'okc_cluster_norm_ordered::':'okc_cluster_norm_dual_order::',
        "compiler['CUDA_sha256']":"compiler['native_CUDA_sha256']",
        "binding['exact_inverse_restores_prior_host_and_addon']":"binding['addon_exact_from_ordered_trial']",
    }
    for a,b in replacements.items():
        assert a in source,a
        source=source.replace(a,b)
    source=source.replace("PACKET/'guard-CPU-admission.json'","PACKET/'guard-CPU-admission-v2.json'")
    needle="    assert compiler['zero_stack_and_spills']"
    extra="""    assert cpu['post_and_pre_trees_balanced_adjacent_32']
    assert cpu['generated_Triton_tree_proved_for_logical_threads']==[0,7,31,65,127,255]
    assert cpu['generated_PTX_sha256']==hashlib.sha256((ROOT/'results/decoder-sfn-offline-reduction-20261029/retained_sfn.ptx').read_bytes()).hexdigest()
    compiled=read(ROOT/'results/decoder-sfn-offline-reduction-20261029/compile.json')
    assert compiled['source_sha256']==PARENT and compiled['CPU_only'] and compiled['GPU_jobs']==0
    retained=(ROOT/'best/decoder_verified.py').read_bytes().decode()
    tiny=next(n for n in ast.parse(retained).body if isinstance(n,ast.FunctionDef) and n.name=='_sfn_kernel')
    assert hashlib.sha256(ast.get_source_segment(retained,tiny).encode()).hexdigest()==compiled['kernel_source_sha256']
"""
    assert needle in source;source=source.replace(needle,extra+needle)
    needle='    return dict(source_sha256=TRIAL,exact_packaged_prefix=True'
    extra="""    assert 'SLICE ? logical_thread+(pack*VEC+v)*LOGICAL_THREADS' in device
    assert 'if constexpr (SLICE) ss=tiny_tree<0,PACKS*VEC>(values);' in device
    assert 'if constexpr (SLICE) pre_ss=tiny_tree<0,PACKS*VEC>(values);' in device
    assert '__fmul_rn(v,v)' in device and '__fadd_rn(tiny_tree<BEGIN,COUNT/2>' in device
"""
    assert needle in source;source=source.replace(needle,extra+needle)
    compile(source,'dual_order_evidence','exec')
    (ROOT/'scripts/decoder_cluster_norm_dual_order_evidence.py').write_bytes(source.encode())
    admission=(ROOT/'experiments/prepared/audit_decoder_cluster_norm_ordered_admission.py').read_text()
    admission=admission.replace('decoder_cluster_norm_ordered_evidence','decoder_cluster_norm_dual_order_evidence')
    admission=admission.replace('okc_cluster_norm_ordered::','okc_cluster_norm_dual_order::')
    admission=admission.replace("PACKET/'guard-CPU-admission.json'","PACKET/'guard-CPU-admission-v2.json'")
    (ROOT/'experiments/prepared/audit_decoder_cluster_norm_dual_order_admission.py').write_bytes(admission.encode())
    for name in ('measure_decoder_cluster_norm_ordered','verify_decoder_cluster_norm_ordered_evidence'):
        driver=(ROOT/'scripts'/f'{name}.py').read_text()
        driver=driver.replace('decoder_cluster_norm_ordered_evidence','decoder_cluster_norm_dual_order_evidence')
        driver=driver.replace('decoder-cluster-norm-ordered','decoder-cluster-norm-dual-order')
        driver=driver.replace('Decode ordered cluster-norm check','Decode dual-order cluster-norm check')
        (ROOT/'scripts'/f'{name.replace("norm_ordered","norm_dual_order")}.py').write_bytes(driver.encode())
    print('Separate dual-order source/result guards and driver bound; policy unchanged.')


if __name__=='__main__':main()
