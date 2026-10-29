"""Adapt the frozen first-trial guards without changing performance policy."""
from pathlib import Path
import ast
from preflight_decoder_cluster_norm import ROOT


def main():
    old=(ROOT/'scripts/decoder_cluster_norm_evidence.py').read_text()
    new=old
    replacements={
        "f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf":"5acf30a5af1f8a868c8f5fb1edd6073b8f5f7e8bfe0f2faaa4749fb29e56c09b",
        "c84debd7016155722eda211709914721b254f8383d091e9154a72f97807e87c6":"2a6336df622148216006121e1e60ff7d4e9b66c10938d28dcfae839406b96997",
        "decoder-cluster-norm-preflight-20261029":"decoder-cluster-norm-ordered-preflight-20261029",
        ".decoder-cluster-norm-measurements":".decoder-cluster-norm-ordered-measurements",
        "submission-v3":"submission-v1",
        "decoder_cluster_norm.cu":"decoder_cluster_norm_ordered.cu",
        "host=(ROOT/'experiments/prepared/decoder_cluster_norm_host.cu').read_bytes()":"host=(PACKET/'submission-v1/host.cu').read_bytes()",
        "49e61e6a225b9a9ec245aeeb41554ff8b717c06cf8053d387dc1ddd6eaa72653":"49bf73e68a8713a431f11d03d6f2f20e361562d4a105140962f8b6a11f269634",
        "'rank*PART+threadIdx.x' in device and 'PART=2048'":"'cluster.block_rank()*NT+threadIdx.x' in device and 'LOGICAL_THREADS=256'",
        "values[i],post":"values[pack][v],post",
        "r[2:4]==[0,72]":"r[2:4]==[0,16]",
        "r[6:]==[1,4,256]":"r[6:]==[1,4,64]",
        "dict(cases=20,changed_input_graph_replays=8)":"dict(cases=36,changed_input_graph_replays=8)",
        "okc_cluster_norm::boundary":"okc_cluster_norm_ordered::boundary",
    }
    for a,b in replacements.items():
        assert a in new,a
        new=new.replace(a,b)
    start=new.index('def validate_packet():');end=new.index('\n\ndef verify_trial(',start)
    validate='''def validate_packet():
    raw=(PACKET/'submission-v1/submission.py').read_bytes();verify_source(raw)
    cpu=read(PACKET/'CPU-model.json');compiler=read(PACKET/'v1/binding.json')
    assert cpu['parent_source_sha256']==compiler['parent_source_sha256']==PARENT
    assert cpu['native_CUDA_sha256']==compiler['CUDA_sha256']=='49bf73e68a8713a431f11d03d6f2f20e361562d4a105140962f8b6a11f269634'
    assert cpu['full_row_ownership_columns']==8192 and cpu['DSM_random_schedules']==256
    assert len(cpu['numerical_cases'])==60 and all(cpu['hazards_detected'].values())
    assert all(r['all_outputs_bitwise_equal_under_CPU_arithmetic'] for r in cpu['numerical_cases'])
    assert compiler['zero_stack_and_spills'] and compiler['offline_register_counts']==[56]*4
    binding=read(PACKET/'submission-v1/binding.json')
    assert binding['source_sha256']==TRIAL and binding['CUDA_sha256']==CUDA
    assert binding['numerical_and_paired_gate_AST_exact'] and binding['exact_inverse_restores_prior_host_and_addon']
    identity=read(PACKET/'evaluator-identity.json')
    assert identity['problem']=='muse_decode_layer'
    assert {r['file'] for r in identity['files']}=={'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md','nvfp4.py'}
    for r in identity['files']:
        assert r['exact'] and r['pinned_sha256']==r['fresh_sha256']
        assert hashlib.sha256((ROOT/'muse_decode_layer'/r['file']).read_bytes()).hexdigest()==r['pinned_sha256']
    admission=read(PACKET/'host-CPU-admission.json')
    assert admission['source_sha256']==TRIAL and admission['unsupported_fallbacks_checked']>=6
    guard=read(PACKET/'guard-CPU-admission.json')
    assert guard['source_sha256']==TRIAL and guard['positive_fixture_passed'] and len(guard['rejected_mutants'])>=20
    assert guard['guard_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return raw
'''
    new=new[:start]+validate+new[end:]
    compile(new,'ordered_evidence','exec')
    (ROOT/'scripts/decoder_cluster_norm_ordered_evidence.py').write_bytes(new.encode())
    admission=(ROOT/'experiments/prepared/audit_decoder_cluster_norm_admission.py').read_text()
    for a,b in {
        'from decoder_cluster_norm_evidence import':'from decoder_cluster_norm_ordered_evidence import',
        'submission-v3':'submission-v1',
        "(ROOT/'experiments/prepared/decoder_cluster_norm_addon.py')":"(PACKET/'submission-v1/addon.py')",
        '[k,32,0,72,32,132,1,4,256]':'[k,56,0,16,32,132,1,4,64]',
        'dict(cases=20,changed_input_graph_replays=8)':'dict(cases=36,changed_input_graph_replays=8)',
        'okc_cluster_norm::boundary':'okc_cluster_norm_ordered::boundary',
        'scripts/decoder_cluster_norm_evidence.py':'scripts/decoder_cluster_norm_ordered_evidence.py',
    }.items():
        assert a in admission,a
        admission=admission.replace(a,b)
    (ROOT/'experiments/prepared/audit_decoder_cluster_norm_ordered_admission.py').write_bytes(admission.encode())
    for name in ('measure_decoder_cluster_norm','verify_decoder_cluster_norm_evidence'):
        driver=(ROOT/'scripts'/f'{name}.py').read_text()
        driver=driver.replace('from decoder_cluster_norm_evidence import','from decoder_cluster_norm_ordered_evidence import')
        driver=driver.replace('Decode cluster-norm check','Decode ordered cluster-norm check')
        if name.startswith('measure_'):
            driver=driver.replace('assert raw == validate_packet()',
                "assert raw == validate_packet()\n        admission=read(ROOT/'results/decoder-cluster-norm-ordered-preflight-20261029/launch-admission.json')\n        assert admission['source_sha256']==sha and admission['GPU_admitted'],admission['reason']")
        (ROOT/'scripts'/f'{name.replace("cluster_norm","cluster_norm_ordered")}.py').write_bytes(driver.encode())
    print('Created separate source-bound ordered guards/drivers; original policy unchanged.')


if __name__=='__main__':main()
