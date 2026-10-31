"""Single-file nc1 payload retaining the original support and launch paths."""
import ast
import hashlib
import json

from attention_nc1_source import OUT, ROOT, derive
from harness import append_harness

HERE = ROOT / 'experiments/decode_moe'


def main():
    parent, _, cuda, cpp = derive()
    assert (OUT / 'packaged.cu').read_bytes() == cuda.encode()
    resources = json.loads((OUT / 'packaged-layout-cpu-audit.json').read_text())
    assert resources['packaged_CUDA_sha256'] == hashlib.sha256(cuda.encode()).hexdigest()
    assert resources['zero_stack_spills']
    text = parent.decode()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == '_build')
    build = ast.get_source_segment(text, node)
    for old, new in [('_build(', '_nc1_build_native('), ('CPP_SRC', 'CPP_NC1_SRC'),
                     ('CUDA_SRC', 'CUDA_NC1_SRC'), ('functions=["run"]', 'functions=["run_nc1", "nc1_resources"]'),
                     ('okc_attn_v298', 'okc_attn_nc1_v1_' + hashlib.sha256(cuda.encode()).hexdigest()[:12])]:
        assert old in build
        build = build.replace(old, new)
    contract = (HERE / 'attention_nc1_contract.py').read_text()
    nodes = [n for n in ast.parse(contract).body if isinstance(n, ast.FunctionDef)]
    pieces = '\n\n'.join(ast.get_source_segment(contract, n) for n in nodes)
    pieces = pieces.replace('def family(', 'def _nc1_family(').replace('def run_family(', 'def _nc1_run_family(')
    pieces = pieces.replace('return family(', 'return _nc1_family(')
    runtime = (HERE / 'attention_nc1_runtime.py').read_text()
    payload = (append_harness(text) + '\nCUDA_NC1_SRC = ' + repr(cuda) + '\nCPP_NC1_SRC = ' + repr(cpp)
               + '\n\n' + build + '\n\n' + pieces + '\n\n' + runtime + '\n').encode()
    ast.parse(payload)
    assert payload.startswith(parent.rstrip(b'\n')) and len(payload) < 1_000_000
    (HERE / 'attention_nc1.py').write_bytes(payload)
    record = dict(source_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                  CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                  parent_sha256=hashlib.sha256(parent).hexdigest(),
                  marker='attn_kernel_nc1', parent_prefix_exact=True,
                  calibration='First eager setup;fixed independent synthetic inputs;no caller activation/length values cached',
                  numerical_gate='Exact output/cache bytes and unchanged-parent repeated controls;no MoE variance exception',
                  timing_gate='Each family needs8strict>3% comparisons,two arm orders/two cohorts/two distributions;nearby<=2%regression',
                  scope='Only metadata-guarded L1 nc1 flat/SPEC-ONE,flags0;all other native and Torch fallbacks original',
                  GPU_jobs=0, admitted_for_GPU=False,
                  pending='CPU runtime/interface audit;actual host compilation,resources,bytes/replay and timing on H200')
    (OUT / 'payload-binding.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
