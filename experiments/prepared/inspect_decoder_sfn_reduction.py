"""Compile the exact retained tiny-token FFN with an explicit CPU-only target.

Run with WSL Python. Requires the verified official wheel extracted under
.tools/triton-offline-3.9.0/site; never installs global packages or loads CUDA.
The evaluator's actual Triton version is unknown: this is generated-code
evidence for the named compiler, not a claim of evaluator binary identity.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
WORK=ROOT/'.tools/triton-offline-3.9.0'
sys.path.insert(0,str(WORK/'site'))
os.environ['TRITON_CACHE_DIR']=str(WORK/'cache')
os.environ['TRITON_DUMP_DIR']=str(WORK/'dump')
os.environ['TRITON_OVERRIDE_DIR']=str(WORK/'override')


def main():
    raw=(ROOT/'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='_sfn_kernel')
    kernel=ast.get_source_segment(raw.decode(),node)
    module=WORK/'retained_sfn.py'
    module.write_text('import triton\nimport triton.language as _sfn_tl\n@triton.jit\n'+kernel+'\n')
    spec=importlib.util.spec_from_file_location('retained_sfn',module)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    assert triton.__version__=='3.9.0'
    signature={k:('*bf16' if k in ('Branch','PostWeight','PreWeight','Output') else '*fp32')
               for k in ('Branch','Stream','PostWeight','PreWeight','Alpha','Beta','Output','NewStream')}
    const=dict(BranchStride=1024,StreamStride=8192,H=8192,Slice=1024,PostEps=1e-8,PreEps=1e-5)
    compiled=triton.compile(ASTSource(loaded._sfn_kernel,signature,constexprs=const),
        target=GPUTarget('cuda',90,32),options=dict(num_warps=8,enable_fp_fusion=False))
    out=ROOT/'results/decoder-sfn-offline-reduction-20261029'
    out.mkdir(exist_ok=True)
    for kind,value in compiled.asm.items():
        if kind not in ('ttir','ttgir','llir','ptx','cubin'):continue
        data=value if isinstance(value,bytes) else value.encode()
        path=out/f'retained_sfn.{kind}'
        assert not path.exists() or path.read_bytes()==data
        path.write_bytes(data)
    metadata=dict(source_sha256=hashlib.sha256(raw).hexdigest(),
        kernel_AST_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest(),
        kernel_source_sha256=hashlib.sha256(kernel.encode()).hexdigest(),
        compiler_host_Python=sys.version,
        compiler_version=triton.__version__,target='cuda/sm90/warp32',
        options=dict(num_warps=8,enable_fp_fusion=False),constants=const,
        CPU_only=True,GPU_jobs=0,evaluator_compiler_version_unknown=True,
        metadata=str(compiled.metadata))
    (out/'compile.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
