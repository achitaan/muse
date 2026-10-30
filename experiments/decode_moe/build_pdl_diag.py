"""Build the source-bound Step 1 PDL diagnostic. Production remains byte-for-byte R2f."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

from harness import append_harness
from pdl_source import twin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / 'results/decode-moe-plan-20261030'

SPIN = r'''
__global__ void pdl_spin(unsigned long long ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x) return;
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
int pdl_spin_launch(int64_t ns, int64_t stream) {
  pdl_spin<<<1, 32, 0, (cudaStream_t)stream>>>((unsigned long long)ns);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}
'''
O_CPP = ('\n#include <pybind11/pybind11.h>\n#include <stdint.h>\n'
         'int pdl_o_launch(int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, int, int, int64_t, int64_t, int64_t);\n'
         'int pdl_spin_launch(int64_t, int64_t);\n')


def source_specs(raw):
    tree = ast.parse(raw)
    nodes = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            nodes.setdefault(n.targets[0].id, []).append(n.value)

    def val(k):
        n = nodes[k][-1]
        if isinstance(n, ast.DictComp):
            assert k == 'tt_v10_sm2_SMALLM2_ROUTES'
            assert ast.unparse(n) == "{key: value for (key, value) in tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}"
            return {('pooler', 8): ast.literal_eval(nodes[k][0])['pooler', 8]}
        if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == '_zsrc':
            return zlib.decompress(base64.b64decode(ast.literal_eval(n.args[0]))).decode()
        return ast.literal_eval(n)

    ns = {k: val(k) for k in ('_R2_DEV', '_R2_HOST')}
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_r2_source')
    exec(compile(ast.Module(body=[f], type_ignores=[]), 'exact-parent-r2-source', 'exec'), ns)
    v, s = val('V18S_CUDA_TEMPLATE'), val('tt_v10_sm2_SMALLM2_CUDA_TEMPLATE')
    entries = []
    for name, prefix, src, inst, fused in (
        ('o1', 'smallm2', s, val('V12_SM2_O1')['o', 1][0], False),
        ('o8', 'v18s', v, val('V18_SMALL')['o', 8][0], False),
        ('r1', 'v18s', v, val('V18_SMALL')['pooler', 1][0], True),
        ('r8', 'smallm2', s, val('tt_v10_sm2_SMALLM2_ROUTES')['pooler', 8][0], True),
    ):
        if fused:
            src = ns['_r2_source'](src, prefix, inst)
            assert src == (HERE / ('r2f_' + prefix + '.cu')).read_text()
            inst = inst[:-1] + (inst[-1] | 64,)
        entries.append((name, prefix, src.replace('@@CASES@@', ''), inst, fused))
    return entries


def encode(x):
    return '_zsrc(' + repr(base64.b64encode(zlib.compress(x.encode(), 9)).decode()) + ')'


def main():
    parent = (HERE / 'decoder_r2f.py').read_bytes()
    parent_hash = hashlib.sha256(parent).hexdigest()
    assert parent_hash.startswith('75fa2a73')
    src = parent.decode()
    sources, hashes, instances, bindings = {}, {}, {}, {}
    for name, prefix, cuda, inst, fused in source_specs(src):
        cuda, binding = twin(cuda, prefix, inst, fused=fused, form='call')
        compiled = RESULTS / 'pdl-cpu/call' / (name + '.cu')
        assert compiled.read_bytes() == cuda.encode(), 'must use the CPU-screened exact CUDA'
        args = ','.join(map(str, inst))
        if not fused:
            cuda += ('\nint pdl_o_launch(int64_t wp,int64_t x,int64_t out,int64_t esc,int m,int N,int K,int WR,int WK,int KU,int S,'
                     'int64_t t01,int64_t u01,int64_t stream){return ' + prefix + '_launch_t<' + args +
                     '>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,0);}\n')
        cfg = ','.join(map(str, inst[10:14]))
        cuda += ('\nint pdl_info(int m,int query){if(query!=1&&query!=2)return -90;return ' + prefix + '_launch_t<' + args +
                 '>(0,0,0,0,m,' + str(inst[8]) + ',' + str(inst[9]) + ',' + cfg + ',0,0,0,query' +
                 (',R2Args{}' if fused else '') + ');}\n')
        if name == 'o1':
            cuda += SPIN
        sources[name] = cuda
        hashes[name] = hashlib.sha256(cuda.encode()).hexdigest()
        instances[name] = inst
        binding['full_translation_unit_sha256'] = hashes[name]
        bindings[name] = binding
        (HERE / ('pdld_' + name + '.cu')).write_bytes(cuda.encode())
    app = '\n# Step 1 PDL diagnostic: no production path is enabled.\n_PDLD_SOURCES = {\n'
    app += ''.join(repr(k) + ': ' + encode(v) + ',\n' for k, v in sources.items()) + '}\n'
    app += '_PDLD_HASHES = ' + repr(hashes) + '\n_PDLD_INST = ' + repr(instances) + '\n_PDLD_O_CPP = ' + repr(O_CPP) + '\n'
    runtime = (HERE / 'pdl_diag_runtime.py').read_text()
    # One extension owns the diagnostic spin as well as O1.
    a = "functions=[fn, 'pdl_info']"
    b = "functions=[fn, 'pdl_info'] + (['pdl_spin_launch'] if name == 'o1' else [])"
    assert runtime.count(a) == 1
    runtime = runtime.replace(a, b)
    out = append_harness(src) + app + runtime
    ast.parse(out)
    b = out.encode()
    assert len(b) < 1_000_000
    # The full original parent prefix, including kernel_fn, is preserved exactly.
    assert out.startswith(src.rstrip('\n'))
    (HERE / 'decoder_pdl_diag.py').write_bytes(b)
    record = {'parent_sha256': parent_hash, 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b),
              'headroom': 1_000_000-len(b), 'production_dispatch_unchanged': True, 'bindings': bindings,
              'GPU_jobs_launched': 0}
    (RESULTS / 'pdl-diag-binding.json').write_bytes((json.dumps(record, indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in record.items() if k!='bindings'}, indent=2))


if __name__ == '__main__':
    main()
