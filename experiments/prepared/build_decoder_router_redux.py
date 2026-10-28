"""Exact ordered-FP32-key warp reductions for router top-eight selection."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
BASE_SHA = '09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67'
REDUCE_OLD = '''#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, d);
      const float os = __shfl_xor_sync(0xffffffffu, raw, d);
      const int oi = __shfl_xor_sync(0xffffffffu, id, d);
      if (ob > best || (ob == best && oi < id)) { best = ob; raw = os; id = oi; }
    }'''
REDUCE_NEW = '''    // Local change: an exact order key, not a quantized score. Local
    // comparisons never select a NaN. Canonicalize only the key for signed
    // zeros, which compare equal in the original floating-point selector.
    unsigned bits = __float_as_uint(best);
    if (best == 0.f) bits = 0u;
    const unsigned key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    const unsigned global_key = __reduce_max_sync(0xffffffffu, key);
    id = __reduce_min_sync(0xffffffffu, key == global_key ? id : INT_MAX);
    // The selected expert's owner lane still holds its original raw score.
    raw = __shfl_sync(0xffffffffu, raw, id & 31);'''


def build():
    raw = (ROOT/'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == '_RF_CUDA' for t in n.targets))
    cuda = ast.literal_eval(node.value)
    assert cuda.count(REDUCE_OLD) == 1 and cuda.count('route_fp32') == 3
    assert 'xshared' not in cuda
    changed = cuda.replace(REDUCE_OLD,REDUCE_NEW,1).replace('route_fp32','route_fp32_redux')
    assert changed.replace('route_fp32_redux','route_fp32').replace(REDUCE_NEW,REDUCE_OLD,1) == cuda
    lines = raw.decode('utf8').splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['_RF_CUDA = '+repr(changed)+'\n']
    result = ''.join(lines).encode('utf8')
    compile(result,'<decode-router-redux>','exec')
    assert len(result) < 1_000_000
    recovered = ast.parse(result)
    for n in recovered.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == '_RF_CUDA' for t in n.targets):
            n.value = ast.Constant(cuda)
    assert ast.dump(recovered,include_attributes=False) == ast.dump(tree,include_attributes=False)
    (ROOT/'.tools/decoder-router-redux-variant.py').write_bytes(result)
    (ROOT/'.tools/decoder-router-redux-inspect.cu').write_bytes(changed.encode('utf8'))
    record = {'source_sha256':hashlib.sha256(result).hexdigest(),'base_sha256':BASE_SHA,
              'packaged_bytes':len(result),'only_changed_literal':'_RF_CUDA',
              'original_module_ast_recovered':True,'cuda_patch_reversible':True,
              'kernel_name':'route_fp32_redux','dot_split_sigmoid_normalization_bytes_unchanged':True,
              'selection':'unsigned monotone FP32 key max; ascending signed ID min; unchanged raw-score broadcast',
              'signed_zero':'canonicalized in key only; score bits unchanged',
              'collective_source_operations_per_eight_picks':{'old_shuffles':120,'new_redux':16,'new_shuffles':8},
              'scope_limit':'Source operation count, not measured SASS instruction count or speedup',
              'gpu_status':'unmeasured'}
    (ROOT/'.tools/decoder-router-redux-source-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    build()
