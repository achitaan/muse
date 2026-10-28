"""Cache the FP32 router input K-split once per CTA; keep all arithmetic."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
BASE_SHA = '09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67'
ANCHOR = '  float a[TM], b[TM];'
COPY = '''  // Local change: all eight warps consume the same FP32 input K-split.
  // Each active element is copied once, with no conversion or arithmetic.
  __shared__ float xshared[TM * 1024];
  for (int i = tid; i < M * 1024; i += blockDim.x) {
    xshared[i] = x[(size_t)(i >> 10) * 8192 + ks * 1024 + (i & 1023)];
  }
  __syncthreads();
'''
READ_OLD = 'const float v = x[(size_t)m * 8192 + k];'
READ_NEW = 'const float v = xshared[m * 1024 + j * 32 + lane];'


def build():
    raw = (ROOT/'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == '_RF_CUDA' for t in n.targets))
    cuda = ast.literal_eval(node.value)
    assert cuda.count(ANCHOR) == cuda.count(READ_OLD) == 1
    assert cuda.count('route_fp32') == 3
    changed = cuda.replace(ANCHOR,COPY+ANCHOR,1).replace(READ_OLD,READ_NEW,1)
    changed = changed.replace('route_fp32','route_fp32_xshared')
    assert changed.replace('route_fp32_xshared','route_fp32').replace(COPY,'',1).replace(READ_NEW,READ_OLD,1) == cuda
    lines = raw.decode('utf8').splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['_RF_CUDA = '+repr(changed)+'\n']
    result = ''.join(lines).encode('utf8')
    compile(result,'<decode-router-shared>','exec')
    assert len(result) < 1_000_000
    recovered = ast.parse(result)
    target = next(n for n in recovered.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id == '_RF_CUDA' for t in n.targets))
    target.value = ast.Constant(cuda)
    assert ast.dump(recovered,include_attributes=False) == ast.dump(tree,include_attributes=False)
    (ROOT/'.tools/decoder-router-shared-variant.py').write_bytes(result)
    (ROOT/'.tools/decoder-router-shared-inspect.cu').write_bytes(changed.encode('utf8'))
    record = {'source_sha256':hashlib.sha256(result).hexdigest(),'base_sha256':BASE_SHA,
              'packaged_bytes':len(result),'original_module_ast_recovered':True,
              'only_changed_literal':'_RF_CUDA','cuda_patch_reversible':True,
              'kernel_name':'route_fp32_xshared','cta_grid':128,'threads':256,
              'shared_input_bytes_by_template':{'1':4096,'8':32768},
              'fp32_fma_reduction_routing_bytes_unchanged':True,
              'no_new_module_or_state_buffers':True,'gpu_status':'unmeasured'}
    (ROOT/'.tools/decoder-router-shared-source-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    build()
