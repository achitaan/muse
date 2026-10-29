"""Compile-only O1/O8 cooperative boundary prototype; never launches GPU work.

This is not a submission. Host launch, graph capture, exact evaluator numerical
validation and timing remain unimplemented. Source selection is bound to D9.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from preflight_decoder_projection_pdl import BASE, CONDITIONAL, ROOT, literal

HELPERS = r'''
struct CoopBoundary {
  const float *stream, *alpha, *beta;
  const __nv_bfloat16 *pre_weight;
  float *new_stream, *router;
  __nv_bfloat16 *output;
  float post_eps, pre_eps;
};
__device__ __forceinline__ float coop_sum(float value, float *scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int d=16; d; d>>=1) value += __shfl_down_sync(0xffffffff, value, d);
  if (!lane) scratch[warp] = value;
  __syncthreads();
  if (!warp) {
    value = lane < blockDim.x/32 ? scratch[lane] : 0.f;
    for (int d=16; d; d>>=1) value += __shfl_down_sync(0xffffffff, value, d);
    if (!lane) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}
// Three streaming passes avoid keeping a full FP32 residual row and all gate
// vectors live in the GEMV register allocation. They deliberately add reads.
__device__ __forceinline__ void coop_boundary(
    const __nv_bfloat16 *branch, int m, CoopBoundary p, float *scratch) {
  cooperative_groups::this_grid().sync();
  const int row = blockIdx.x;
  if (row >= m) return;
  constexpr int H = 8192;
  const size_t base = (size_t)row * H;
  float sum = 0.f;
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float b = __bfloat162float(branch[base+h]);
    sum = fmaf(b,b,sum);
  }
  const float post = rsqrtf(coop_sum(sum,scratch)/H+p.post_eps);
  float pre_sum = 0.f;
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float b = __bfloat162float(branch[base+h]);
    const float normed = __bfloat162float(__float2bfloat16_rn(b*post));
    const float a = __fmul_rn(p.alpha[h],p.stream[base+h]);
    const float c = __fmul_rn(p.beta[h],normed);
    const float value = __fadd_rn(a,c);
    p.new_stream[base+h] = value;
    pre_sum = fmaf(value,value,pre_sum);
  }
  // Separate scratch prevents a WAR race on the first returned sum.
  const float pre = rsqrtf(coop_sum(pre_sum,scratch+32)/H+p.pre_eps);
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float value = __fmul_rn(__fmul_rn(p.new_stream[base+h],pre),
                                 __bfloat162float(p.pre_weight[h]));
    p.output[base+h] = __float2bfloat16_rn(value);
    // Exact selected CUDA source retains FP32 here with pre_norm_bf16=false.
    // Native/Triton router alternatives must not be silently substituted.
    p.router[base+h] = value;
  }
}
'''


def freeze(path, data):
    assert not path.exists() or path.read_bytes() == data, path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    raw = (ROOT/'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    values = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values.setdefault(target.id, []).append(node.value)

    def get(name):
        assert len(values[name]) == 1
        return literal(values[name][0])

    out = ROOT/'.tools/decoder-cooperative-boundary-preflight-v1'
    out.mkdir(exist_ok=True)
    specs = [
        ('o1', get('tt_v10_sm2_SMALLM2_CUDA_TEMPLATE'), 'smallm2', get('V12_SM2_O1')['o',1][0],1),
        ('o8', get('V18S_CUDA_TEMPLATE'), 'v18s', get('V18_SMALL')['o',8][0],8),
    ]
    manifest = dict(baseline_sha256=BASE, GPU_jobs=0, candidate_installed=False,
                    host_launcher_implemented=False, variants=[])
    for label,cuda,symbol,inst,m in specs:
        assert inst[0] == 1 and inst[4:6] == (0,0) and inst[13] == 1
        assert inst[8:10] == (8192,1024) and inst[11] > 1
        cutoff = f'static int {symbol}_smem_optin()'
        assert cuda.count(cutoff)==1
        prefix = cuda.split(cutoff)[0]
        proxy = prefix.replace('#include <algorithm>','').replace('#include <type_traits>',CONDITIONAL)
        anchor = 'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>'
        assert proxy.count(anchor)==1
        changed = proxy.replace(anchor, HELPERS+'\n'+anchor)
        signature = '\n    uint32_t U0, uint32_t U1) {'
        assert changed.count(signature)==1
        changed=changed.replace(signature,'\n    uint32_t U0, uint32_t U1, CoopBoundary boundary) {')
        tail='\n  }\n}\n\n'
        assert changed.endswith(tail)
        # Only RED=0/WK>1, non-diagnostic direct selected paths are instantiated.
        # None of their threads takes an early return before this grid barrier.
        changed=changed[:-len(tail)]+'\n  }\n  coop_boundary(reinterpret_cast<const __nv_bfloat16*>(outp), m, boundary, reinterpret_cast<float*>(smem));\n}\n\n'
        inverse=changed.replace(HELPERS+'\n','',1).replace(
            '\n    uint32_t U0, uint32_t U1, CoopBoundary boundary) {',signature,1).replace(
            '\n  }\n  coop_boundary(reinterpret_cast<const __nv_bfloat16*>(outp), m, boundary, reinterpret_cast<float*>(smem));\n}\n\n',tail,1)
        assert inverse==proxy, 'GEMV source or ownership changed outside the explicit append/signature'
        expressions=[symbol+'_gemv<'+','.join(map(str,(fmt,)+inst[1:]))+'>' for fmt in (1,0)]
        path=out/(label+'-cooperative.cu')
        freeze(path,changed.encode())
        command=[sys.executable,str(ROOT/'experiments/prepared/local_nvrtc_compile.py'),
                 '--source',str(path),'--minimal-std-tags','--out',str(path.with_suffix(''))]
        for expression in expressions: command+=['--name',expression]
        if not path.with_suffix('.json').exists(): subprocess.run(command,cwd=ROOT,check=True)
        report=json.loads(path.with_suffix('.json').read_text())
        assert report['return_code']==0
        assert report['original_source_sha256']==hashlib.sha256(changed.encode()).hexdigest()
        manifest['variants'].append(dict(label=label,tokens=m,baseline_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
            source_sha256=report['original_source_sha256'],instances=expressions,
            threads=inst[6],ctas=8192//(inst[10]*inst[2]*16),
            dynamic_shared_bytes=inst[10]*inst[11]*inst[2]*inst[1]*512,
            log=path.with_suffix('.log').read_text()))
    freeze(out/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(dict(variants=len(manifest['variants']),GPU_jobs=0,candidate_installed=False)))


if __name__=='__main__': main()
