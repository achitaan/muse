"""CPU device-proxy screen of immutable norm arithmetic and trigger ordering."""

import hashlib
import json
import subprocess
from collections import Counter

from pdl_norm_sources import OUT, UPSTREAM, pdln_derive
from check_pdl_diag_cpu import Capture, MAIN
from check_pdl16_cpu import instructions
import nvrtc_screen_alternatives as nv


PRE = r'''
#include <cuda_bf16.h>
#include <stdint.h>
namespace std { using size_t = decltype(sizeof(0)); }
#define SGL_DEVICE __device__ __forceinline__
namespace sglang {
using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;
using fp32_t = float;
using fp32x2_t = float2;
namespace device {
constexpr int kMaxVecBytes = 16, kWarpThreads = 32;
template<bool Enabled> SGL_DEVICE void PDLWaitPrimary() {
 if constexpr (Enabled) asm volatile("griddepcontrol.wait;" ::: "memory");
}
template<bool Enabled> SGL_DEVICE void PDLTriggerSecondary() {
 if constexpr (Enabled) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
template<class T, class U> SGL_DEVICE T cast(const U& v);
template<> SGL_DEVICE float2 cast<float2,__nv_bfloat162>(const __nv_bfloat162& v) {
 return __bfloat1622float2(v);
}
template<> SGL_DEVICE __nv_bfloat162 cast<__nv_bfloat162,float2>(const float2& v) {
 return __float22bfloat162_rn(v);
}
namespace math { SGL_DEVICE float rsqrt(float x) { return rsqrtf(x); } }
namespace warp {
template<class T> SGL_DEVICE T reduce_sum(T value) {
 #pragma unroll
 for (int mask=16;mask>0;mask>>=1) value += __shfl_xor_sync(0xffffffff,value,mask,32);
 return value;
}
}
}
}
'''


def proxy(source):
    # Preserve the complete vector storage/load/store implementation. Only its
    # unused generic reduction tail and host includes are excluded.
    vec = (UPSTREAM / 'vec.cuh').read_text()
    vec = vec[vec.index('namespace sglang {'):vec.index('/// \\brief Maximum vector width')]
    vec += '\n} }\n'
    cta = (UPSTREAM / 'cta.cuh').read_text()
    start = cta.index('template <int kNumThreads, typename T>')
    end = cta.index('\n}', start) + 2
    cta = 'namespace sglang::device::cta {\n' + cta[start:end] + '\n}\n'
    scalar = source[source.index('namespace sglang {'):source.index('// Multi-CTA path')]
    scalar += '\n}\n'
    return PRE + vec + cta + scalar


def main():
    parent = (UPSTREAM / 'norm_residual.cuh').read_text()
    srcs = {'parent': parent, 'early': pdln_derive(parent)}
    records = {}; math_counts = []
    for arm, source in srcs.items():
        code = proxy(source)
        name = 'muse_spark_norm_residual_kernel' if arm == 'parent' else 'pdln_norm_kernel'
        expr = '&sglang::' + name + '<8192,256,true,true,false,false,false,true>'
        compiler = Capture()
        result = compiler.compile(code.encode(), [expr], [], tags=True, lineinfo=True)
        (OUT / (arm + '-device-proxy.cu')).write_bytes(code.encode())
        (OUT / (arm + '-compile.log')).write_text(result['log'])
        assert result['rc'] == 0, result['log'][-2500:]
        result['kernels'] = nv.parse_ptxas(result['log'])
        assert len(result['kernels']) == 1
        cubin = OUT / (arm + '.cubin'); cubin.write_bytes(compiler.cubin)
        sass = subprocess.check_output([str(MAIN / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),
                                        '-sass', str(cubin)], text=True)
        (OUT / (arm + '.sass')).write_bytes(sass.encode())
        ins = instructions(sass)
        waits = [a for a, t in ins if 'ACQBULK' in t]
        triggers = [a for a, t in ins if 'PREEXIT' in t]
        loads = [a for a, t in ins if 'LDG.' in t]
        stores = [a for a, t in ins if 'STG.' in t]
        assert len(waits) == len(triggers) == 1 and loads and stores
        if arm == 'parent':
            assert waits[0] < loads[0] and triggers[0] > stores[-1]
        else:
            assert triggers[0] < waits[0] < loads[0]
        math = Counter(t.split()[0] for a, t in ins if t.startswith(
            ('FFMA','FADD','FMUL','MUFU','F2F','F2FP','SHFL')))
        math_counts.append(math)
        result.update(device_proxy_sha256=hashlib.sha256(code.encode()).hexdigest(),
                      trigger_address=triggers[0], wait_address=waits[0], first_LDG=loads[0],
                      last_STG=stores[-1], math_counts=dict(math))
        records[arm] = result
    assert math_counts[0] == math_counts[1]
    assert records['parent']['kernels'][0]['registers'] == records['early']['kernels'][0]['registers']
    assert all(not any(k[f] for f in ['stack_frame','spill_stores','spill_loads'])
               for r in records.values() for k in r['kernels'])
    result = dict(GPU_jobs=0, selected_instance='8192,256,PDL,prefetch,std,no_router,no_post_weight,gate',
                  records=records, arithmetic_instruction_counts_match=True,
                  limitation='Device proxy: exact scalar body/vector storage/CTA reduction; selected CUDA intrinsic/type adapters replace SDK host-heavy headers. Actual SDK build, resources and execution remain unverified.')
    (OUT / 'cpu-compile-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({a:{k:v for k,v in r.items() if k not in ('log','math_counts')}
                      for a,r in records.items()}, indent=2))


if __name__ == '__main__':
    main()
