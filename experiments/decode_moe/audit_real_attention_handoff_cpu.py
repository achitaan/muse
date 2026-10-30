"""Source/ABI/protocol audit, plus one native Attention instruction screen."""

import ast
import hashlib
import json
import re
import subprocess
import math
from typing import TypedDict

import torch

from pdl16_sources import HERE, ROOT, BASE_SHA
from build_real_attention_handoff_diag import OUT, early_attention
from check_pdl_diag_cpu import Capture, MAIN
from check_pdl16_cpu import instructions
import nvrtc_screen_alternatives as nv


def main():
    binding=json.loads((OUT/'source-binding.json').read_text(encoding='utf-8'))
    raw=(HERE/'dense_real_attention_handoff.py').read_bytes();dense=(ROOT/'best/verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==binding['source_sha256'] and raw.startswith(dense)
    suffix=raw[len(dense):].decode();nodes=ast.parse(suffix).body
    assert not any(isinstance(n,ast.FunctionDef)and n.name=='kernel_fn'for n in nodes)
    native=(OUT/'native.py').read_text(encoding='utf-8')
    # Initializing the frozen namespace must not build modules or touch CUDA.
    ns={'torch':torch,'math':math,'TypedDict':TypedDict};exec(compile(native,'frozen-native','exec'),ns)
    assert ns['_native_attn__MOD']is None and ns['_native_attn__WS']=={}
    assert ns['_native_attn_CUDA_SRC']==(OUT/'attention-parent.cu').read_text(encoding='utf-8')
    parent=(OUT/'attention-parent.cu').read_text(encoding='utf-8')
    early=(OUT/'attention-early-unscoped.cu').read_text(encoding='utf-8')
    assert early_attention(parent)==early
    binding_cpp=(OUT/'attention-private.cpp').read_text(encoding='utf-8')
    assert 'int eac_info(int);' in binding_cpp
    declaration=binding_cpp.split(';',1)[0]+';'
    assert declaration==ns['_native_attn_CPP_SRC'].strip().replace('void run(','void eac_run(',1)
    expected_names=[re.search(r'(\w+)\s*$',x)[1]for x in declaration.split('(',1)[1].rsplit(')',1)[0].split(',')]
    assert len(expected_names)==28
    private=(OUT/'attention-private.cu').read_text(encoding='utf-8')
    assert 'eac_attention_private::run('+','.join(expected_names)+');' in private
    scoped_part=private.split('\n} // eac_attention_private\n',1)[0]
    assert scoped_part.replace('namespace eac_attention_private {\n','',1)==early
    assert '#include <pybind11/pybind11.h>'in (OUT/'projections.cpp').read_text(encoding='utf-8')
    # All public callbacks stay retained; four fixed causal arms, no flag sweep.
    assert "for key in ('parent','late8','early8','early4')"in suffix
    assert "for stage in (8,4)"in suffix and "(data['k_cache'],data['v_cache'])"in suffix
    assert "['parent','late8','early8']"in suffix and 'mutate(0)'in suffix
    assert "return state"in suffix and "stage*24592"in suffix
    assert suffix.index('for thread in _eac_threads:thread.join()')<suffix.index("report['byte_pass'],")
    assert "if report['byte_pass']['parent_repeat']:"in suffix
    assert "Unchanged parent is not byte-stable; no performance admission."in suffix
    results={}
    exe=MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    for label,source in [('parent',parent),('early',early)]:
        if label=='early':
            device=private.split('template <int L>\nvoid launch(',1)[0]+'\n} // eac_attention_private\n'
            expression='&eac_attention_private::attn_kernel_fp8pv<1,false,1>'
        else:
            device=source.split('template <int L>\nvoid launch(',1)[0]
            expression='&attn_kernel_fp8pv<1,false,1>'
        # Only the tag's compile-time ::value is used in device code. This
        # minimal definition avoids depending on the host C++ standard library.
        device=re.sub(r'#include <(torch/extension.h|ATen/cuda/CUDAContext.h|type_traits|cuda.h)>\n','',device)
        device='#include <math_constants.h>\n#define INFINITY CUDART_INF_F\nnamespace std { template<class T,T V> struct integral_constant { static constexpr T value=V; }; }\n'+device
        nvr=Capture();r=nvr.compile(device.encode(),[expression],[],tags=True,lineinfo=True)
        (OUT/(label+'-device-proxy.cu')).write_bytes(device.encode())
        (OUT/(label+'-device.log')).write_text(r['log'],encoding='utf-8')
        assert r['rc']==0,r['log'][-2200:]
        r['kernels']=nv.parse_ptxas(r['log']);assert len(r['kernels'])==1
        assert all(r['kernels'][0][k]==0 for k in ('stack_frame','spill_stores','spill_loads'))
        binary=OUT/(label+'.cubin');binary.write_bytes(nvr.cubin)
        sass=subprocess.check_output([str(exe),'-sass',str(binary)],text=True)
        (OUT/(label+'.sass')).write_bytes(sass.encode());ins=instructions(sass)
        signals=[a for a,t in ins if'PREEXIT'in t];assert signals
        loads=[a for a,t in ins if'LDG.'in t]
        assert loads
        r.update(first_outgoing_signal_address=min(signals),first_global_load_address=min(loads),GPU_jobs=0)
        if label=='early':
            # The compiler hoists only the independent seq_lens load ahead of
            # the entry signal. It still precedes path exits and the mainloop.
            before=[(a,t)for a,t in ins if a<min(signals)and 'LDG.'in t]
            assert len(before)==1 and min(signals)<512
            assert any(a<before[0][0]and 'LDC.64'in t and '[0x218]'in t for a,t in ins)
            assert all(a>min(signals)for a,t in ins if 'EXIT'in t and 'PREEXIT'not in t)
            r['independent_seq_len_load_before_signal']=before
        results[label]=r;print(label,r['kernels'],'signal',min(signals),'load',min(loads),flush=True)
    assert results['parent']['kernels'][0]['registers']==results['early']['kernels'][0]['registers']==129
    assert results['early']['first_outgoing_signal_address']<results['parent']['first_outgoing_signal_address']
    report=dict(source_sha256=binding['source_sha256'],checks_passed=True,
        unchanged_Dense_prefix_and_kernel=True,source_inverse_exact=True,native_namespace_no_GPU_initialization=True,
        host_ABI_forwarded_arguments=28,explicit_projection_binding_header=True,
        fixed_causal_arms=True,cache_and_intermediate_bytes_required=True,CUDA_device_screens=results,
        GPU_jobs=0,admitted_for_GPU=False,
        limitation='Native host wrapper/JIT duration, H200 bytes/resources and real-chain net benefit remain unverified.')
    (OUT/'cpu-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('CPU source/ABI/device/protocol checks passed; zero GPU jobs.')


if __name__=='__main__':main()
