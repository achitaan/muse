"""Pin the resource conflict and screen one smaller O32 ring without GPU work."""

import ast
import hashlib
import json
import re
import subprocess

from pdl16_sources import HERE, ROOT, BASE_SHA
from build_pdl32_handoff import scoped
from check_pdl_diag_cpu import Capture, MAIN
from check_pdl16_cpu import instructions
import nvrtc_screen_alternatives as nv

OUT=ROOT/'results/decode-moe-plan-20261030/pdl32-residency'


def ring_proof(stages):
    # WGMMA wait_group 1 leaves at most the latest committed group pending.
    # Original PIPE1 refills done=i-1 only after that wait.
    slots={i:i for i in range(stages)};pending=[];consumed=[];loaded=list(range(stages));phases=[]
    for i in range(8):
        assert slots[i%stages]==i
        phases.append((i,i%stages,(i//stages)&1))
        pending.append(i)
        while len(pending)>1:consumed.append(pending.pop(0))
        done=i-1
        if done>=0 and done+stages<8:
            assert done in consumed and done not in pending
            slot=done%stages
            assert slots[slot]==done
            slots[slot]=done+stages;loaded.append(done+stages)
    consumed.extend(pending)  # original final wait_group 0, before BF16 stores
    assert loaded==list(range(8)) and consumed==list(range(8))
    return dict(stages=stages,logical_tiles_loaded=loaded,consumed_in_K_order=consumed,
                wait_phases=phases,refill_only_after_old_group_completion=True,
                transaction_bytes_per_load=(64+32)*128*2,
                dynamic_shared_bytes=stages*((64+32)*128*2+16))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=(HERE/'decoder_pdli.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==BASE_SHA
    tree=ast.parse(raw)
    values={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)}
    attn=ast.literal_eval(values['_native_attn_CUDA_SRC'])
    def struct(name):
        start=attn.rfind('template',0,attn.index('struct '+name+' {'))
        end=attn.index('\n};',start)+4
        return attn[start:end]
    tag='#include <stdint.h>\n#include <cuda_bf16.h>\n'
    tag+='constexpr int STAGE_BYTES=8192,TILE=64,ROWB=128,MAXB=512,MAXSEG=64;\n'
    tag+=struct('Cfg')+'\n'+struct('Smem')+'\n'
    tag+='template<int Bytes,int Threads> __global__ void resource_tag(){}\n'
    nvr=Capture();r=nvr.compile(tag.encode(),['&resource_tag<sizeof(Smem<1,false>)+1024,Cfg<1,false>::NTX>'],[],tags=True,lineinfo=False)
    (OUT/'attention-size-proxy.cu').write_bytes(tag.encode());(OUT/'attention-size.log').write_text(r['log'],encoding='utf-8')
    assert r['rc']==0,r['log'][-1800:]
    kernels=nv.parse_ptxas(r['log']);assert len(kernels)==1
    match=re.search(r'ILi(\d+)ELi(\d+)E',kernels[0]['symbol']);assert match
    shared,threads=map(int,match.groups());assert (shared,threads)==(58368,128)
    # Execute only the pinned pure metadata planner, with recorded H200 SM count.
    names={'_native_attn__splits','_native_attn__copies','_native_attn__plan'}
    env={'_native_attn__num_sms':lambda _:132,'_native_attn_MAX_BATCH':512,'_native_attn_TILE_KEYS':64}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[]),'<pinned-planner>','exec'),env)
    plan=env['_native_attn__plan'](32,1,1,2048,80001,1,None)
    assert plan==dict(n_splits=8,flat=0,cs=0,ncl=1,nc=1)
    tuned_keys=[ast.literal_eval(k) for k in values['_native_attn__TUNED'].keys]
    assert (32,1,True,1)not in tuned_keys
    hardware=json.loads((OUT.parent/'o32-resource-fixed/summary-decoded.json').read_text(encoding='utf-8'))
    capacity=hardware['resources']['private'][7];assert capacity==233472
    old=(OUT.parent/'pdl16/o32-candidate.cu').read_text(encoding='utf-8')
    assert hashlib.sha256(old.encode()).hexdigest()=='b45d6e23af85adc9cb16e41e5c02f445714188dd28d0fa5c8a14855e96526f7f'
    new=old;edits=[]
    for a,b in [('32,128,8,1,1,1,1','32,128,4,1,1,1,1'),('ST==8','ST==4'),
                ('196736','98368'),('fixed full-K O32 ring','fixed four-stage O32 ring')]:
        count=new.count(a);assert count>0
        new=new.replace(a,b);edits.append(dict(before=a,after=b,count=count))
    inverse=new
    for e in reversed(edits):inverse=inverse.replace(e['after'],e['before'])
    assert inverse==old
    (OUT/'o32-st4.cu').write_bytes(new.encode())
    dev=new.split('using OLEncode=',1)[0]
    dev=scoped(dev,'o32s4_private','#define OD __device__ __forceinline__')
    dev=dev.replace('#include <cuda.h>','struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };').replace('#include <cuda_runtime.h>','')
    nvr=Capture();c=nvr.compile(dev.encode(),['&o32s4_private::pdlo32_projection<32,128,4,1,1,1,1>'],[],tags=True,lineinfo=True)
    (OUT/'o32-st4-device-proxy.cu').write_bytes(dev.encode());(OUT/'o32-st4.log').write_text(c['log'],encoding='utf-8')
    assert c['rc']==0,c['log'][-1800:]
    c['kernels']=nv.parse_ptxas(c['log']);assert len(c['kernels'])==1
    assert all(c['kernels'][0][k]==0 for k in ('stack_frame','spill_stores','spill_loads'))
    binary=OUT/'o32-st4.cubin';binary.write_bytes(nvr.cubin)
    exe=MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    sass=subprocess.check_output([str(exe),'-sass',str(binary)],text=True);(OUT/'o32-st4.sass').write_bytes(sass.encode())
    ins=instructions(sass);acq=[a for a,t in ins if 'ACQBULK'in t];assert len(acq)==1
    calls=[(a,int(re.search(r'0x([0-9a-f]+)',t)[1],16))for a,t in ins if'CALL.REL.NOINC'in t]
    calls=[(a,t)for a,t in calls if t<=acq[0]<t+64];assert len(calls)==1
    call=calls[0][0];before=[(a,t)for a,t in ins if a<call and'UBLKCP.S.G'in t]
    activation=[(a,t)for a,t in ins if'UTMALDG.3D'in t]
    assert len(before)==4 and activation and all(a>call for a,t in activation)
    m8,m4=ring_proof(8),ring_proof(4)
    report=dict(decode_sha256=BASE_SHA,Attention_CUDA_sha256=hashlib.sha256(attn.encode()).hexdigest(),
        M32_sliding_plan=plan,attention_grid_CTAs=256,attention_threads=threads,attention_dynamic_shared_bytes=shared,
        H200_shared_bytes_per_SM=capacity,O32_ST8_plus_one_attention_bytes=m8['dynamic_shared_bytes']+shared,
        ST8_cannot_share_an_SM_with_attention=m8['dynamic_shared_bytes']+shared>capacity,
        ST4_plus_two_attention_bytes=m4['dynamic_shared_bytes']+2*shared,
        ST4_shared_memory_necessary_condition_passes=m4['dynamic_shared_bytes']+2*shared<=capacity,
        ST4_source_sha256=hashlib.sha256(new.encode()).hexdigest(),edits=edits,inverse_exact=True,
        ring_models=[m8,m4],ST4_CPU_kernels=c['kernels'],ST4_weights_before_wait=before,
        ST4_first_activation_TMA=activation[0],GPU_jobs=0,admitted_for_GPU=False,
        conclusion='ST8 overlap is limited to unused/freed SMs; the one-CTA ceiling cannot establish real attention overlap. ST4 removes this shared-memory conflict but is an unmeasured new schedule. Registers, CTA placement, trigger timing, byte correctness and net real-chain benefit still need GPU evidence. No speedup claimed.')
    (OUT/'model.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
