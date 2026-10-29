"""CPU ownership, replay-state and numerical screens; not GPU validation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from build_attention_external_finalize import ROOT,OUT,freeze


def bf16(x):
    x=np.asarray(x,dtype=np.float32)
    bits=x.view(np.uint32)
    rounded=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
    return rounded.view(np.float32)


def normalized(acc,denom,gates,scale,mutant=None):
    inverse=np.divide(np.float32(scale),denom,out=np.zeros_like(denom),where=denom>0)
    y=(acc*inverse[:,None]).astype(np.float32)
    if mutant!='skip_attention_BF16':y=bf16(y)
    if mutant=='wrong_norm_domain':
        ss=np.sum(y.astype(np.float64)**2,axis=1)/8192
        gain=(1/np.sqrt(ss+1e-5)).astype(np.float32)[:,None]
    else:
        lane=y.reshape(-1,8,8)
        ss=np.zeros(lane.shape[:2],dtype=np.float32)
        for e in range(8):
            ss=(lane[:,:,e].astype(np.float64)**2+ss.astype(np.float64)).astype(np.float32)
        for offset in (1,2,4):ss=(ss+ss[:,np.arange(8)^offset]).astype(np.float32)
        gain=(np.float32(1)/np.sqrt((ss/np.float32(64)+np.float32(1e-5)).astype(np.float32))).repeat(8,axis=1)
    normed=(y*gain).astype(np.float32)
    if mutant=='round_norm_before_gate':normed=bf16(normed)
    gate=(np.float32(1)/(np.float32(1)+np.exp(-gates).astype(np.float32))).astype(np.float32)
    return bf16((normed*gate).astype(np.float32))


def reference(acc,denom,gates,scale):
    # Independent mathematical row oracle, with the required BF16 attention
    # materialization and one final BF16 output. Uses high-precision row sums.
    inv=np.divide(np.float32(scale),denom,out=np.zeros_like(denom),where=denom>0)
    y=bf16((acc*inv[:,None]).astype(np.float32))
    rms=np.sqrt(np.mean(y.astype(np.float64)**2,axis=1)+1e-5)
    gate=1/(1+np.exp(-gates.astype(np.float64)))
    return bf16(((y.astype(np.float64)/rms[:,None])*gate).astype(np.float32))


def main():
    binding=json.loads((OUT/'binding.json').read_bytes())
    assert hashlib.sha256((OUT/'submission.py').read_bytes()).hexdigest()==binding['source_sha256']
    ownership=0;largest=0
    for batch in (2,3,4,8,16,32,33,64,65,128,512):
        for pairs in (1,2,4,8):
            if batch*pairs>4096:continue
            tokens=np.arange(batch*8,dtype=np.int64)[:,None,None]
            pair=np.arange(pairs,dtype=np.int64)[None,:,None]
            head=np.arange(16,dtype=np.int64)[None,None,:]
            workspace=(((tokens>>3)*pairs+pair)*8+(tokens&7))*16+head
            output=tokens*(16*pairs)+pair*16+head
            count=batch*8*pairs*16
            assert np.array_equal(np.sort(workspace.reshape(-1)),np.arange(count))
            assert np.array_equal(np.sort(output.reshape(-1)),np.arange(count))
            # Original L8 addresses, independently expanded from (b,pair,j,h).
            b=tokens//8;j=tokens%8
            assert np.array_equal(workspace,((b*pairs+pair)*8+j)*16+head)
            assert np.array_equal(output,(b*8+j)*(16*pairs)+16*pair+head)
            assert (batch*8)*(pairs*2304)<2**31 and count*64<2**31
            ownership+=1;largest=max(largest,count)
    # Every local head uses eight lanes, each owning eight consecutive values.
    offsets=[(tid//8)*64+(tid%8)*8+e for tid in range(128) for e in range(8)]
    assert sorted(offsets)==list(range(16*64))
    assert all((tid//32)==((tid//8*8)//32) for tid in range(128))
    # Alternate old cooperative epochs and external launches on the same
    # sequential workspace. External finalization clears payloads but leaves
    # the epoch/counter state untouched, which is valid for the next old call.
    replay=0
    for grid in (1,33,128,264,512):
        epoch=0;counters=[0,0]
        payload=np.zeros(1024,dtype=np.float32)
        for index,arm in enumerate(('old','external','external','old','old','external','old')):
            assert not np.any(payload)
            payload[:]=np.float32(index+1)
            before=(epoch,tuple(counters))
            if arm=='old':
                parity=epoch&1
                assert counters[parity]==0
                counters[parity]=grid;counters[parity^1]=0;epoch+=1
            else:assert before==(epoch,tuple(counters))
            observed=payload.copy();payload[:]=0
            assert np.all(observed==np.float32(index+1)) and not np.any(payload)
            replay+=1
    rng=np.random.default_rng(20261029)
    numerical=[];mutants={m:0 for m in ('skip_attention_BF16','round_norm_before_gate','wrong_norm_domain')}
    for batch,pairs in ((2,1),(3,2),(4,4)):
        rows=batch*pairs*8*16
        for fixture in ('random','zero','constant','small','outlier'):
            acc=rng.normal(size=(rows,64)).astype(np.float32)
            denom=(np.abs(rng.normal(size=rows))+.1).astype(np.float32)
            gates=bf16(rng.normal(0,3,size=(rows,64)).astype(np.float32))
            if fixture=='zero':acc[:]=0;denom[:]=0
            if fixture=='constant':acc[:]=1;denom[:]=1;gates[:]=0
            if fixture=='small':acc*=np.float32(1e-5)
            if fixture=='outlier':acc[::17,31]=32;gates[::13,27]=-16
            for scale in (.00390625,.03125,1.):
                old=reference(acc,denom,gates,scale)
                value=normalized(acc,denom,gates,scale)
                # Gather workspace rows in the new token-major CTA order and
                # scatter to output rows; check the gate addresses independently.
                token=np.arange(batch*8)[:,None,None]
                pair=np.arange(pairs)[None,:,None];head=np.arange(16)[None,None,:]
                wi=((((token>>3)*pairs+pair)*8+(token&7))*16+head).reshape(-1)
                oi=(token*(16*pairs)+16*pair+head).reshape(-1)
                gate_output=np.empty_like(gates);gate_output[oi]=gates[wi]
                scattered=np.empty_like(value)
                scattered[oi]=normalized(acc[wi],denom[wi],gate_output[oi],scale)
                expected=np.empty_like(old);expected[oi]=old[wi]
                assert np.isfinite(scattered).all()
                l2=float(np.linalg.norm((scattered-expected).astype(np.float64))/max(np.linalg.norm(expected.astype(np.float64)),1e-12))
                assert l2<5e-4,(fixture,scale,l2)
                # The two index orders produce exactly equal FP32 source data,
                # independent of the CPU approximation to CUDA rsqrt/exp2.
                same=np.empty_like(value);same[oi]=value[wi]
                assert np.array_equal(same,scattered)
                if fixture=='constant' and scale==1.:
                    assert np.all(scattered==np.float32(.5))
                for m in mutants:
                    if not np.array_equal(value,normalized(acc,denom,gates,scale,m)):mutants[m]+=1
                numerical.append(dict(batch=batch,pairs=pairs,fixture=fixture,vscale=scale,relative_L2_vs_row_oracle=l2))
    assert all(mutants.values())
    # Pair-sensitive address mutation: token*pairs+pair is not the [B,pairs,L]
    # workspace order. It must differ whenever both L>1 and pairs>1.
    assert any((((t//8)*2+p)*8+t%8)!=(t*2+p) for t in range(24) for p in range(2))
    baseline=json.loads((ROOT/'results/attention-segment-repair-20261028/publication'/binding['parent_source_sha256']/'result.json').read_bytes())['result']
    rows=[r for r in baseline['records'] if r.get('event')=='benchmark']
    screens=[]
    for saving in (1.,2.,4.):
        loads={}
        for row in rows:
            a,b=loads.setdefault(row['case']['load'],(0.,0.))
            old=row['median_ns']/1000;new=old-saving if row['index'] in (9,11) else old
            w=row['case']['weight'];loads[row['case']['load']]=(a+w*old,b+w*new)
        screens.append(dict(hypothetical_net_full_case_saving_us=saving,
                            track_gain=float(np.prod([a/b for a,b in loads.values()])**(1/6))))
    report=dict(source_sha256=binding['source_sha256'],CUDA_sha256=binding['CUDA_sha256'],CPU_only=True,GPU_jobs=0,
        ownership_shapes=ownership,largest_head_rows_checked=largest,per_CTA_values_written_once=1024,
        sequential_old_external_epoch_cases=replay,numerical_proxy_cases=numerical,mutants_detected=mutants,
        pair_sensitive_index_mutant_detected=True,extra_timed_kernel_launches=1,extra_workspace_bytes=0,
        per_workspace_head_row_payload_bytes_read=260,per_workspace_head_row_payload_bytes_zeroed=260,
        per_workspace_head_row_BF16_gate_read_bytes=128,per_workspace_head_row_BF16_output_write_bytes=128,
        traffic_scope='Representation payload only; scalar-broadcast requests/cache-line traffic excluded. Same payload as old nc1 finalizer; no measured HBM claim.',
        optimistic_net_saving_screens=screens,
        limitations=['CPU mathematical/indexing screens do not establish GPU accuracy, race freedom or speed',
                     'Global FP32 partial reduction order may differ; GPU full suite remains mandatory',
                     'Original/native scope still excludes L8 B>64; large mathematical ownership cases do not enable those paths',
                     'Native actual resource/old cooperative selection must be observed; comments/planner alone are not proof'])
    freeze(OUT/'CPU-model.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='numerical_proxy_cases'},indent=2))


if __name__=='__main__':main()
