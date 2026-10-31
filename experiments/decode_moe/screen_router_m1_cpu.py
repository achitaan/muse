"""One-token specialization of retained129-CTA R1; no scheduling sweep or GPU work."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from screen_router_sidecar_cpu import ROOT, DECODER

OUT = ROOT / 'results/decode-moe-plan-20261030/router-m1'


def derive():
    raw=(ROOT/'experiments/decode_moe/decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DECODER
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')=='_PDLI_EXPECTED')
    expected=ast.literal_eval(node.value)
    parent=(ROOT/'results/decode-moe-plan-20261030/pdli-cpu/r1.cu').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==expected['r1']
    parent=parent.decode();src=parent;edits=[]
    for old,new,count in [
        ('__shared__ float r2x[8 * 1024];','__shared__ float r2x[1024];',1),
        ('const int M = ra.M;','const int M = 1;  // host enforces m == 1',2),
        ('v18s_launch_t<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 70>',
         'v18s_launch_t<1, 1, 1, 1, 0, 0, 1024, 1, 4096, 8192, 2, 16, 8, 1, 70>',1)]:
        assert src.count(old)==count,(old,src.count(old));src=src.replace(old,new);edits.append((old,new,count))
    inv=src
    for old,new,count in reversed(edits):
        assert inv.count(new)==count;inv=inv.replace(new,old)
    assert inv==parent
    assert 'if ((CM && m != CM)'in src
    # Both public native entry guards enforce m==1 before the specialized template launches.
    assert src.index('if ((CM && m != CM)')<src.index('cudaLaunchKernelEx(&lc')
    return parent,src


def bounds():
    for m in range(258):
        old_valid=1<=m<=8
        new_valid=old_valid and m==1
        assert new_valid==(m==1)
    staging=list(range(1024))
    used=[j*32+lane for j in range(32)for lane in range(32)]
    assert staging==used and max(used)==1023
    partials=[((b>>4)*8)*256+(b&15)*16+w for b in range(128)for w in range(16)]
    assert len(partials)==len(set(partials))==2048 and max(partials)<8*8*256
    return {'host_M_cases':258,'only_M1_enters':True,'all_shared_input_indices':1024,
            'shared_input_max_index':1023,'unique_router_partials':2048,
            'part_buffer_entries':8*8*256,'original_grid_CTA_count':129,
            'original_FMA_shuffles_and_counter_expressions_retained':True,
            'limitation':'Source guard/index argument;not GPU bitwise or race validation.'}


def main():
    OUT.mkdir(exist_ok=True);(OUT/'.gitignore').write_bytes(b'*.cubin\n')
    parent,candidate=derive();(OUT/'candidate.cu').write_bytes(candidate.encode())
    from check_r2_cpu import device_part
    from check_pdl_diag_cpu import Capture,MAIN
    import nvrtc_screen_alternatives as nv
    record={'decoder_sha256':DECODER,'parent_CUDA_sha256':hashlib.sha256(parent.encode()).hexdigest(),
            'candidate_CUDA_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'exact_inverse':True,
            'scope':'m1 only;129 CTAs and existing mixed router/pooler ownership retained;dynamic shared allocation unchanged',
            'GPU_jobs':0,'admitted_for_GPU':False,'bounds':bounds(),'resources':{}}
    for name,source,cm in [('parent',parent,0),('candidate',candidate,1)]:
        compiler=Capture();expr='&v18s_gemv<1,1,1,1,0,0,1024,'+str(cm)+',4096,8192,2,16,8,1,70>'
        r=compiler.compile(device_part(source).encode(),[expr],[],tags=True,lineinfo=True)
        (OUT/(name+'.log')).write_bytes(r['log'].encode());assert r['rc']==0,r['log'][-3000:]
        binary=OUT/(name+'.cubin');binary.write_bytes(compiler.cubin)
        sass=subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(binary)],text=True)
        (OUT/(name+'.sass')).write_bytes(sass.encode())
        ins=re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);',sass)
        record['resources'][name]={'kernels':nv.parse_ptxas(r['log']),'instructions':len(ins),
            'global_load_instructions':sum('LDG.'in t for a,t in ins),'shared_load_instructions':sum(bool(re.search(r'\bLDS(?:\.|\s)',t))for a,t in ins),
            'predicate_set_instructions':sum('SETP.'in t for a,t in ins),
            'local_instructions':sum(bool(re.search(r'\b(?:LDL|STL)\b',t))for a,t in ins),
            'cubin_sha256':r['cubin_sha256']}
        print(name,json.dumps(record['resources'][name]),flush=True)
    a,b=record['resources']['parent'],record['resources']['candidate']
    clean=b['local_instructions']==0 and all(k['registers']<=64 and k['stack_frame']==k['spill_stores']==k['spill_loads']==0 for k in b['kernels'])
    record.update(clean_resource_screen=clean,instruction_ratio=b['instructions']/a['instructions'],
                  decision='CPU resource screen passes;dynamic work,exact bytes and latency unproven'if clean else'CPU reject;no GPU',
                  limitation='Specialized source/compiler evidence only;lower static count or shared allocation does not prove exposed time saved.')
    (OUT/'cpu-audit.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in record.items()if k!='resources'},indent=2))


if __name__=='__main__':main()
