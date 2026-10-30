"""Locate the standalone/Decode MoE source differences without GPU execution."""

import ast
import base64
import hashlib
import json
import zlib

from pdl16_sources import HERE, ROOT

OUT=ROOT/'results/decode-moe-plan-20261030/moe-repeat-v2'


def erase_body(text,name):
    start=text.index('void '+name+'(');left=text.index('{',start)
    i=left;depth=0;mode='code'
    while i<len(text):
        ch=text[i];pair=text[i:i+2]
        if mode in ('"',"'"):
            if ch=='\\':i+=2;continue
            if ch==mode:mode='code'
        elif mode=='line':
            if ch=='\n':mode='code'
        elif mode=='comment':
            if pair=='*/':mode='code';i+=2;continue
        elif pair=='//':mode='line';i+=2;continue
        elif pair=='/*':mode='comment';i+=2;continue
        elif ch in ('"',"'"):mode=ch
        elif ch=='{':depth+=1
        elif ch=='}':
            depth-=1
            if depth==0:return text[:left]+'{ /* WGS-only body excluded */ }'+text[i+1:]
        i+=1
    raise AssertionError(name)


def main():
    sources={};flags={};payloads={}
    for label,path,prefix in [('standalone',ROOT/'best/moe_verified.py',''),
                              ('decode',HERE/'decoder_pdli.py','_native_moe_')]:
        raw=path.read_bytes();tree=ast.parse(raw);values={n.targets[0].id:n.value for n in tree.body
            if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)}
        node=values[prefix+'CUDA_SRC']
        if isinstance(node,ast.Constant):src=node.value
        else:
            encoded=next(n.value for n in ast.walk(node)if isinstance(n,ast.Constant)
                and isinstance(n.value,str)and len(n.value)>10000)
            src=zlib.decompress(base64.b64decode(encoded)).decode()
        sources[label]=src;payloads[label]=hashlib.sha256(raw).hexdigest()
        flags[label]={k:ast.literal_eval(values[prefix+k])for k in ['TINY_DEFINES','TINY2_DEFINES','MID_DEFINES']}
    assert flags['standalone']==flags['decode']
    reduced={}
    for label,text in sources.items():
        for name in ['wgs_flush','wgs_seg']:text=erase_body(text,name)
        reduced[label]=text
    assert reduced['standalone']==reduced['decode']
    assert '#ifndef OKC_WGS\n#define OKC_WGS 0' in sources['standalone']
    result=dict(GPU_jobs=0,payload_SHA256=payloads,
        CUDA_SHA256={k:hashlib.sha256(v.encode()).hexdigest()for k,v in sources.items()},
        sole_differing_bodies=['wgs_flush','wgs_seg'],
        all_other_CUDA_source_identical=True,TINY_TINY2_MID_defines_identical=True,
        default_WGS=0,non_WGS_atomic_norm_sites_identical=True,
        inference='The FINREG source difference concerns WGS work. Tiny-path atomic norm accumulation remains present in Decode; standalone variability is relevant to its byte-gate risk.',
        limitation='Static source bridge only. No Decode H200 repeat experiment, measured variance rate, actual compiler/preprocessor equality or whole-layer numerical result is established.')
    (OUT/'decode-source-bridge.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
