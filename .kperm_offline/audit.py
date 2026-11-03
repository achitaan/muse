"""Scope/pipeline audit and a focused reference-vs-base inspection diff."""
import ast,collections,difflib,hashlib,importlib.util,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.kperm_offline'
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=load(ROOT/'experiments/boards/moe/moe_b15.py','base_audit');port=load(ROOT/'experiments/boards/moe/moe_b16.py','port_audit')
a,b=base.CUDA_SRC,port.CUDA_SRC
ref=(OUT/'reference.cu').read_text()
def section(s,name):
    i=s.index(name);return s[i:s.index('\n}',i)+2]
# Explicit invariants: don't import the reference's other numerical/runtime edits.
for name in ('struct Params {','struct Smem {','void route(', 'void wgs_wproducer(', 'void wgs_xproducer(',
             'void wgs_epi_tile(', 'void wgs_flush(', '__device__ __noinline__ void wgs_ranges(', 'float2 achita_marlin_out2(',
             'void dq4(', 'void wgs_phase(', 'int wgs_cost('):
    assert section(a,name)==section(b,name),name
assert a[a.index('template <int NTT> struct WGSMMA;'):a.index('// two producer warps:')]==b[b.index('template <int NTT> struct WGSMMA;'):b.index('// two producer warps:')]
assert a[a.index('template <bool BIG>\n__global__'):]==b[b.index('template <bool BIG>\n__global__'):]
assert 'if (nj == 0)' in section(b,'__device__ __noinline__ void wgs_ranges(')
assert '#define OKC_FINREG 1' in b and '#define OKC_WGSERL 1' in b
for name in ('MAIN','TINY','TINY2','MID','MAIN_FALLBACK'):assert getattr(base,name+'_DEFINES')==getattr(port,name+'_DEFINES')
assert base.CPP_SRC==port.CPP_SRC and port.KPERM==3
for helper in ('_pack_codes','_pack_codes_nat','_pack_scales','_fat_order','_unit_order','kernel_fn','_moe_select'):
    def defs(p):
        t=p.read_text();return [ast.get_source_segment(t,n) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name==helper]
    assert defs(ROOT/'experiments/boards/moe/moe_b15.py')==defs(ROOT/'experiments/boards/moe/moe_b16.py'),helper
diff=[]
for name in ('void wgs_gather(','void wgs_seg('):
    diff += list(difflib.unified_diff(section(a,name).splitlines(True),section(ref,name).splitlines(True),fromfile='our_b15/'+name,tofile='read_only_alygg/'+name))
(OUT/'reference_changes.diff').write_text(''.join(diff))
(OUT/'our_changes.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='our_moe_b15.cu',tofile='our_moe_b16.cu')))
print('PASS HINT/ERL/FINREG/J0, routing, numerical boundaries, producer/epilogue/finalizer/host ABI preserved')
if '--sass' not in sys.argv:raise SystemExit(0)
def functions(name):
    s=(OUT/(name+'.sass')).read_text()
    return {p.splitlines()[0].strip():[(int(pc,16),v.strip()) for pc,v in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]+?;)\s+/\*',p)] for p in s.split('Function : ')[1:]}
assert functions('BASE')==functions('OFF'),'macro0 device code changed'
anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
compiled=b.split('#include <torch/extension.h>')[0].replace(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S,int pb,int s,int G);\n'+anchor)
assert compiled==(OUT/'KPERM3.cu').read_text(),'final CUDA differs from compiled preflight'
for k in ('TINY','TINY2','MID','MAIN_FALLBACK'):
    assert functions(k+'_BASE')==functions(k+'_B16'),k
records=[]
for label in ('BASE','KPERM3'):
    for fn,ins in functions(label).items():
        # A continuing-stage loop contains16 same-N MMA instructions. Deduplicate
        # alternate backedges around the same MMA sites, retaining widest interval.
        loops={}
        for pc,v in ins:
            m=re.search(r'\bBRA(?:\.\w+)*\s+(?:(?:P\d+|UR\d+),\s*)?(0x[0-9a-f]+)',v)
            if not m or int(m[1],16)>=pc:continue
            lo=int(m[1],16);seq=[(p,x) for p,x in ins if lo<=p<=pc]
            mma=[(p,int(m[1])) for p,x in seq if (m:=re.search(r'HGMMA\.64x(\d+)x16',x))]
            if len(mma)!=16 or len({n for _,n in mma})!=1:continue
            key=tuple(p for p,_ in mma)
            rec=dict(N=mma[0][1],start=hex(lo),end=hex(pc),instructions=len(seq),
                LDS64=sum(bool(re.search(r'\bLDS\.64\b',x)) for _,x in seq),
                PRMT=sum(bool(re.search(r'\bPRMT\b',x)) for _,x in seq),
                local=[dict(pc=hex(p),ins=x) for p,x in seq if re.search(r'\b(?:LDL|STL)\b',x)])
            if key not in loops or len(seq)>loops[key]['instructions']:loops[key]=rec
        assert collections.Counter(x['N'] for x in loops.values())==collections.Counter({n:2 for n in range(8,65,8)})
        print(label,fn,'steady-stage (N,LDS64,PRMT,local)',[(x['N'],x['LDS64'],x['PRMT'],len(x['local'])) for x in loops.values()])
        records.append(dict(build=label,function=fn,loops=list(loops.values())))
    sas=(OUT/(label+'.sass')).read_text()
    assert len(re.findall(r'\bHGMMA\.',sas))==1024
    assert sas.count('WARPGROUP.DEPBAR.LE gsb0, 0x1 ;')==512
    assert sas.count('WARPGROUP.DEPBAR.LE gsb0, 0x0 ;')==32
assert 'C7512' not in (OUT/'KPERM3.log').read_text()
(OUT/'sass_audit.json').write_text(json.dumps(records,indent=2))
print('PASS OFF/non-MAIN SASS identity, original async MMA inventory, no C7512')
