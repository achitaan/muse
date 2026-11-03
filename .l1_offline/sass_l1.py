"""Read-only SASS evidence: dynamic role budgets and packed-stage release order."""
from pathlib import Path
import collections,json,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.l1_offline'
def functions(name):
    src=(OUT/(name+'.sass')).read_text()
    return {p.splitlines()[0].strip():[(int(pc,16),ins.strip()) for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]+?;)\s+/\*',p)] for p in src.split('Function : ')[1:]}
base=functions('BASE');l1=functions('L1');reports=[]
for fname,ins in l1.items():
    tab=dict(ins);pcs=sorted(tab);nextpc={pc:pcs[i+1] for i,pc in enumerate(pcs[:-1])}
    dec=[pc for pc,s in ins if 'USETMAXREG.DEALLOC.CTAPOOL 0x20' in s]
    inc=[pc for pc,s in ins if 'USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0x70' in s]
    assert len(dec)==len(inc)==1
    def reachable(start):
        seen=set();todo=[start]
        while todo:
            pc=todo.pop()
            if pc in seen:continue
            assert pc in tab,hex(pc)
            seen.add(pc);s=tab[pc];targets=[];fall=True
            if re.search(r'\bBRA\b',s):
                m=re.search(r'\bBRA(?:\.\w+)*\s+(?:(?:P\d+|UR\d+),\s*)?(0x[0-9a-f]+)',s)
                assert m,s
                targets.append(int(m[1],16))
                fall=not bool(re.fullmatch(r'BRA\s+0x[0-9a-f]+\s*;',s))
            elif 'BSSY ' in s:
                targets.append(int(re.search(r'0x[0-9a-f]+',s)[0],16))
            elif 'CALL.' in s:
                m=re.search(r'CALL\.\S+\s+(0x[0-9a-f]+)',s);assert m,s
                targets.append(int(m[1],16))
            elif re.search(r'\b(?:EXIT|RET\.)',s):fall=s.startswith('@')
            if fall and pc in nextpc:targets.append(nextpc[pc])
            todo+=targets
        return seen
    producer=reachable(dec[0]);consumer=reachable(inc[0])
    def maxreg(path):
        # Explicit operand registers, plus wide LDS/STG vector footprints.
        regs=[]
        for pc in path:
            s=tab[pc];regs += [int(x) for x in re.findall(r'\bR(\d+)\b',s)]
            m=re.search(r'\b(?:LDS|LDG|LDC)(?:\.E)?\.(128|64)\s+R(\d+)',s)
            if m:regs.append(int(m[2])+int(m[1])//32-1)
            m=re.search(r'\b(?:STS|STG)(?:\.E)?\.(128|64)\s+.+,\s*R(\d+)\s*;',s)
            if m:regs.append(int(m[2])+int(m[1])//32-1)
            m=re.search(r'\bHGMMA\.64x(\d+)x16\.\S+\s+R(\d+),\s*R(\d+)',s)
            if m:regs.extend((int(m[2])+int(m[1])//2-1,int(m[3])+3))
        return max(regs,default=-1)
    assert not any(re.search(r'\b(?:LDL|STL)\b',s) for _,s in ins)
    assert inc[0] not in producer and dec[0] not in consumer
    pr,cr=maxreg(producer),maxreg(consumer)
    assert pr<32 and cr<112,(pr,cr)
    assert '0x2, 0x280' in '\n'.join(tab[x] for x in producer)
    assert '0x7, 0x220' in '\n'.join(tab[x] for x in producer)
    # Each real per-lane mask WARPSYNC marks the captured-stage release. There
    # must be4 LDS.128 +4 LDS.64 in its straight-line load window, then the
    # empty arrival, then the x full wait, with no intervening decode or LDS.
    capture=[]
    for i,(pc,s) in enumerate(ins):
        if not re.search(r'\bWARPSYNC R\d+',s):continue
        before=[x for _,x in ins[max(0,i-95):i]]
        lastwait=max(j for j,x in enumerate(before) if 'SYNCS.PHASECHK' in x)
        window=before[lastwait+1:]
        assert sum('LDS.128' in x for x in window)==4,(hex(pc),'codes')
        assert sum('LDS.64' in x for x in window)==4,(hex(pc),'scales')
        assert sum('LOP3.LUT' in x for x in window)>=8,(hex(pc),'mask dependency')
        after=ins[i+1:i+15]
        arr=next(j for j,(_,x) in enumerate(after) if 'SYNCS.ARRIVE' in x)
        wait=next(j for j,(_,x) in enumerate(after) if 'SYNCS.PHASECHK' in x)
        assert arr<wait and not any('LDS' in x or 'HGMMA' in x for _,x in after[:wait])
        capture.append(dict(sync=hex(pc),release=hex(after[arr][0]),xwait=hex(after[wait][0])))
    assert len(capture)==32,(fname,len(capture)) # PB2 * NTT8 * first/continuing2
    def mmaops(seq):return collections.Counter(re.findall(r'\bHGMMA\.[\w.]+','\n'.join(s for _,s in seq)))
    assert mmaops(ins)==mmaops(base[fname]),'WGMMA shape/count inventory changed'
    reports.append(dict(function=fname,launch_regs=96,consumer_limit=112,producer_limit=32,
        producer_max_operand_R=pr,consumer_max_operand_R=cr,producer_instructions=len(producer),
        consumer_instructions=len(consumer),capture_sites=capture,local_load_store_instructions=0))
    print(fname,'producer max R',pr,'consumer max R',cr,'capture sites',len(capture),'no local loads/stores')
(OUT/'sass_evidence.json').write_text(json.dumps(reports,indent=2))
print('PASS role CFG/register limits,32 early capture sites per instantiation, and WGMMA shape/count inventory')
