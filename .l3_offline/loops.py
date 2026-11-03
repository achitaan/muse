"""Audit complete straight-line steady-stage loop regions, not just MMA intervals."""
from pathlib import Path
import collections,json,re,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.l3_offline'
reports=[]
for name in (sys.argv[1:] or ('BASE','L2','RUNTIME96','TWO96','OVER112')):
    for p in (OUT/(name+'.sass')).read_text().split('Function : ')[1:]:
        fn=p.splitlines()[0].strip()
        ins=[(int(pc,16),s.strip()) for pc,s in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]+?;)\s+/\*',p)]
        candidates={}
        for pc,s in ins:
            m=re.search(r'\bBRA(?:\.\w+)*\s+(?:(?:P\d+|UR\d+),\s*)?(0x[0-9a-f]+)',s)
            if not m:continue
            lo=int(m[1],16)
            if lo>=pc:continue
            seq=[(a,b) for a,b in ins if lo<=a<=pc]
            sites=[(a,int(m[1])) for a,b in seq if (m:=re.search(r'HGMMA\.64x(\d+)x16',b))]
            mm=[n for _,n in sites]
            if len(mm)!=16 or len(set(mm))!=1:continue
            spills=[dict(pc=hex(a),ins=b) for a,b in seq if re.search(r'\b(?:LDL|STL)\b',b)]
            key=tuple(a for a,n in sites)
            span=dict(N=mm[0],start=hex(lo),end=hex(pc),instructions=len(seq),local=spills)
            # Compiler may produce multiple backedges around the SAME16 MMA
            # sites. Keep the enclosing interval rather than double-counting it.
            if key not in candidates or len(seq)>candidates[key]['instructions']:candidates[key]=span
        spans=list(candidates.values())
        # Two steady-stage copies (A/B) per N specialization are expected.
        counts=collections.Counter(s['N'] for s in spans)
        assert counts and all(n==2 for n in counts.values()),(name,fn,counts)
        reports.append(dict(build=name,function=fn,loops=spans))
        print(name,fn,{n:[len(s['local']) for s in spans if s['N']==n] for n in sorted(counts)},flush=True)
(OUT/('loops_'+('_'.join(sys.argv[1:]) or 'all')+'.json')).write_text(json.dumps(reports,indent=2))
