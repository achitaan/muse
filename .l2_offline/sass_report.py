"""Stage-local resource evidence, separate from ptxas whole-function totals."""
from pathlib import Path
import collections,json,re,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.l2_offline'
reports=[]
for name in (sys.argv[1:] or ['L2','NARROW','TWO_SLOT']):
    text=(OUT/(name+'.sass')).read_text()
    for part in text.split('Function : ')[1:]:
        fn=part.splitlines()[0].strip()
        code=re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]+?;)\s+/\*',part)
        byn=collections.defaultdict(list)
        for i,s in enumerate(code):
            m=re.search(r'HGMMA\.64x(\d+)x16',s)
            if m:byn[int(m[1])].append(i)
        stages={}
        for n,sites in byn.items():
            assert len(sites)%16==0,(name,n,len(sites))
            intervals=[]
            for k in range(0,len(sites),16):
                seq=code[sites[k]:sites[k+15]+1]
                intervals.append(dict(local_loads=sum(bool(re.search(r'\bLDL\b',s)) for s in seq),
                    local_stores=sum(bool(re.search(r'\bSTL\b',s)) for s in seq),
                    depbar0=sum('WARPGROUP.DEPBAR.LE' in s and '0x0' in s for s in seq)))
            stages[n]=intervals
        reports.append(dict(build=name,function=fn,stages=stages,
            total_LDL=sum(bool(re.search(r'\bLDL\b',s)) for s in code),
            total_STL=sum(bool(re.search(r'\bSTL\b',s)) for s in code)))
        print(name,fn, 'stage inner LDL/STL ranges',
              {n:(min(x['local_loads']+x['local_stores'] for x in v),max(x['local_loads']+x['local_stores'] for x in v)) for n,v in stages.items()})
(OUT/('sass_'+('_'.join(sys.argv[1:]) or 'final')+'.json')).write_text(json.dumps(reports,indent=2))
