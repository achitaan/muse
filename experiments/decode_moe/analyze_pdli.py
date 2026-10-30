"""Read saved OKC results, reconstruct weighted loads, and preserve source-bound decisions."""
import argparse
import collections
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'results/decode-moe-plan-20261030'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def measurements(doc):
    result=doc['result']
    rows={}
    loads=collections.defaultdict(lambda:[0.,0.])
    for row in result.get('records',[]):
        if row['event']!='benchmark':continue
        case=row['case'];key=(case['load'],case['layer'])
        assert key not in rows
        base,ours=row['baseline_ns']/1000.,row['median_ns']/1000.
        w=case['weight'];load=case['load']
        loads[load][0]+=w*base;loads[load][1]+=w*ours
        rows[key]={'load':load,'layer':case['layer'],'tokens':case['batch']*case.get('q_len',1),
                   'baseline_us':base,'ours_us':ours,'weight':w,'status':row['status'],'flags':row.get('flags',[])}
    rebuilt=math.exp(sum(math.log(a/b) for a,b in loads.values())/len(loads)) if loads else None
    if rebuilt is not None:
        assert abs(rebuilt-result['speedup'])<1e-9,(rebuilt,result['speedup'])
    return rows,loads,rebuilt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('result',nargs='?',default=str(RESULTS/'pdli-status-final.json'))
    ap.add_argument('--parent',default=str(RESULTS/'pdl-status-final.json'));ap.add_argument('--out',default=str(RESULTS/'pdli-analysis.json'));args=ap.parse_args()
    doc,parent=read(args.result),read(args.parent)
    if doc.get('result') is None:
        print(json.dumps({'job_id':doc.get('id'),'status':doc.get('status')},indent=2));return
    result=doc['result'];rows,loads,score=measurements(doc);old,oldloads,oldscore=measurements(parent)
    tests=[x for x in result.get('records',[]) if x['event']=='test']
    setup=next((x for x in result.get('records',[]) if x['event']=='setup'),{})
    log=result.get('log_tail','')
    selections={m+'_'+kind:int(mask) for m,kind,mask in re.findall(r'PDLI_SELECT M(1|8)_(sliding|full) mask=\s*(\d+)',log)}
    comparisons=[];control_ratios=[]
    for key,row in rows.items():
        if key not in old:continue
        before=old[key]['ours_us'];after=row['ours_us']
        comparisons.append(dict(row,parent_us=before,saving_us=before-after,parent_over_new=before/after,
                                targeted=row['tokens'] in (1,8)))
        if row['tokens'] not in (1,8):control_ratios.append(after/before)
    drift=math.exp(sum(math.log(v) for v in control_ratios)/len(control_ratios)) if control_ratios else None
    for row in comparisons:
        row['control_normalized_parent_over_new']=row['parent_over_new']*drift if drift else None
    output={'job_id':doc.get('id'),'API_content_hash':doc.get('content_hash'),'status':doc.get('status'),
            'passed':result.get('passed'),'speedup':result.get('speedup'),'speedup_ci':result.get('speedup_ci'),
            'reconstructed_speedup':score,'parent_job_id':parent.get('id'),'parent_speedup':oldscore,
            'prepare_s':setup.get('prepare_s'),'test_count':len(tests),
            'tests_passed':sum(x['status']=='pass' for x in tests),
            'benchmarks_passed':sum(x['status']=='pass' for x in rows.values()),
            'checks':result.get('checks'),'flags':result.get('flags'),'error':result.get('error'),
            'selections':selections,'real_new_path_selected':any(selections.values()),
            'unit_logs':[l for l in log.splitlines() if l.startswith('PDLI_')],
            'loads':{k:{'baseline_weighted_us':a,'ours_weighted_us':b,'ratio':a/b} for k,(a,b) in loads.items()},
            'case_comparisons':comparisons,'control_new_over_parent_geomean':drift,
            'comparison_limit':'Cross-run comparisons and control normalization are drift diagnostics, not paired experiments or significance tests.',
            'publication_automatically_authorized':False}
    Path(args.out).write_bytes((json.dumps(output,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in output.items() if k not in ('unit_logs','case_comparisons','loads')},indent=2))
    for row in comparisons:
        print(row['load'],row['layer'],'%.3f -> %.3f us, saved %.3f us' % (row['parent_us'],row['ours_us'],row['saving_us']))
    for line in output['unit_logs']:print(line)


if __name__=='__main__':main()
