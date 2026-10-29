"""Keep required compact evidence together within the service's bounded tail."""
import ast
import hashlib
import json
from pathlib import Path
from preflight_decoder_cluster_norm import ROOT,freeze

DRAFT=ROOT/'results/dense-cluster-resource-draft-20261029'
OUT=ROOT/'results/dense-cluster-resources-preflight-20261029'


def main():
    raw=(DRAFT/'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='7de2abc394419b2c80a866765dbd6b6a855828b8ab1aedbb0d35e7f862221139'
    source=raw.decode()
    line="    print('DENSE_CLUSTER_RESOURCES '+_lg_json.dumps(resources,separators=(',',':')),flush=True)\n"
    anchor="    print('DENSE_CLUSTER_EVIDENCE', _lg_json.dumps(evidence, separators=(',', ':')), flush=True)"
    assert source.count(line)==source.count(anchor)==1
    source=source.replace(line,'');source=source.replace(anchor,line+anchor)
    recovered=source.replace(line+anchor,anchor)
    entry='    resources=[ext.lg_info(kind) for kind in range(2)]\n'
    assert recovered.count(entry)==1
    recovered=recovered.replace(entry,entry+line)
    assert recovered.encode()==raw
    old,new=ast.parse(raw),ast.parse(source)
    for name in ('_lg_key','_lg_run','_lg_input','_lg_verify','_lg_compare','kernel_fn'):
        assert [ast.dump(n) for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name]==[ast.dump(n) for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name]
    prior=json.loads((ROOT/'results/dense-cluster-nearby-preflight-20261029/nearby-selection-audit.json').read_bytes())
    bound=prior['diagnostic_bytes_bound']+len('DENSE_CLUSTER_RESOURCES ')+2*(13*26+14)+3+100
    assert bound<15000
    meta=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),previous_draft_sha256=hashlib.sha256(raw).hexdigest(),
        only_resource_marker_moved_to_compact_tail=True,removal_line=line,insertion_anchor=anchor,original_entry=entry,
        inverse_restores_previous_draft=True,all_timed_AST_unchanged=True,CUDA_unchanged=True,
        diagnostic_compact_bytes_bound=bound,CPU_only=True,GPU_jobs=0)
    freeze(OUT/'submission.py',source.encode())
    freeze(OUT/'source-diagnostic-binding.json',(json.dumps(meta,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in meta.items() if k not in ('removal_line','insertion_anchor','original_entry')},indent=2))


if __name__=='__main__':main()
