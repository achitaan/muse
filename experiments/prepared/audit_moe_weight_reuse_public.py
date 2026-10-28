"""Logical reuse opportunity using independent draws of the public formula."""
import json
from pathlib import Path
import numpy as np

root = next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
rng = np.random.default_rng(20261028)
summary = []
for m in [1,8,16,32,128,256]:
    stats = []
    for _ in range(64):
        scores = rng.random((m,256))
        z, u = rng.standard_normal(256), np.maximum(rng.random(256),1e-12)
        scores += .03*z/np.sqrt(-np.log(u))
        ids = np.argpartition(scores,-8,axis=1)[:,-8:]
        ranks = np.bincount((ids//32).ravel(),minlength=8)
        ids = (ids-32*np.argmax(ranks))%256
        counts = np.bincount(ids.ravel(),minlength=256)[:32]
        active, jobs = np.count_nonzero(counts), np.sum((counts+63)//64)
        stats.append((int(active),int(jobs),int(counts.max()),float(jobs/active if active else 1)))
    a = np.array(stats)
    summary.append({'m':m,'draws':64,'draws_with_repeated_expert_weights':int(np.sum(a[:,1]>a[:,0])),
        'mean_active_experts':float(a[:,0].mean()),'mean_wgs_jobs':float(a[:,1].mean()),
        'mean_job_over_unique_weight_loads':float(a[:,3].mean()),'largest_expert_count':int(a[:,2].max())})
record = {'seed':20261028,'numpy_version':np.__version__,
          'method':'Own NumPy routing draws using the public reference popularity formula and busiest-rank rotation, not official inputs/seeds; torch RNG/results will differ',
          'summary':summary,'interpretation':'Logical weight-source repetition only; no measured L2 hits or speedup prediction'}
(root/'.tools/moe-weight-reuse-public-model.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
