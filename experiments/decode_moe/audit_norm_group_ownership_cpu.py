"""Source-derived static WS range/bucket ownership, no GPU calls."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]
r=ROOT/'results/decode-moe-plan-20261030/norm-group'
source=ROOT/'experiments/decode_moe/moe_norm_group.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='e5c0e7ec4e28ccc2c9a91af85e41231a9ec356daaf0db13667c1560d6110b85b'
cases=0;empty=0
# Oracle: retained WS half-open unit ranges; RB_B=4096/16=256.
for jobs in range(33):
 for grid in (1,2,7,132,148,512):
  for count in (1,8):
   coverage=[0]*jobs
   for c in range(grid):
    a,b=jobs*256*c//grid,jobs*256*(c+1)//grid
    lo=(a//256)*8;hi=((b-1)//256+1)*8 if a<b else lo
    initialized=set(range(lo,hi));touched=set()
    for u in range(a,b):
     j=u//256;coverage[j]+=1
     for tk in range(count):
      assert j*8+tk in initialized;touched.add(j*8+tk)
    published={i for i in range(lo,hi)if i%8<count}
    assert published==touched
    if a==b:assert not initialized and not published;empty+=1
    if grid==132:assert hi-lo<=16
   assert coverage==[256]*jobs;cases+=1
assert empty>0
record=dict(cases=cases,empty_CTA_intervals=empty,all_unit_addends_covered_once=True,every_atomic_bucket_initialized=True,published_valid_buckets_equal_touched_valid_buckets=True,at_most16_initialized_slots_per_CTA_at_grid132=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),GPU_calls=0,limitation='Integer ownership model,not CUDA race,replay or performance evidence.')
(r/'ownership-cpu-audit.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record,indent=2))
