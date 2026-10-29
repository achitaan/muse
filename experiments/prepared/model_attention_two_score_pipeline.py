"""Conservative event/resource model for one L8 two-score-bank hypothesis.

No GPU code is generated, imported or executed. Pending WGMMA groups complete
only when a documented wait guarantees it; this does not assume lucky latency.
"""
import hashlib
import json
from pathlib import Path
import random

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PARENT='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'


class Hazard(AssertionError):pass


def require(value,reason):
    if not value:raise Hazard(reason)


class Head:
    def __init__(self,tiles,mutant):
        self.tiles=tiles;self.mutant=mutant;self.i=0
        self.phase='init' if tiles else 'done'
        self.pending=[];self.complete=set();self.banks=[None,None]
        self.consumed=set();self.raw_v_read=set();self.sf=[];self.pv=[]
        self.maximum_groups=0

    def issue_qk(self,tile):
        bank=0 if self.mutant=='one_score_bank' else tile%2
        previous=self.banks[bank]
        require(previous is None or previous in self.consumed,'score bank overwritten before scalar consumption')
        self.banks[bank]=tile;self.pending.append(('qk',tile))
        self.maximum_groups=max(self.maximum_groups,len(self.pending))
        require(len(self.pending)<=2,'more than two committed WGMMA groups')

    def wait(self,n):
        while len(self.pending)>n:
            group=self.pending.pop(0);self.complete.add(group)
            if group[0]=='pv':
                require(group[1]==len(self.pv),'PV updates changed numerical order')
                self.pv.append(group[1])


def simulate(tiles,patches,seed,mutant=None,keep_trace=False):
    rng=random.Random(seed);heads=[Head(tiles,mutant),Head(tiles,mutant)]
    full=[None]*4;released={};requests={};patched=set();producer=0
    trace=[];events=0

    def can_prepare(tile,h):
        if full[tile%4]!=tile:return False
        if tile in patches:
            req=requests.get(tile,set())
            return h not in req or len(req)==2
        return True

    def prepare(tile,h):
        require(full[tile%4]==tile,'QK reads unready stage generation')
        if tile in patches:
            req=requests.setdefault(tile,set())
            if h not in req:
                req.add(h)
                if len(req)==2:patched.add(tile)
                return False
            require(tile in patched,'query/new-key consumer barrier skipped')
        return True

    while producer<tiles or any(h.phase!='done' for h in heads):
        allowed=[]
        if producer<tiles and (producer<4 or released.get(producer-4,set())=={0,1}):allowed.append('producer')
        for h,H in enumerate(heads):
            if H.phase=='done':continue
            target=0 if H.phase=='init' else H.i+1
            if H.phase in ('init','next'):
                if can_prepare(target,h):allowed.append(h)
            else:allowed.append(h)
        require(allowed,'producer/consumer dependency deadlock')
        actor=rng.choice(allowed);events+=1
        require(events<100*max(tiles,1)+100,'event model failed to make bounded progress')
        if actor=='producer':
            if producer>=4:
                require(full[producer%4]==producer-4,'stage generation skipped')
                require(released.get(producer-4,set())=={0,1},'producer overwrites live K/V stage')
            full[producer%4]=producer
            if keep_trace:trace.append(['produce',producer])
            producer+=1;continue
        h=actor;H=heads[h];i=H.i;phase=H.phase
        if keep_trace:trace.append([h,i,phase,list(H.pending)])
        if phase=='init':
            if prepare(0,h):H.issue_qk(0);H.phase='score'
        elif phase=='score':
            H.wait(2 if mutant=='current_wait_too_weak' else (0 if i==0 else 1))
            require(('qk',i) in H.complete,'score accumulator read before QK completion')
            H.phase='next' if i+1<tiles else 'transpose'
        elif phase=='next':
            if prepare(i+1,h):H.issue_qk(i+1);H.phase='transpose'
        elif phase=='transpose':
            require(not any(kind=='pv' and tile%2==i%2 for kind,tile in H.pending),'V transpose overwrites live PV input')
            H.raw_v_read.add(i);H.phase='pv_wait'
        elif phase=='pv_wait':
            if mutant!='skip_previous_PV_wait':H.wait(1 if i+1<tiles else 0)
            require(i==0 or ('pv',i-1) in H.complete,'P/O/lacc accessed before previous PV completes')
            H.phase='release'
        elif phase=='release':
            require(('qk',i) in H.complete and i in H.raw_v_read,'raw stage released before K/V readers finish')
            released.setdefault(i,set()).add(h);H.phase='softmax'
        elif phase=='softmax':
            require(('qk',i) in H.complete,'softmax consumes unready scores')
            require(not any(kind=='pv' for kind,t in H.pending),'P registers or O/lacc overwritten while PV pending')
            H.consumed.add(i);H.sf.append(i);H.phase='issue_pv'
        elif phase=='issue_pv':
            require(i in H.raw_v_read and i in H.consumed,'PV missing V/softmax data')
            H.pending.append(('pv',i));H.maximum_groups=max(H.maximum_groups,len(H.pending))
            require(len(H.pending)<=2,'too many WGMMA groups')
            H.phase='advance'
        elif phase=='advance':
            H.i+=1
            if H.i==tiles:H.phase='drain'
            else:H.phase='score'
        elif phase=='drain':
            if mutant!='missing_tail_drain':H.wait(0)
            require(not H.pending,'final numerator/row sums read before tail PV completion')
            require(H.sf==H.pv==list(range(tiles)),'missing/reordered numerical tile work')
            H.phase='done'
        else:raise AssertionError(phase)
    require(all(released.get(t)=={0,1} for t in range(tiles)),'stage not released exactly by both heads')
    return dict(tiles=tiles,patch_tiles=sorted(patches),seed=seed,events=events,
                maximum_committed_groups=max(h.maximum_groups for h in heads),trace=trace if keep_trace else None)


def resource_contract(initial,consumer=152,producer=32):
    # Per-CTA pool, not the total unused SM register file. All sizes are GPRs.
    pool=initial*384;need=256*consumer+128*producer
    require(initial>=producer,'producer dec target exceeds initial register limit')
    require(initial<=consumer,'consumer inc target is below initial register limit')
    require(pool>=need,'CTA register pool cannot satisfy both consumers')
    require(need<=65536 and initial*384<=65536,'SM register capacity exceeded')
    return dict(initial_registers=initial,CTA_pool_registers=pool,redistributed_registers=need,
                unused_CTA_pool=pool-need)


def main():
    assert hashlib.sha256((ROOT/'best/attention_checked.py').read_bytes()).hexdigest()==PARENT
    shapes=[]
    counts=(0,1,2,3,4,5,7,8,15,31,64,127,251,1024)
    for n in counts:
        patch_sets={tuple(sorted({0} if n else set())),tuple(sorted({0,n-1} if n else set())),
                    tuple(sorted({0,max(0,n-2),n-1} if n else set()))}
        for patches in sorted(patch_sets):
            for seed in range(12):shapes.append(simulate(n,set(patches),20261029+seed))
    # A large supported-context arithmetic stress case, not an actual GPU shape.
    shapes.append(simulate(8192,{0,8190,8191},131313))
    rejected=[]
    for mutant in ('one_score_bank','current_wait_too_weak','skip_previous_PV_wait','missing_tail_drain'):
        try:simulate(7,{0,5,6},123,mutant)
        except Hazard as exc:rejected.append(dict(mutant=mutant,reason=str(exc)))
        else:raise AssertionError('Unsafe pipeline mutation accepted: '+mutant)
    budgets=[resource_contract(n) for n in (112,120,128,136,144,152)]
    for name,initial,consumer in [('old80_pool_for_new152',80,152),('inc_below_initial',160,152),('unproven232_pool',152,232)]:
        try:resource_contract(initial,consumer)
        except Hazard as exc:rejected.append(dict(mutant=name,reason=str(exc)))
        else:raise AssertionError('Unsafe resource mutation accepted: '+name)
    report=dict(parent_source_sha256=PARENT,CPU_only=True,GPU_jobs=0,CUDA_prototype_written=False,
        event_cases=len(shapes),maximum_modeled_tiles=8192,max_committed_groups=max(s['maximum_committed_groups']for s in shapes),
        two_consumer_warpgroups=True,one_unchanged_128thread_producer=True,producer_stage_slots=4,score_register_banks=2,
        event_case_records=[{k:v for k,v in row.items()if k!='trace'} for row in shapes],
        rejected_hazards=rejected,source_order_trace=simulate(3,{0,2},42,keep_trace=True),
        fixed_resource_plan=dict(threads=384,producer_threads=128,consumer_threads=256,producer_target_registers=32,
            consumer_target_registers=152,score_registers_per_consumer=64,old_score_registers_per_consumer=32,
            initial_native_register_range=[112,152],register_pool_is_CTA_local=True,shared_bytes=114688,
            expected_active_blocks_per_SM=1,total_grid_blocks_at_most_SM_count=True,
            global_split_count='floor(SM_count/KV_pairs); update Params.n_splits and its magic division metadata'),
        feasible_native_initial_register_contracts=budgets,
        intended_scope='L8 full multi-request flat/nc1 only; retained global cooperative finalizer; no external-finalizer reuse',
        numerical_schedule='Per-head QK results consumed in tile order; PV updates stay strictly in tile order; no arithmetic or precision change modeled',
        unresolved=['Actual compiler must preserve two distinct banks and partial WGMMA waits without C751x serialization',
                    'Native register query must satisfy the CTA-local pool bounds; launch_bounds maximum alone is not proof',
                    'Changing grid changes split boundaries and FP32/FP8 rounding; full official accuracy and fresh graph replay remain required',
                    'No measured score/softmax critical-path time or speedup prediction; no GPU admission yet'])
    out=ROOT/'results/attention-two-score-pipeline-model-20261029';out.mkdir(exist_ok=True)
    data=(json.dumps(report,indent=2)+'\n').encode();path=out/'model.json';assert not path.exists()or path.read_bytes()==data;path.write_bytes(data)
    print(json.dumps({k:v for k,v in report.items()if k not in ('event_case_records','source_order_trace')},indent=2))


if __name__=='__main__':main()
