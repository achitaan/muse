"""Retain only the H200-qualified M1 mechanism; preserve prior frozen drafts."""

import ast
import hashlib
import json

from pdl16_sources import HERE, ROOT, BASE_SHA

OUT = ROOT / 'results/decode-moe-plan-20261030/pdl-norm-one'
PARENT_SHA = '85a0ae65ee872d63fc9ca707883b9c9a5d2d1c9c8c23808ae4bab6578b7d818f'

UNIT = '''
def _pdls1_unit_ab(label, fns, mutate, dev, start):
    if not _okh_budget(start, 274.):
        return False
    if not _pdli_probe(fns, mutate, label=label):
        print('PDLS1_UNIT_BYTES', label, False, flush=True)
        return False
    print('PDLS1_UNIT_BYTES', label, True, flush=True)
    mutate(0)
    passes = []
    for order in (('parent', 'pdl'), ('pdl', 'parent')):
        if not _okh_budget(start, 278.):
            return False
        times = _okh_time({key: fns[key] for key in order}, dev, reps=25)
        saving = times['parent'] - times['pdl']
        ok = _okh_pick(times, 'parent', .97) == 'pdl' and saving >= 1.5
        passes.append(ok)
        print('PDLS1_UNIT_TIME', label, order, times, 'saving_us=', saving, 'passes=', ok, flush=True)
    return len(passes) == 2 and all(passes)
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = (HERE / 'decoder_pdls_draft.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    src = raw.decode(); tree = ast.parse(src)
    funcs = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    segments = {name:ast.get_source_segment(src, funcs[name])
                for name in ['_pdls_select', '_pdls_boundary', 'kernel_fn']}
    changes = {}
    for name in ['_pdls_select', '_pdls_boundary', 'kernel_fn']:
        old = segments[name]; new = old
        if name == '_pdls_select':
            edits = [
                ('for m in (1, 8):', 'for m in (1,):'),
                ("if not _pdli_ab('PDLS_norm_r2f_M'", "if not _pdls1_unit_ab('PDLS_norm_r2f_M'"),
                ('ao.normal_(generator=gen); stream.normal_(generator=gen)',
                 'ao.normal_(generator=gen); stream.normal_(generator=gen)\n'
                 '            if seed == 1:\n'
                 '                ao.zero_(); stream.zero_()\n'
                 '            elif seed == 2:\n'
                 '                ao.mul_(16.); stream.mul_(.001)'),
            ]
        elif name == '_pdls_boundary':
            edits = [('branch.shape[0] not in (1, 8)', 'branch.shape[0] != 1')]
        else:
            edits = [("cfg['q_len'] in (1, 8)", "cfg['q_len'] == 1")]
        for a,b in edits:
            assert new.count(a)==1,(name,a,new.count(a));new=new.replace(a,b)
        assert src.count(old)==1;src=src.replace(old,new)
        changes[name]=dict(old=hashlib.sha256(old.encode()).hexdigest(),new=hashlib.sha256(new.encode()).hexdigest())
    blob=src.encode()+UNIT.encode();ast.parse(blob)
    scored=(HERE/'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(scored).hexdigest()==BASE_SHA and blob.startswith(scored)and len(blob)<1_000_000
    (HERE/'decoder_pdls1_draft.py').write_bytes(blob)
    result=dict(candidate_sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),
        headroom=1_000_000-len(blob),scored_parent_sha256=BASE_SHA,
        previous_CPU_draft_sha256=PARENT_SHA,changed_functions=changes,
        mechanism_H200_job='d34f5724621b42cf',mechanism_CUDA_sha256='f24c9d33e6be7f433a8c9629019f66f250b569f4cd235fd80b098499c399795e',
        checked_scope='Norm/R2f M1 unit only; candidate source and whole layer unrun.',
        M8='excluded after predeclared absolute timing gate failed; parent fallback',
        unit_gate='finite byte equality across random/zero/outlier replays; >3% and >=1.5us both cold-L2 orders',
        whole_layer_gate='unchanged: complete output/cache bytes and >3% both orders',
        GPU_jobs_for_this_source=0,GPU_admitted=False)
    (OUT/'source-binding.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
