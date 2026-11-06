"""Extract source-bound component features without assigning serving concurrency.

Block ranges describe observed run variation, NOT a confidence/prediction interval.
No replays are treated as independent experiments and no routing patterns are pooled.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVING = HERE.parent


def summarize(rows, arm):
    groups = {}
    for line, row in rows:
        if row.get('event') != 'timing':
            continue
        key = (row['tokens'], row['pattern'], row['cache'], row['offset'])
        groups.setdefault(key, []).append((line, row))
    output = []
    for (tokens, pattern, cache, offset), group in sorted(groups.items()):
        blocks = [r['block'] for _, r in group]
        if len(set(blocks)) != len(blocks):
            raise ValueError('duplicate block: do not treat repeated retrievals as new data')
        ratios, savings, controls = [], [], []
        for _, row in group:
            m = row['medians_us']
            if not all(type(m[a]) in (float, int) and math.isfinite(m[a]) and m[a]>0
                       for a in ('stock_a','stock_b',arm)):
                raise ValueError('invalid observed timing')
            ratios.append(m[arm]/m['stock_a'])
            savings.append(m['stock_a']-m[arm])
            controls.append(abs(m['stock_b']/m['stock_a']-1))
        output.append(dict(tokens=tokens, pattern=pattern, cache=cache, ep_offset=offset,
                           blocks=blocks, source_lines=[line for line, _ in group],
                           stage_new_old_ratio=[min(ratios),max(ratios)],
                           saved_us=[min(savings),max(savings)],
                           max_control_fraction=max(controls),
                           range_kind='observed block range, not a predictive interval',
                           path_scope='complete local expert operation after pre-expert norm',
                           serving_concurrency=None, independent_sessions=1))
    return output


def extract():
    pins=json.loads((SERVING/'PRECOMP_SESSION.json').read_text())
    binding=json.loads((SERVING/'PRECOMP_SOURCE_HASHES.json').read_text())
    output=[]
    for candidate, filename, arm, patch in (
        ('s1','s1-paired-v2.jsonl','s1','s1_postnorm_v2.patch.diff'),
        ('native-v3','native-release-paired.jsonl','native','moe_direct_marlin.experimental.patch.diff')):
        path=SERVING/'results/precomp-20261105/results'/filename
        patch_path=SERVING/patch
        if hashlib.sha256(patch_path.read_bytes()).hexdigest()!=binding[patch]:
            raise ValueError('retained patch changed')
        rows=[(i,json.loads(line)) for i,line in enumerate(path.read_text().splitlines(),1) if line.startswith('{')]
        environment=next(r for _,r in rows if r.get('event')=='environment')
        if candidate=='s1' and environment['patch_sha256']!=binding[patch]:
            raise ValueError('S1 measurement/artifact mismatch')
        output.append(dict(candidate=candidate, schema_version=1, fidelity='integrated_component',
                           source=path.relative_to(SERVING).as_posix(),
                           source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           patch_sha256=binding[patch], sglang_revision=pins['sglang_commit'],
                           harness_sha256=binding['bench_precomp.py' if candidate=='s1' else 'bench_native_marlin.py'],
                           kernel_source_sha256=None if candidate=='s1' else binding['native_marlin_wgs.cu'],
                           image=pins['image'], environment=environment,
                           environment_missing=['cuda_version_in_this_log'] if 'cuda' not in environment else [],
                           session='precomp-20261105', workload='synthetic random weights and specified routes',
                           validity='pending_official_answer_gate' if candidate=='s1' else 'rejected',
                           observations=summarize(rows, arm)))
    return dict(serving_labels=0, component_sessions=1, families=output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    text=json.dumps(extract(),indent=2)+'\n'
    if args.output:
        args.output.write_text(text,encoding='utf8',newline='\n')
    else:
        print(text,end='')
