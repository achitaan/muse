import json, sys
sys.stdout.reconfigure(encoding='utf-8')
j = json.load(open(sys.argv[1], encoding='utf-8-sig'))
r = j.get('result') or {}
print(j.get('board'), r.get('status'), r.get('passed'), r.get('speedup'), r.get('speedup_ci'), str(r.get('error'))[:300])
pre = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else ()
for l in (r.get('log_tail') or '').split('\n'):
    if pre and l.startswith(pre):
        print(l[:240])
print(' '.join(f"{b['median_ns']/1000:.1f}" for b in r.get('records', []) if b.get('event') == 'benchmark'))
