import json, os, subprocess, sys, time
sys.stdout.reconfigure(encoding='utf-8')
OKC = "/mnt/c/Users/Achita/Documents/ChatGPT/Muse 2/.tools/okc"
JOBS = dict(arg.split('=') for arg in sys.argv[1:])


def fetch(jid):
    # achitaan: fail closed: one fresh result per job id; never reuse a previous job's JSON
    for _ in range(30):
        try:
            r = subprocess.run(['wsl', '-d', 'Ubuntu', '--', OKC, 'status', jid, '--wait', '--json'],
                               capture_output=True, text=True, encoding='utf-8', timeout=7200)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[:200])
            j = json.loads(r.stdout)
            if not str(j.get('id', '')).startswith(jid) or j.get('status') not in ('done', 'failed', 'rejected', 'duplicate'):
                raise RuntimeError(f"unexpected id/status {j.get('id')} {j.get('status')}")
            return j
        except Exception as e:
            print('retry', jid[:8], str(e)[:120], file=sys.stderr)
            time.sleep(30)
    return None


out = {}
for name, jid in JOBS.items():
    j = fetch(jid)
    if j is None:
        print(name, jid[:8], 'NO RESULT (not stored)')
        continue
    res = j.get('result') or {}
    os.makedirs('kb/runs', exist_ok=True)
    path = f'kb/runs/{name}.{jid[:8]}.json'
    if not os.path.exists(path):
        json.dump(j, open(path, 'w'))
    bench = [b for b in res.get('records', []) if b.get('event') == 'benchmark']
    valid = res.get('passed') is True and bench and all(b.get('median_ns', 0) > 0 and not b.get('flags') for b in bench)
    print(name, jid[:8], j.get('board'), j.get('mode'), res.get('status'), res.get('passed'), res.get('speedup'),
          'valid' if valid else 'INVALID', str(res.get('error'))[:120])
    if valid:
        out[name] = bench
rows = {}
for name, bench in out.items():
    for b in bench:
        c = b['case']; key = (c.get('load'), c.get('layer', ''), c.get('bs'))
        rows.setdefault(key, {})[name] = b['baseline_ns'] / b['median_ns']
names = list(out)
print('per-shape speedup vs SGLang (same judge, each run its own SGLang baseline)')
print('shape'.ljust(28) + ''.join(n[:12].rjust(13) for n in names))
for k in sorted(rows, key=str):
    print(str(k)[:28].ljust(28) + ''.join((f"{rows[k][n]:.3f}" if n in rows[k] else '-').rjust(13) for n in names))
