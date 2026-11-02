import json, os, subprocess, sys, time
sys.stdout.reconfigure(encoding='utf-8')
OKC = "/mnt/c/Users/Achita/Documents/ChatGPT/Muse 2/.tools/okc"
JOBS = dict(arg.split('=') for arg in sys.argv[1:])
out = {}
for name, jid in JOBS.items():
    for _ in range(30):
        try:
            r = subprocess.run(['wsl', '-d', 'Ubuntu', '--', OKC, 'status', jid, '--wait', '--json'], capture_output=True, text=True, encoding='utf-8', timeout=7200)
            j = json.loads(r.stdout); break
        except Exception:
            time.sleep(30)
    res = j.get('result') or {}
    out[name] = res
    os.makedirs('kb/runs', exist_ok=True)
    json.dump(j, open(f'kb/runs/{name}.{jid[:8]}.json', 'w'))
    print(name, jid[:8], j.get('board'), res.get('status'), res.get('passed'), res.get('speedup'), str(res.get('error'))[:120])
rows = {}
for name, res in out.items():
    for b in res.get('records', []):
        if b.get('event') == 'benchmark':
            c = b['case']; key = (c.get('load'), c.get('layer', ''), c.get('bs'))
            rows.setdefault(key, {})[name] = b['baseline_ns'] / b['median_ns']
names = list(out)
print('per-shape speedup vs SGLang (same judge, each run its own SGLang baseline)')
print('shape'.ljust(28) + ''.join(n[:12].rjust(13) for n in names))
for k in sorted(rows, key=str):
    print(str(k)[:28].ljust(28) + ''.join((f"{rows[k][n]:.3f}" if n in rows[k] else '-').rjust(13) for n in names))
