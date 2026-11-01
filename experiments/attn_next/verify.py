"""Measure experiments/attn_next/attention.py with one official muse_attention_decode check.

Final stdout line: JSON {"score_us": ..., "speedup": ..., "passed": ..., "job": ...}. score_us is our own
time (lower is better), so it excludes the SGLang reference's run-to-run noise that the speedup carries.
A failed test run reports score_us = 1e9 (worse than any real run) so the trial is discarded.
Everything this script writes goes to ignored folders (runs/, and profiles/ written by okc).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OKC = ['wsl', '-d', 'Ubuntu', '--', '/mnt/c/Users/Achita/Documents/ChatGPT/Muse 2/.tools/okc']
WSL_ROOT = '/mnt/c/Users/Achita/Documents/ChatGPT/Muse 2'


def wsl_path(p):
    s = str(p).replace('\\', '/')
    i = s.index('Muse 2/')
    return WSL_ROOT + '/' + s[i + len('Muse 2/'):]


def run(args, timeout):
    r = subprocess.run(OKC + args, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout.decode('utf-8', 'replace'), r.stderr.decode('utf-8', 'replace')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sub = wsl_path(HERE / 'attention.py')
    job = None
    for attempt in range(10):
        rc, out, err = run(['check', 'muse_attention_decode', sub, '--no-wait', '--json'], 300)
        try:
            job = json.loads(out)['job_id']
            break
        except Exception:
            print(f'submit attempt {attempt} failed: {(out + err)[-300:]}', file=sys.stderr)
            if 'limit' in (out + err).lower() or 'quota' in (out + err).lower():
                sys.exit(3)
            time.sleep(15)
    if job is None:
        sys.exit(2)
    print(f'job {job}', file=sys.stderr)
    rc, out, err = run(['status', job, '--wait', '--json'], 3600)
    d = json.loads(out)
    if d.get('status') == 'duplicate' and d.get('duplicate_of'):
        # the judge does not re-run byte-identical code: use the original job's result
        print(f"duplicate of {d['duplicate_of']}", file=sys.stderr)
        job = d['duplicate_of']
        rc, out, err = run(['status', job, '--wait', '--json'], 3600)
        d = json.loads(out)
    (HERE / 'runs').mkdir(exist_ok=True)
    (HERE / 'runs' / f'{job}.json').write_text(out, encoding='utf-8')
    r = d.get('result') or {}
    passed = bool(r.get('passed'))
    txt = HERE / 'profiles' / f'attention.{job}.txt'
    if txt.exists():
        for line in txt.read_text(encoding='utf-8', errors='replace').splitlines()[:24]:
            print(line, file=sys.stderr)
    score = float(r.get('score_us')) if passed and r.get('score_us') else 1e9
    print(json.dumps({'score_us': score, 'speedup': r.get('speedup'), 'passed': passed, 'job': job}))


if __name__ == '__main__':
    main()
