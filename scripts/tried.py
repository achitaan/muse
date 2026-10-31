"""Shared ledger of every optimization idea tried in the Muse 2 repo: TRIED.md on origin/main.

Works from any worktree or branch: reads origin/main directly and commits to main with git plumbing
(temporary index, commit-tree, push), so no checkout or working tree is touched.

  python tried.py search WORD [WORD ...]     rows containing all words (case-insensitive)
  python tried.py show                       the whole ledger
  python tried.py add --area MoE --idea "..." --outcome closed --number "..." [--job ID] [--source PATH]
                     [--replace "text of an existing row to supersede"]

Outcomes: kept | closed | parked | in progress | open (not tried).
Use "in progress" to claim an idea before spending checks on it; then add the result with --replace.
"""
import argparse
import datetime
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILE = 'TRIED.md'
COLS = '| Idea | Outcome | Number | Job | Where | Date |'
SEP = '|---|---|---|---|---|---|'


def git(*args, env=None, check=True, data=None):
    r = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, env=env, input=data)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r


def fetch():
    for i in range(5):
        if git('fetch', '-q', 'origin', 'main', check=False).returncode == 0:
            return
        time.sleep(3 * (i + 1))
    raise RuntimeError('git fetch origin main failed')


def ledger():
    return git('show', f'origin/main:{FILE}').stdout.decode('utf-8')


def cell(s):
    return (s or '-').replace('|', '/').replace('\n', ' ').strip() or '-'


def insert(text, area, row, replace=None):
    lines = text.split('\n')
    if replace:
        hits = [i for i, l in enumerate(lines) if l.startswith('|') and replace.lower() in l.lower()]
        if len(hits) != 1:
            raise SystemExit(f'--replace matched {len(hits)} rows; make it more specific')
        del lines[hits[0]]
    head = next((i for i, l in enumerate(lines) if l.strip().lower() == f'## {area}'.lower()), None)
    if head is None:
        end = len(lines)
        while end and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = ['', f'## {area}', '', COLS, SEP, row]
    else:
        i = head + 1
        while i < len(lines) and not lines[i].startswith('## '):
            i += 1
        last = max((j for j in range(head, i) if lines[j].startswith('|')), default=None)
        if last is None:
            lines[head + 1:head + 1] = ['', COLS, SEP, row]
        else:
            lines.insert(last + 1, row)
    return '\n'.join(lines)


def commit(new_text, message):
    blob = git('hash-object', '-w', '--stdin', data=new_text.encode('utf-8')).stdout.decode().strip()
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(d) / 'index'))
        git('read-tree', 'origin/main', env=env)
        git('update-index', '--add', '--cacheinfo', f'100644,{blob},{FILE}', env=env)
        tree = git('write-tree', env=env).stdout.decode().strip()
    return git('commit-tree', tree, '-p', 'origin/main', '-m', message).stdout.decode().strip()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search'); s.add_argument('words', nargs='+')
    sub.add_parser('show')
    a = sub.add_parser('add')
    for k in ('area', 'idea', 'outcome', 'number'):
        a.add_argument('--' + k, required=True)
    a.add_argument('--job'); a.add_argument('--source'); a.add_argument('--replace')
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    fetch()
    if args.cmd == 'show':
        print(ledger()); return
    if args.cmd == 'search':
        area, n = '', 0
        for l in ledger().split('\n'):
            if l.startswith('## '):
                area = l[3:]
            elif l.startswith('|') and all(w.lower() in (area + ' ' + l).lower() for w in args.words):
                print(f'[{area}] {l}'); n += 1
        print(f'{n} matching rows'); return
    row = '| ' + ' | '.join(cell(x) for x in (args.idea, args.outcome, args.number, args.job and f'`{args.job}`',
                                                args.source, datetime.date.today().strftime('%b %d'))) + ' |'
    for attempt in range(6):
        sha = commit(insert(ledger(), args.area, row, args.replace), f'TRIED: {args.area}: {args.idea[:60]} ({args.outcome})')
        if git('push', '-q', 'origin', f'{sha}:refs/heads/main', check=False).returncode == 0:
            git('fetch', '-q', 'origin', 'main', check=False)
            print('recorded on main', sha[:8], row); return
        time.sleep(3 * (attempt + 1)); fetch()
    raise SystemExit('push to main failed 6 times; nothing recorded')


if __name__ == '__main__':
    main()
