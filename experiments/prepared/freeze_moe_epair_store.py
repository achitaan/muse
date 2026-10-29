"""Freeze the audited source and freshly downloaded evaluator identity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())


def main():
    folder = ROOT / 'results/moe-epair-store-preflight-20261029'
    folder.mkdir(exist_ok=True)

    def write(name, data):
        p = folder / name
        if p.exists():
            assert p.read_bytes() == data, 'Frozen packet collision: ' + name
        else:
            p.write_bytes(data)

    for src, name in [('moe-epair-store.py', 'submission.py'),
                      ('moe-epair-store.cu', 'epair.cu'),
                      ('moe-epair-store-edits.json', 'edits.json'),
                      ('moe-epair-store-source-audit.json', 'source-audit.json'),
                      ('moe-epair-store-cpu-audit.json', 'cpu-audit.json'),
                      ('moe-epair-store-host-audit.json', 'host-audit.json')]:
        write(name, (ROOT / '.tools' / src).read_bytes())
    identity_path = folder / 'evaluator-identity.json'
    if identity_path.exists():
        report = json.loads(identity_path.read_bytes())
    else:
        rows = []
        for name in ('TASK.md', 'reference.py', 'checks.py', 'baseline.py', 'nvfp4.py', 'AGENTS.md'):
            old = (ROOT / 'muse_moe_w4a16' / name).read_bytes()
            fresh = (ROOT / '.tools/fresh-moe-epair-20261029' / name).read_bytes()
            assert old == fresh, 'Official evaluator changed: ' + name
            rows.append(dict(file=name, pinned_sha256=hashlib.sha256(old).hexdigest(),
                             fresh_sha256=hashlib.sha256(fresh).hexdigest(), exact=True))
        report = dict(problem='muse_moe_w4a16', files=rows,
                      identity_verified_utc=datetime.now(timezone.utc).isoformat(),
                      fresh_download='Official okc init into .tools/fresh-moe-epair-20261029; no GPU job')
        write('evaluator-identity.json', (json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(dict(packet=str(folder.relative_to(ROOT)), evaluator_files=len(report['files']), gpu_jobs=0)))


if __name__ == '__main__':
    main()
