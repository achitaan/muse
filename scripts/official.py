"""Deadline-aware official CLI driver with immutable local payloads.

Run with the existing Windows Python. WSL access may require an approved shell.
No credentials are read, printed, or copied by this utility.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LINUX_ROOT = ("/mnt/" + ROOT.drive[0].lower() + ROOT.as_posix()[2:]) if os.name == "nt" else str(ROOT)
DEADLINE = datetime(2026, 10, 28, 8, tzinfo=timezone.utc)


def cli(args):
    if os.name == "nt":
        command = ["wsl", "-d", "Ubuntu", "--", f"{LINUX_ROOT}/.tools/okc", *args]
    else:
        local_cli = ROOT / ".tools" / "okc"
        executable = str(local_cli) if local_cli.exists() else shutil.which("okc")
        if not executable:
            raise RuntimeError("Install the official okc CLI first; see HANDOFF.md")
        command = [executable, *args]
    result = subprocess.run(
        command,
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=55,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "submit", "status", "jobs", "quotas"])
    parser.add_argument("target", nargs="?")
    args = parser.parse_args()
    (ROOT / "results" / "payloads").mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    if args.action in {"check", "submit"}:
        if now >= DEADLINE:
            raise RuntimeError("Sprint deadline reached; refusing new official work")
        source = (ROOT / args.target).resolve()
        source.relative_to(ROOT)
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        frozen = ROOT / "results" / "payloads" / f"{digest}.py"
        if frozen.exists() and frozen.read_bytes() != raw:
            raise RuntimeError("Frozen payload collision")
        frozen.write_bytes(raw)
        linux = f"{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}"
        result = cli([args.action, "muse_bf16_proj", linux, "--no-wait", "--json"])
        record = {
            "action": args.action, "source": str(source.relative_to(ROOT)),
            "payload": str(frozen.relative_to(ROOT)), "local_sha256": digest,
            "submitted_utc": now.isoformat(), **result,
        }
        (ROOT / "results" / f"{result['job_id']}-launch.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
    elif args.action == "status":
        result = cli(["status", args.target, "--json"])
        (ROOT / "results" / f"{args.target}-result.json").write_text(json.dumps(result, indent=2))
        # Result layouts can evolve: save complete evidence and expose compact metadata.
        body = result.get("result") or {}
        summary = {key: body.get(key) for key in
                   ("status", "speedup", "speedup_ci", "score_us", "error", "checks", "gpu")}
        print(json.dumps({"id": result.get("id"), "status": result.get("status"),
                          "result": summary if body else None}, indent=2))
    elif args.action == "jobs":
        result = cli(["jobs", "--json"])
        (ROOT / "results" / "jobs.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    else:
        result = cli(["problems", "--json"])
        (ROOT / "results" / "current-quotas.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(next(p for p in result["problems"] if p["name"] == "muse_bf16_proj"), indent=2))


if __name__ == "__main__":
    main()
