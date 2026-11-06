"""Run an existing diagnostic against separate ordered-fixup sources."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
allowed = {"check_native_marlin.py", "check_native_replay.py", "bench_native_marlin.py"}
script = sys.argv.pop(1)
if script not in allowed:
    raise ValueError("unknown diagnostic")
sys.path.insert(0, str(root / "experiments/ordered_fixup"))
runpy.run_path(str(root / script), run_name="__main__")
