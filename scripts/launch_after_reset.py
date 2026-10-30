import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
target = (now + timedelta(days=1)).replace(hour=0, minute=1, second=30, microsecond=0) if now.hour >= 12 else now.replace(hour=0, minute=1, second=30, microsecond=0)
print(f'waiting until {target.isoformat()} (now {now.isoformat()})', flush=True)
while datetime.now(timezone.utc) < target:
    time.sleep(30)
print('launching', datetime.now(timezone.utc).isoformat(), flush=True)
sys.exit(subprocess.call([sys.executable, 'scripts/measure_dense_lane.py', *sys.argv[1:]]))
