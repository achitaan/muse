"""Bind the audited skinny16 prototype to a bounded ordinary prepare selector."""
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
raw = (ROOT / 'results/moe-skinny16-preflight-20261028/prototype.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '921dc83ec4bdf76b0c33e777c028adfdfec84265439ca390a7369e1b6f410f96'
addon = (ROOT / 'experiments/prepared/moe_skinny16_addon.py').read_text(encoding='utf8')
out = raw + ('\n' + addon).encode()
compile(out, '<moe-skinny16-selection>', 'exec')
(ROOT / '.tools/moe-skinny16-selected.py').write_bytes(out)
report = dict(source_sha256=hashlib.sha256(out).hexdigest(), bytes=len(out),
              prototype_sha256=hashlib.sha256(raw).hexdigest(), original_and_prototype_prefix_exact=True,
              eligible_tokens=[9, 16], numerical_shapes=[9, 15, 16], routing_patterns=['diffuse', 'hot', 'mixed'],
              analytic_cases='all-remote -> exact zero at M9/15/16',
              timing='8 alternating cold-L2 graph pairs for each of three patterns at M16',
              selection='geomean new/old<.97; no pattern ratio>1.02',
              status='prepared; no GPU evaluation')
(ROOT / '.tools/moe-skinny16-selection-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
