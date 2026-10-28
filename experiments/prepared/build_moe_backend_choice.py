"""Add a bounded backend comparison to the immutable checked DQ8 candidate."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sha = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
raw = (root / 'results/resumed-moe-phase4-20261028' / sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == sha
source = raw.decode('utf-8')
old = '    if TINY_DEFINES and M <= TINY_MAX_M:\n'
new = '''    selected = state.get('backend_by_bucket', {}).get(16 if M <= 16 else 32) if 8 < M <= 32 else None
    if selected is not None:
        ext = _ext(selected)
    elif TINY_DEFINES and M <= TINY_MAX_M:
'''
assert source.count(old) == 1
addon = (root / 'experiments/prepared/moe_backend_choice_addon.py').read_text(encoding='utf-8')
out = source.replace(old, new) + '\n' + addon
assert out[:len(source.replace(old, new))].replace(new, old) == source
compile(out, '<moe-backend-choice>', 'exec')
target = root / '.tools/moe-backend-choice-variant.py'
target.write_bytes(out.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(out.encode('utf-8')).hexdigest(),
          'base_sha256': sha, 'cuda_and_weight_packing_unchanged': True,
          'compared_algorithms': 'incumbent vs existing mixed skinny/fat fallback',
          'eligible_tokens': '9..32; two adjacent shape buckets',
          'synthetic_patterns': ['diffuse', 'hot', 'mixed'],
          'selection_threshold': '3% average gain, no pattern over 5% slower',
          'cached_state': 'backend defines only, no synthetic input or output',
          'gpu_status': 'unmeasured'}
(root / '.tools/moe-backend-choice-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
