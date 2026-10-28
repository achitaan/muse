"""Build one expert-input fusion from the immutable published decoder."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
base_sha = '660f4ac563357a66f3a45632abf3d38113277734025cb0ad10a51a728f0ba1ff'
raw = (root / 'results/resumed-decoder-phase4-20261028' / base_sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == base_sha
base = raw.decode('utf-8')
old = " local_ids = (topk_ids - data['local_start']) % N_EXPERTS\n expert_output = _experts(state['pre_expert_norm'](expert_input), topk_weights, local_ids, state)"
new = " expert_input, local_ids = _pn_expert_inputs(expert_input, topk_ids, data['local_start'], state)\n expert_output = _experts(expert_input, topk_weights, local_ids, state)"
assert base.count(old) == 1
addon = (root / 'experiments/prepared/decoder_expert_input_addon.py').read_text(encoding='utf-8')
text = base.replace(old, new) + '\n' + addon
assert text[:len(base.replace(old, new))].replace(new, old) == base
compile(text, '<expert-input-fusion>', 'exec')
assert len(text.encode('utf-8')) < 1_000_000
out = root / '.tools/decoder-expert-inputs-variant.py'
out.write_bytes(text.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
          'base_sha256': base_sha, 'packaged_bytes': len(text.encode('utf-8')),
          'changed_calls': 'pre-expert RMSNorm and global-to-local ID rotation only',
          'all_previous_cuda_literals_unchanged': True,
          'normalization': 'BF16 input/gain/output, FP32 intermediate and reduction',
          'gpu_status': 'unmeasured'}
(root / '.tools/decoder-expert-inputs-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
