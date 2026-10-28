"""Prepare a small-token padding/boundary fusion without changing larger paths."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
base_sha = 'a61735ee0cb7774ae8e187526d2e104f3df4e2e95453f66ee3b46fdc33f9186d'
raw = (root / 'results/resumed-decoder-phase5-20261028' / base_sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == base_sha
source = raw.decode('utf-8')
old = ''' if tokens <= SLICED_UNPOOL_MAX_TOKENS:
  unpooled = _achita_dense_project(expert_output, 'unpooler', state, UNPOOL_SLICE)
  ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
 else:
  ffn_output = _achita_dense_project(expert_output, 'unpooler', state)
 (next_attn_input, stream, _) = state['ffn_boundary'](stream, ffn_output, branch_dtype=torch.bfloat16)'''
new = ''' if tokens <= SLICED_UNPOOL_MAX_TOKENS:
  unpooled = _achita_dense_project(expert_output, 'unpooler', state, UNPOOL_SLICE)
  next_attn_input, stream = _sfn_boundary(stream, unpooled, state)
 else:
  ffn_output = _achita_dense_project(expert_output, 'unpooler', state)
  next_attn_input, stream, _ = state['ffn_boundary'](stream, ffn_output, branch_dtype=torch.bfloat16)'''
assert source.count(old) == 1
addon = (root / 'experiments/prepared/sliced_ffn_boundary_addon.py').read_text(encoding='utf-8')
text = source.replace(old, new) + '\n' + addon
assert text[:len(source.replace(old, new))].replace(new, old) == source
compile(text, '<sliced-ffn-boundary>', 'exec')
assert len(text.encode('utf-8')) < 1_000_000
target = root / '.tools/decoder-sliced-ffn-variant.py'
target.write_bytes(text.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
          'base_sha256': base_sha, 'packaged_bytes': len(text.encode('utf-8')),
          'enabled_tokens': '1..8', 'norm_domain': '8192, including implicit zero tail',
          'branch_product_and_norm_output': 'BF16 boundaries preserved',
          'residual_and_pre_norm_input': 'FP32', 'all_original_cuda_unchanged': True,
          'larger_token_path_unchanged': True, 'gpu_status': 'unmeasured'}
(root / '.tools/decoder-sliced-ffn-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
