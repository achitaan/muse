"""Transfer the already-prepared wide router without changing native payloads."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
base_sha = 'fd3b27730d02cc4c5ae282b50db87eaba5bc17e4ca1f990400b14379a91e8d1c'
raw = (root / 'results/resumed-decoder-phase5-20261028' / base_sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == base_sha
text = raw.decode('utf-8')
old = "     logits = torch.mm(ffn_input, state['router_hi_t'], out_dtype=torch.float32)\n     logits += torch.mm(ffn_input, state['router_lo_t'], out_dtype=torch.float32)"
new = "     logits = _achita_wide_router(ffn_input, state)\n     if logits is None:\n      logits = torch.mm(ffn_input, state['router_hi_t'], out_dtype=torch.float32)\n      logits += torch.mm(ffn_input, state['router_lo_t'], out_dtype=torch.float32)"
assert text.count(old) == 1
helper = '''

def _achita_wide_router(x, state):
    # Both existing arms use dynamic M, masked stores and TMA boundary checks.
    # Keep the full FP32 small-token router and the 9..32 fusion unchanged.
    if x.shape[0] <= 32 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    rw = state['achita_dense'].get('bundle', {}).get('rw', {})
    if not rw.get('enabled') or x.device != rw['weight'].device:
        return None
    return rw_run_routerwide_candidate(x, rw, 128 if x.shape[0] <= 128 else 256)
'''
out = text.replace(old, new) + helper
assert out[:-len(helper)].replace(new, old) == text
compile(out, '<wide-router-transfer>', 'exec')
assert len(out.encode('utf-8')) < 1_000_000
path = root / '.tools/decoder-routerwide-variant.py'
path.write_bytes(out.encode('utf-8'))
report = {'source_sha256': hashlib.sha256(out.encode('utf-8')).hexdigest(),
          'base_sha256': base_sha, 'packaged_bytes': len(out.encode('utf-8')),
          'prior_cuda_and_packing_unchanged': True, 'extra_prepare_compilations': 0,
          'router_input_threshold': '>32; original branch is fallback',
          'router_precision': 'BF16 hi/lo weights and input, FP32 accumulation/output',
          'small_fp32_router_unchanged': True, 'gpu_status': 'unmeasured'}
(root / '.tools/decoder-routerwide-source-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
