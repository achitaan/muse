"""Read-only CPU inversion of the pinned public dense12-bit fragment codec."""
import ast
import hashlib
import json
from pathlib import Path
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
raw = (ROOT / 'results/dense-pinned-codec-screen-20261028/upstream_dense.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'c1dd241c0c08b42fddfa7bc9656d0f596b59bb13005c50c0db4418d9f2f1dc7e'
tree = ast.parse(raw)
ns = {'torch': torch}
for n in tree.body:
    if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in ('EXPONENT_WINDOW', 'EXPONENT_VALUES', 'MAX_ESCAPE_DENOMINATOR', 'BASE4_REPLICATE') for x in n.targets):
        exec(compile(ast.Module(body=[n], type_ignores=[]), '<public-codec-constant>', 'exec'), ns)
    if isinstance(n, ast.FunctionDef) and n.name in ('_exponent_window', '_escapes', 'pack_fragments', 'pack_compressed'):
        exec(compile(ast.Module(body=[n], type_ignores=[]), '<public-codec>', 'exec'), ns)
torch.set_num_threads(2)


def inverse(c, n, k):
    nrg, s4 = n // 16, k // 64
    packed = c['packed'].transpose(0, 1).contiguous()
    lo = packed[:, :, :2].reshape(nrg, s4, 2, 32, 2, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, s4, 4, 32, 8).int()
    cw = packed[:, :, 2].reshape(nrg, s4, 32, 4, 4).permute(0, 1, 3, 2, 4).contiguous().int()
    code = torch.cat((cw & 15, cw >> 4), dim=-1)
    base = c['base4'] // ns['BASE4_REPLICATE']
    bits = ((code & 8) << 12) | (((code & 7) + base) << 8) | lo
    bits = bits.reshape(nrg, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).contiguous().reshape(n, k)
    decoded = bits.to(torch.int16).view(torch.bfloat16)
    corrected = decoded.float().clone()
    for rg in range(nrg):
        lo_, hi_ = int(c['poff'][rg]), int(c['poff'][rg + 1])
        corrected[rg * 16 + c['pr'][lo_:hi_].long(), c['pk'][lo_:hi_].long()] += c['pd'][lo_:hi_]
    return decoded, corrected


records = []
for base in (0, 1, 16, 31, 63, 96, 119):
    bits = []
    for sign in (0, 1):
        for off in range(8):
            for lo in range(256):
                bits.append((sign << 15) | ((base + off) << 8) | lo)
    w = torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(32, 128)
    c = ns['pack_compressed'](w)
    assert c is not None and c['n_esc'] == 0
    d, corrected = inverse(c, *w.shape)
    assert torch.equal(d.view(torch.int16), w.view(torch.int16))
    records.append(dict(base=base, codes=len(bits), direct_fragment_bits_exact=True))

# A deliberately escaped negative zero exposes numeric equality vs bit identity.
# This is a representation counterexample, not a challenge-input prediction.
w = torch.ones((32, 128), dtype=torch.bfloat16)
w.view(torch.int16)[0, 0] = -32768
c = ns['pack_compressed'](w)
assert c is not None and c['n_esc'] == 1
d, corrected = inverse(c, *w.shape)
numeric_equal = bool((corrected == w.float()).all())
bit_equal = torch.equal(corrected.to(torch.bfloat16).view(torch.int16), w.view(torch.int16))
assert numeric_equal and not bit_equal
report = dict(upstream_sha256=hashlib.sha256(raw).hexdigest(), window_cases=records,
              directly_inverted_bf16_values=sum(r['codes'] for r in records),
              escaped_negative_zero=dict(original_bits=int(w.view(torch.int16)[0, 0]) & 65535,
                                        corrected_bits=int(corrected.to(torch.bfloat16).view(torch.int16)[0, 0]) & 65535,
                                        accepted_by_upstream=True,numeric_equal=numeric_equal,bitwise_equal=bit_equal),
              required_local_eligibility='Reject this encoding when reconstructed BF16 bits differ, including signed zero; retain original fallback',
              other_sk2_cg_codecs='not yet audited', candidate_modified=False,
              gpu_correctness='unmeasured', speedup='unmeasured')
(ROOT / '.tools/dense-pinned-codec-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
