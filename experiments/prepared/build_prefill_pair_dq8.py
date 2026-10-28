"""Transfer exact FP4 placement and paired Marlin roundings to grouped Prefill.

No scheduling, tile geometry, activations, routing, or dense/attention changes.
The byte-exact inverse ledger makes the scope independently reviewable.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7'


def build():
    raw = (ROOT / '.prefill-measurements' / PARENT / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    source = raw.decode('utf8')
    tree = ast.parse(source)
    cuda_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == '_pg__CUDA_SRC' for t in n.targets))
    pos_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == '_pg__CODE_POS' for t in n.targets))
    original_cuda = ast.literal_eval(cuda_node.value)
    cuda = original_cuda
    ledger = []

    def replace(old, new):
        nonlocal cuda
        assert cuda.count(old) == 1, old
        cuda = cuda.replace(old, new)
        ledger.append({'before': old, 'after': new})

    old = '''  const uint32_t r0 = q & M;
  const uint32_t r1 = (q * m16) & M;
  const uint32_t r2 = __funnelshift_l(q, q, 10) & M;
  const uint32_t r3 = ((q * m64) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);'''
    new = '''  // Exact eight-operation placement, Lee Penkman (Apache-2.0),
  // lee101/openkernels c565e373d34575c5f7df862b05f8be7619078b0f.
  // Matching prepare-time bit permutation; every BF16 fragment bit is unchanged.
  const uint32_t r0 = q & M;
  const uint32_t r1 = __funnelshift_l(q, q, 10) & M;
  const uint32_t r2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t r3 = ((q & 0x001B001Bu) * 0x1084u) & M;'''
    replace(old, new)

    old = '''DEVI float achita_marlin_out(float encoded_acc, float encoded_scale) {
  const float acc = __bfloat162float(__float2bfloat16_rn(encoded_acc * 128.f));
  const float gs = __bfloat162float(__float2bfloat16_rn(encoded_scale * 0x1p-7f));
  return __bfloat162float(__float2bfloat16_rn(acc * gs));
}'''
    new = '''// Pair independent outputs, retaining BF16 accumulator/global-scale/product boundaries.
DEVI __nv_bfloat162 achita_marlin_pair(float encoded_acc0, float encoded_acc1, float encoded_scale) {
  const __nv_bfloat162 acc = __floats2bfloat162_rn(encoded_acc0 * 128.f, encoded_acc1 * 128.f);
  const __nv_bfloat16 gs = __float2bfloat16_rn(encoded_scale * 0x1p-7f);
  return __hmul2_rn(acc, __bfloat162bfloat162(gs));
}'''
    replace(old, new)
    replace('''        const float gt = achita_marlin_out(acc[4 * jj + b], cs), up = achita_marlin_out(acc[4 * jj + 2 + b], cs);''',
            '''        const float2 pair = __bfloat1622float2(achita_marlin_pair(acc[4 * jj + b], acc[4 * jj + 2 + b], cs));
        const float gt = pair.x, up = pair.y;''')
    replace('''        r[c0] = __float2bfloat16_rn(achita_marlin_out(acc[4 * jj + b], cs));
        r[c0 + 8] = __float2bfloat16_rn(achita_marlin_out(acc[4 * jj + 2 + b], cs));''',
            '''        const __nv_bfloat162 pair = achita_marlin_pair(acc[4 * jj + b], acc[4 * jj + 2 + b], cs);
        r[c0] = __low2bfloat16(pair);
        r[c0 + 8] = __high2bfloat16(pair);''')
    replace('''        const float h0 = silu_fast(achita_marlin_out(acc[4 * jj], cs)) * (achita_marlin_out(acc[4 * jj + 2], cs));
        const float h1 = silu_fast(achita_marlin_out(acc[4 * jj + 1], cs)) * (achita_marlin_out(acc[4 * jj + 3], cs));''',
            '''        const float2 gates = __bfloat1622float2(achita_marlin_pair(acc[4 * jj], acc[4 * jj + 1], cs));
        const float2 ups = __bfloat1622float2(achita_marlin_pair(acc[4 * jj + 2], acc[4 * jj + 3], cs));
        const float h0 = silu_fast(gates.x) * ups.x;
        const float h1 = silu_fast(gates.y) * ups.y;''')
    replace('''            r[2 * u] = bf162_u32(__floats2bfloat162_rn(achita_marlin_out(acc[ai], cs), achita_marlin_out(acc[ai + 1], cs)));
            r[2 * u + 1] = bf162_u32(__floats2bfloat162_rn(achita_marlin_out(acc[ai + 2], cs), achita_marlin_out(acc[ai + 3], cs)));''',
            '''            r[2 * u] = bf162_u32(achita_marlin_pair(acc[ai], acc[ai + 1], cs));
            r[2 * u + 1] = bf162_u32(achita_marlin_pair(acc[ai + 2], acc[ai + 3], cs));''')
    replace('''            r[c0] = __float2bfloat16_rn(achita_marlin_out(acc[ai], cs));
            r[c0 + 8] = __float2bfloat16_rn(achita_marlin_out(acc[ai + 2], cs));''',
            '''            const __nv_bfloat162 pair = achita_marlin_pair(acc[ai], acc[ai + 2], cs);
            r[c0] = __low2bfloat16(pair);
            r[c0 + 8] = __high2bfloat16(pair);''')
    assert 'achita_marlin_out(' not in cuda
    assert cuda.count('achita_marlin_pair(') == 8
    restored = cuda
    for edit in reversed(ledger):
        assert restored.count(edit['after']) == 1
        restored = restored.replace(edit['after'], edit['before'])
    assert restored == original_cuda
    old_pos = ast.literal_eval(pos_node.value)
    slots = ((3, 7), (2, 6), (1, 5), (0, 4))
    new_pos = {}
    for reg, pair in enumerate(slots):
        for upper, slot in enumerate(pair):
            new_pos[slot] = tuple(((out_bit + 16 * upper - (0, 10, 13)[reg]) % 32
                                  if reg < 3 else (4, 0, 1, 3)[bit] + 16 * upper)
                                 for bit, out_bit in enumerate((6, 7, 8, 15)))
    assert sorted(b for values in new_pos.values() for b in values) == list(range(32))
    cuda_literal = ast.get_source_segment(source, cuda_node.value)
    pos_literal = ast.get_source_segment(source, pos_node.value)
    output = source.replace(cuda_literal, repr(cuda), 1).replace(pos_literal, repr(new_pos), 1)
    assert output.replace(repr(cuda), cuda_literal, 1).replace(repr(new_pos), pos_literal, 1) == source
    output += ('\n# Exact FP4 placement: Lee Penkman, copyright 2026, Apache-2.0.\n'
               '# lee101/openkernels c565e373d34575c5f7df862b05f8be7619078b0f.\n'
               '# Local integration pairs BF16 epilogues and losslessly permutes packed bits.\n')
    compile(output, '<prefill-pair-dq8>', 'exec')
    raw_out = output.encode('utf8')
    report = dict(parent_sha256=PARENT, source_sha256=hashlib.sha256(raw_out).hexdigest(),
                  changed_components=['convert8 bit placement', 'matching _pg__CODE_POS', 'BF16 paired epilogues', 'attribution'],
                  old_positions=old_pos, new_positions=new_pos, inverse_recovers_parent=True,
                  epilogue_pairs=7, cuda_edits=ledger, gpu_status='unmeasured')
    (ROOT / '.tools/prefill-pair-dq8.py').write_bytes(raw_out)
    (ROOT / '.tools/prefill-pair-dq8-source-audit.json').write_bytes(json.dumps(report, indent=2).encode())
    print(json.dumps({k: v for k, v in report.items() if k not in ('cuda_edits', 'old_positions', 'new_positions')}))
    return report


if __name__ == '__main__':
    build()
