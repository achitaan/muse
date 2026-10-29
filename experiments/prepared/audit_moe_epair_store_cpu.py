"""Independent full-layout inverse, payload and tail audit; CPU only."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())


def layout(count, shuffle=4, swap=False, bad_tail=False):
    ntt = (count + 7) // 8
    # Model every output-column group and every warp call of the real epilogue.
    r16 = np.arange(512, dtype=np.int32)[:, None, None]
    nt = np.arange(ntt, dtype=np.int32)[None, :, None]
    lane = np.arange(32, dtype=np.int32)[None, None, :]
    g, t = lane >> 2, lane & 3
    col, tk0 = r16 * 8 + g, nt * 8 + 2 * t
    v0 = np.where(tk0 < count, tk0 * 4096 + col, -1)
    v1 = np.where(tk0 + 1 < count, (tk0 + 1) * 4096 + col, -1)
    odd = (g & 1) != 0
    incoming = np.where(odd, v0, v1)
    cross = incoming[..., np.arange(32) ^ shuffle]
    lo, hi = np.where(odd, cross, v0), np.where(odd, v1, cross)
    if swap:
        lo, hi = hi, lo
    tk = tk0 + odd
    c = r16 * 8 + (g & ~1)
    dest = (((c >> 8) * ntt + (tk >> 3)) * 32 + ((c & 255) >> 3)) * 64 + (tk & 7) * 8 + (c & 7)
    mask = np.broadcast_to(tk <= count if bad_tail else tk < count, dest.shape)
    dest, lo, hi = dest[mask], lo[mask], hi[mask]
    assert np.all(dest % 2 == 0), '32bit stores are not aligned'
    cells = np.concatenate((dest, dest + 1))
    assert cells.size == np.unique(cells).size, 'Duplicate writers'
    assert cells.size == count * 4096, 'Dropped output or padding write'
    out = np.full(ntt * 8 * 4096, -1, dtype=np.int32)
    out[dest], out[dest + 1] = lo, hi
    # Independent reference: transpose dense token/column tiles into the
    # documented [k256][n8][k8][token8][inner8] layout; no lane-address formula.
    dense = np.arange(ntt * 8 * 4096, dtype=np.int32).reshape(ntt * 8, 4096)
    dense[count:] = -1
    expected = dense.reshape(ntt, 8, 16, 32, 8).transpose(2, 0, 3, 1, 4).reshape(-1)
    assert np.array_equal(out, expected), 'Payload or layout differs'
    assert np.all((dest * 2) % 4 == 0)
    return int(cells.size)


def main():
    source_path = ROOT / '.tools/moe-epair-store.py'
    audit = json.loads((ROOT / '.tools/moe-epair-store-source-audit.json').read_bytes())
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == audit['source_sha256']
    cuda = (ROOT / '.tools/moe-epair-store.cu').read_text()
    assert hashlib.sha256(cuda.encode()).hexdigest() == audit['epair_cuda_sha256']
    assert audit['original_cuda_inverse_exact'] and audit['phase_B_exact_ignoring_namespace']
    body = (ROOT / 'experiments/prepared/moe_epair_store_phase_a.cuh').read_text()
    assert body.count('__shfl_xor_sync') == 1
    assert body.index('__shfl_xor_sync') < body.index('    if (tk < cnt) {', body.index('__shfl_xor_sync'))
    # Scalar arithmetic expressions and Marlin helper arguments are literal exact.
    assert 'achita_marlin_out2(d4[h], d4[2 + h], gsc)' in body
    assert 'SILU_DIV(gt, 1.f + __expf(-gt)) * up' in body
    assert '__bfloat16_as_ushort(__float2bfloat16(probe_q(v)))' in body
    assert 'odd ? vb[0] : vb[1], 4)' in body
    checks = sum(layout(count) for count in range(1, 65))
    rejected = []
    for name, kwargs in [('wrong_lane_bit', dict(shuffle=2)),
                         ('swapped_columns', dict(swap=True)),
                         ('padding_writer', dict(bad_tail=True))]:
        try:
            layout(9, **kwargs)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Mutant escaped: ' + name)
    # All BF16 payload patterns, including signed zero, subnormals and NaNs,
    # are handled as raw bits. This tests packing, not a floating conversion.
    bits = np.arange(65536, dtype=np.uint32)
    other = bits ^ np.uint32(0xa55a)
    packed = bits | (other << np.uint32(16))
    assert np.array_equal(packed & 65535, bits)
    assert np.array_equal(packed >> np.uint32(16), other)
    source = source_path.read_bytes()
    tree = ast.parse(source)
    parent = (ROOT / 'best/moe_verified.py').read_bytes()
    assert source.startswith(parent)
    parent_tree = ast.parse(parent)
    assert ast.dump(ast.Module(body=tree.body[:len(parent_tree.body)], type_ignores=[])) == ast.dump(parent_tree)
    report = dict(source_sha256=audit['source_sha256'], cuda_sha256=audit['epair_cuda_sha256'],
                  cpu_only=True, gpu_jobs=0, counts=list(range(1, 65)), all4096columns=True,
                  unique_halfword_writes=checks, complete_reference_layout_equal=True,
                  padding_untouched=True, uint32_stores_aligned=True,
                  raw_bf16_pattern_pairs=65536, rejected_mutants=rejected,
                  unconditional_full_warp_shuffle=True, phase_B_unchanged=True,
                  parent_prefix_and_inverse_exact=True,
                  generated_SASS_resources_and_GPU_performance_unverified=True)
    (ROOT / '.tools/moe-epair-store-cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
