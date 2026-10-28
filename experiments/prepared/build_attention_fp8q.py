"""One-query FP8 QK, from the corrected source; all other math is retained."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
BASE_SHA = '24782695e507801dcf4e753349a97de723d99e3463b2aac0c67a645ff53a884a'


def literal(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    return node, ast.literal_eval(node.value)


def transform(old):
    changes = []
    def change(before, after, count=1):
        nonlocal old
        assert old.count(before) == count, (before[:120], old.count(before))
        old = old.replace(before, after)
        changes.append(dict(before=before, after=after, count=count))
    marker = '__device__ __forceinline__ uint4 lds128'
    change(marker, '''// FP8 QK for a single query: eight live A rows and eight zero rows.
__device__ __forceinline__ void mma_f8q(float* c, uint32_t a0, uint32_t a2,
                                      uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%5}, {%7,%8}, "
      "{%0,%1,%2,%3};\\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(b0), "r"(b1));
}

''' + marker)
    before = '''#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }   // (scout: quad order 0,2,1,3)
    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }'''
    change(before, '''    if constexpr (L == 1) {
      // Keep each thread's original 16-byte slot: compacting the whole row
      // here would race another thread's raw BF16 query read in WS mode.
      const uint2 q8 = make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]),
                                 e4m3x4f(x[4], x[5], x[6], x[7]));
      uint8_t* qs = WS ? S.kvq + rr * 128 :
          (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES + (j * 8 + gg) * 128;
      *reinterpret_cast<uint2*>(qs + partL * 16) = q8;
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
      if constexpr (WS) {
        *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
      } else {
        uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
        *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
      }
    }''')
    start = old.index('  // Q A-fragments (f16, pre-scaled')
    end = old.index('\n  for (int i = 0; i < ntiles;', start)
    change(old[start:end], '''  // PTX m16n8k32: A0/A2 are row g, dims 4*tig + {0..3}
  // and +16; A1/A3 are zero. Two K32 steps cover all 64 dimensions.
  uint32_t qa[2][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int d = h * 32 + r * 16 + 4 * tig;
        qa[h][r] = *reinterpret_cast<const uint32_t*>(qs + g * 128 + (d >> 3) * 16 + (d & 7));
      }
    }
  }
''')
    start = old.index('    uint32_t kw[4][4];', old.index('  // PTX m16n8k32:'))
    end = old.index('    if (!interior) {', start)
    change(old[start:end], '''    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        // B0/B1: key g, dimensions 4*tig+{0..3} and +16.
        // Each warp's 32 words touch 32 distinct shared-memory banks.
        const int off = key * ROWB + (((hk * 4 + r) ^ (key & 7)) << 4) + 4 * tig;
        kw[nt][r] = *reinterpret_cast<const uint32_t*>(K + off);
      }
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h)
        mma_f8q(sc, qa[h][0], qa[h][1], kw[nt][2 * h], kw[nt][2 * h + 1]);
      // Q stays unit-RMS through E4M3 conversion, including nonunit KV scales.
      s[nt][0] = sc[0] * sl2; s[nt][1] = sc[1] * sl2;
      s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
''')
    change('attn_kernel', 'attn_kernel_fp8q', old.count('attn_kernel'))
    return old, changes


def build():
    raw = (ROOT/'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    source = raw.decode('utf8')
    node, old = literal(source)
    new, changes = transform(old)
    reverse = new
    for edit in reversed(changes):
        assert reverse.count(edit['after']) == edit['count']
        reverse = reverse.replace(edit['after'], edit['before'])
    assert reverse == old
    lines = source.splitlines(keepends=True)
    begin = sum(map(len, lines[:node.value.lineno-1])) + node.value.col_offset
    end = sum(map(len, lines[:node.value.end_lineno-1])) + node.value.end_col_offset
    result = source[:begin] + repr(new) + source[end:]
    tree = ast.parse(result)
    changed_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    changed_node.value = node.value
    assert ast.dump(tree) == ast.dump(ast.parse(source))
    output = ROOT/'.tools/attention-fp8q-variant.py'
    output.write_bytes(result.encode('utf8'))
    (ROOT/'.tools/attention-fp8q.cu').write_bytes(new.encode('utf8'))
    report = dict(parent_sha256=BASE_SHA, source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  bytes=output.stat().st_size, inverse_exact=True, other_python_ast_unchanged=True,
                  edits=changes)
    (ROOT/'.tools/attention-fp8q-source-audit.json').write_bytes(json.dumps(report,indent=2).encode())
    print(json.dumps({k:v for k,v in report.items() if k != 'edits'},indent=2))


if __name__ == '__main__':
    build()
