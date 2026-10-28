"""Swap L1 QK operands to use full16key x8query FP16 tensor tiles.

LDMATRIX reads FP8 pairs; a common K permutation avoids register shuffles.
Warp-private shared transpose reuses dead query staging, with an explicit
consumer barrier before its reuse. No new shared-memory allocation or geometry.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'


def transform(cuda):
    edits = []

    def replace(before, after):
        nonlocal cuda
        assert cuda.count(before) == 1, before[:120]
        cuda = cuda.replace(before, after)
        edits.append(dict(before=before, after=after))

    anchor = '__device__ __forceinline__ uint4 lds128(const uint8_t* p)'
    helper = '''// L1 QK: two 8x8 matrices of FP8 pairs. Lane group g receives rows g/g+8,
// four adjacent FP8 K elements each. Conversion maps K by (2*t+e)->(4*t+e),
// (8+2*t+e)->(4*t+2+e); the Q B fragments use the same permutation.
__device__ __forceinline__ void qkswap_ld2(uint32_t addr, uint32_t& lo, uint32_t& hi) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(lo), "=r"(hi) : "r"(addr));
}

'''
    replace(anchor, helper + anchor)
    begin = cuda.index('  // Q A-fragments (f16, pre-scaled to log2 units)')
    end = cuda.index('\n  for (int i = 0; i < ntiles; ++i)', begin)
    old_q = cuda[begin:end]
    new_q = '''  // Q as the eight columns of K Q^T. Original FP16 bits and log2 scaling.
  uint32_t qb[4][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int pair0 = 8 * ks + 2 * tig;
      // Undo only the inherited query staging's (0,2,1,3) word order.
      const int p0 = (pair0 & ~3) + ((pair0 & 1) << 1) + ((pair0 & 2) >> 1);
      const int pair1 = pair0 + 1;
      const int p1 = (pair1 & ~3) + ((pair1 & 1) << 1) + ((pair1 & 2) >> 1);
      qb[ks][0] = *reinterpret_cast<const uint32_t*>(qs + g * 128 + 4 * p0);
      qb[ks][1] = *reinterpret_cast<const uint32_t*>(qs + g * 128 + 4 * p1);
    }
  }
  // All consumers have read their queries before any warp reuses S.kvq.
  csync<WS, NT>();
  float* qk_transpose = reinterpret_cast<float*>(S.kvq) + warp * 128;
'''
    replace(old_q, new_q)
    begin = cuda.index('    float s[4][4];\n    uint32_t kw[4][4];', begin)
    end = cuda.index('\n    if (!interior)', begin)
    old_qk = cuda[begin:end]
    new_qk = '''    float s[4][4];
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        // All32lanes participate. Lower16lanes supply row addresses; upper16
        // repeat valid addresses. The existing128Bswizzle is unchanged.
        const int key = 32 * wsub + 16 * kt + (lane & 15);
        const uint32_t addr = smem_u32(K + key * ROWB + (((hk * 4 + ks) ^ (key & 7)) << 4));
        uint32_t kl, kh;
        qkswap_ld2(addr, kl, kh);
        mma_f16(c, cvt2(kl), cvt2(kh), cvt2h(kl), cvt2h(kh), qb[ks][0], qb[ks][1]);
      }
      // C is16keys x8queries. Warp-private XOR layout: conflict-free scalar
      // stores and two128Btransactions for each32-lane float2 read.
      const int head0 = 2 * tig, key0 = g;
      qk_transpose[((head0 * 16 + key0) ^ (tig * 8))] = c[0];
      qk_transpose[(((head0 + 1) * 16 + key0) ^ (tig * 8))] = c[1];
      qk_transpose[((head0 * 16 + key0 + 8) ^ (tig * 8))] = c[2];
      qk_transpose[(((head0 + 1) * 16 + key0 + 8) ^ (tig * 8))] = c[3];
      __syncwarp();
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int ix = (g * 16 + 8 * half + 2 * tig) ^ ((g >> 1) * 8);
        const float2 v = *reinterpret_cast<const float2*>(qk_transpose + ix);
        s[2 * kt + half][0] = v.x;
        s[2 * kt + half][1] = v.y;
        s[2 * kt + half][2] = 0.f;
        s[2 * kt + half][3] = 0.f;
      }
      __syncwarp();  // every read is complete before the next16key tile reuses it
    }'''
    replace(old_qk, new_qk)
    return cuda, edits


def build():
    raw = (ROOT / '.attention-measurements' / PARENT / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    source = raw.decode('utf8')
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == 'CUDA_SRC' for x in n.targets))
    cuda = ast.literal_eval(node.value)
    changed, edits = transform(cuda)
    reverse = changed
    for e in reversed(edits):
        assert reverse.count(e['after']) == 1
        reverse = reverse.replace(e['after'], e['before'])
    assert reverse == cuda
    old_literal = ast.get_source_segment(source, node.value)
    result = source.replace(old_literal, repr(changed), 1)
    assert result.replace(repr(changed), old_literal, 1) == source
    compile(result, '<attention-qkswap>', 'exec')
    out = result.encode('utf8')
    (ROOT / '.tools/attention-qkswap.py').write_bytes(out)
    (ROOT / '.tools/attention-qkswap.cu').write_bytes(changed.encode())
    report = dict(parent_sha256=PARENT, source_sha256=hashlib.sha256(out).hexdigest(),
                  bytes=len(out), inverse_recovers_parent=True, edits=edits,
                  bf16_boundaries_and_fp16_qk=True, fp8_pv_unchanged=True,
                  routing_cache_partition_geometry_l8_unchanged=True,
                  qk_mma_per_warp_tile_before=16, qk_mma_per_warp_tile_after=8,
                  reused_query_smem_bytes=2048, gpu_status='unmeasured')
    (ROOT / '.tools/attention-qkswap-source-audit.json').write_bytes(json.dumps(report, indent=2).encode())
    print(json.dumps({k: v for k, v in report.items() if k != 'edits'}))


if __name__ == '__main__':
    build()
