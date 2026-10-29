"""Reversible cluster2 device-prefix patch. No host launcher or GPU execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
CUDA = '1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'

HELPERS = r'''
// Cluster2 X ring: original default counter semantics, remote DSM targets.
__device__ __forceinline__ void cx_sync() {
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cx_remote(uint32_t address, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(address), "r"(rank));
  return result;
}
__device__ __forceinline__ void cx_wait(uint64_t* bar, uint32_t phase) {
  mbar_wait(bar, phase);
}
__device__ __forceinline__ void cx_xrelease(uint64_t* bar) {
  const uint32_t target = cx_remote(smem_u32(bar), 0);
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(target) : "memory");
}
__device__ __forceinline__ void cx_xcopy(unsigned char* dst, const __nv_bfloat16* src, uint32_t bytes, uint64_t* bar) {
  const uint32_t local = smem_u32(bar), peer = cx_remote(local, 1);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %2; mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%1], %2;"
               :: "r"(local), "r"(peer), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(local), "h"((uint16_t)3) : "memory");
}
'''


def main():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    tree = ast.parse(raw)
    full = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    assert hashlib.sha256(full.encode()).hexdigest() == CUDA
    original = full.split('#include <torch/extension.h>', 1)[0]
    source, edits = original, []

    def edit(name, old, new, count=1):
        nonlocal source
        assert source.count(old) == count, (name, source.count(old), count)
        source = source.replace(old, new)
        edits.append(dict(name=name, old=old, new=new, count=count))

    edit('MAIN-only namespace', 'namespace okc {',
         '#if !OKC_WGS || !OKC_PRMTSC\n#error Cluster2 requires eligible WGS MAIN\n#endif\nnamespace okc_cluster2 {')
    edit('namespace comment', '}  // namespace okc', '}  // namespace okc_cluster2')
    edit('cluster helpers', 'template <int NTT> struct WGSMMA;', HELPERS + '\ntemplate <int NTT> struct WGSMMA;')
    edit('producer startup rendezvous', '  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready',
         '  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready\n  cx_sync();', 2)
    edit('consumer startup rendezvous', '  WS_JOIN();\n  DBG(6);', '  WS_JOIN();\n  cx_sync();\n  DBG(6);')
    edit('producer actual unit mapping', '      const int u = gs >> 4, st = gs & 15;',
         '      const int u = 2 * (gs >> 4) + (blockIdx.x & 1), st = gs & 15;', 2)
    edit('consumer actual unit mapping', '    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));',
         '    const int u = 2 * (gs >> 4) + (blockIdx.x & 1), s0 = gs & 15, s1 = min(16, s0 + (ge - gs));')
    # All producer lanes participate in startup before the follower/lane exits.
    old = '''  cx_sync();
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;'''
    new = '''  cx_sync();
  if ((threadIdx.x & 31) != 0 || (blockIdx.x & 1)) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;'''
    edit('only leader issues X', old, new)
    edit('combined empty wait', 'if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);',
         'if (n >= WGS_NX) cx_wait(S.bars + BX_E + xs, xph ^ 1);')
    old = '''      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");'''
    edit('one multicast for two receivers', old,
         '      cx_xcopy(xring + xs * WGS_XS, jb + (size_t)st * npad * 256, bytes, bar);')
    old = '''    }
  }
}
// phase A's x gather'''
    new = '''    }
  }
  // No receiver may retain a last-generation operand or pending remote ack.
  for (int slot = 0; slot < WGS_NX; ++slot)
    if (n > slot) cx_wait(S.bars + BX_E + slot, (uint32_t)((n - 1 - slot) / WGS_NX) & 1u);
}
// phase A's x gather'''
    edit('drain last X slots', old, new)
    edit('consumer full barrier scope',
         'wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);',
         'cx_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u);')
    edit('two CTA consumer releases', 'mbar_arrive1(S.bars + BX_E + prev);',
         'cx_xrelease(S.bars + BX_E + prev);', 2)
    edit('A grouped cost prefix', 'acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;',
         'acc += (uint32_t)wgs_cost(0, S.jC[j]) << 8;')
    edit('B grouped cost prefix', 'accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;',
         'accB += (uint32_t)wgs_cost(1, S.jC[j]) << 7;')
    edit('empty job guard', '''__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre''', '''__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  if (nj == 0) return 0;
  const int* pre''')
    edit('grouped stage_at shift', 'return (lo << (9 - pb)) +', 'return (lo << (8 - pb)) +')
    edit('grouped range endpoint', '(nj << (9 - pb))', '(nj << (8 - pb))')
    for phase in (0, 1):
        edit(f'phase{phase} cluster coordinate',
             f'wgs_range(S, nj, {phase}, blockIdx.x, gridDim.x, S.wgr[{2 * phase}], S.wgr[{2 * phase + 1}]);',
             f'wgs_range(S, nj, {phase}, blockIdx.x >> 1, gridDim.x >> 1, S.wgr[{2 * phase}], S.wgr[{2 * phase + 1}]);')
    start = source.index('__device__ __forceinline__ int wgs_cta_of(')
    end = source.index('\n}\n', start) + 2
    old = source[start:end]
    new = old.replace('  const uint32_t tot',
                      '  const int rank = (s >> 4) & 1;\n  s = (s >> 5) << 4;\n  G >>= 1;\n  const uint32_t tot', 1)
    new = new.replace('  const int j =', '  if (tot == 0) return rank;\n  const int j =', 1)
    new = new.replace('(9 - pb)', '(8 - pb)').replace('return c;', 'return 2 * c + rank;')
    edit('actual-unit fixup owner inverse', old, new)
    edit('empty range guard', '''__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue''', '''__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  if (g1 <= g0) return g0;
  const int ue''')
    edit('required cluster dimension', '__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p)',
         '__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(NTL, 1) moe_fused(Params p)')
    edit('both CTAs release every X slot', 'k < BX_E ? 1 : NW);', 'k < BX_E ? 1 : 2 * NW);')
    old = '''    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;'''
    new = '''    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    __syncwarp();  // reconverge lane0 producer helpers before the aligned barrier
    cx_sync();
    return;'''
    edit('producer lifetime rendezvous', old, new)
    edit('consumer lifetime rendezvous',
         '  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered',
         '  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered\n  __syncwarp();\n  cx_sync();')
    inverse = source
    for item in reversed(edits):
        assert inverse.count(item['new']) == item['count'], item['name']
        inverse = inverse.replace(item['new'], item['old'])
    assert inverse == original
    for name, next_anchor in [('struct Smem {', '// floor(a * c / G)'),
                              ('__device__ __forceinline__ void wgs_epi_tile(', 'template <bool PB>\n__device__ __forceinline__ void wgs_flush(')]:
        old = original[original.index(name):original.index(next_anchor, original.index(name))]
        assert old in source, name
    out = ROOT / '.tools/moe-cluster-x'
    out.mkdir(exist_ok=True)
    (out / 'device.cu').write_bytes(source.encode())
    (out / 'edits.json').write_text(json.dumps(edits, indent=2) + '\n', encoding='utf-8')
    report = dict(parent_submission_sha256=PARENT, parent_CUDA_sha256=CUDA,
                  parent_device_prefix_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  device_prefix_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  device_prefix_bytes=len(source.encode()), reversible_edits=len(edits),
                  inverse_recovers_parent_prefix=True, Smem_layout_and_epilogue_text_unchanged=True,
                  physical_weight_layout_and_tile_dimensions_unchanged=True,
                  barrier_protocol='Original default release/acquire semantics, remote DSM counters; original local waits retained. See protocol proof and actual runtime gates.',
                  host_launcher_present=False, challenge_submission_present=False,
                  kernel_execution=False, GPU_jobs=0)
    (out / 'device-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
