"""Exact source derivation of an early outgoing trigger in the stock norm."""

import hashlib
import json
from pathlib import Path

from pdl16_sources import ROOT, HERE

OUT = ROOT / 'results/decode-moe-plan-20261030/pdl-norm'
UPSTREAM = OUT / 'upstream'
COMMIT = 'a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f'
NOTICE = ('// Derived from sgl-project/sglang a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (Apache-2.0).\n'
          '// Local modifications: scalar outgoing PDL trigger moved before its wait; private names and resource query.\n')


def pdln_derive(source):
    edits = [
        ('  PDLWaitPrimary<kUsePDL>();',
         '  PDLTriggerSecondary<kUsePDL>();  // independent downstream weight prologue\n  PDLWaitPrimary<kUsePDL>();', 2, 1),
        ('  PDLTriggerSecondary<kUsePDL>();\n}\n\n// Multi-CTA path',
         '}\n\n// Multi-CTA path', 1, 1),
        ('muse_spark_norm_residual_kernel', 'pdln_norm_kernel', 2, -1),
        ('MuseSparkNormResidual', 'PdlnMuseSparkNormResidual', 1, -1),
    ]
    old = source
    for a, b, count, replacements in edits:
        assert source.count(a) == count, (a, source.count(a))
        source = source.replace(a, b, replacements)
    reverse = source
    for a, b, count, replacements in reversed(edits):
        reverse = reverse.replace(b, a, replacements)
    assert reverse == old
    return source


QUERY = r'''
// Local, separate resource fields; no allocations or input-data reads.
namespace sglang {
int64_t pdln_query(int64_t what) {
  auto kernel = pdln_norm_kernel<8192,256,true,true,false,false,false,true>;
  cudaFuncAttributes a{};
  cudaError_t status = cudaFuncGetAttributes(&a, kernel);
  if (status != cudaSuccess) return -1000-(int)status;
  if (what == 0) return a.numRegs;
  if (what == 1) return a.localSizeBytes;
  if (what == 2) return a.sharedSizeBytes;
  if (what == 3) {
    int blocks = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,256,0);
    return status == cudaSuccess ? blocks : -2000-(int)status;
  }
  return -1;
}
}
'''


def main():
    source = (UPSTREAM / 'norm_residual.cuh').read_text()
    old_pin = (OUT.parent / 'pdl-chain/stock-norm-pinned/norm_residual.cuh').read_text()
    assert source == old_pin, 'Public pinned commit differs from the recorded source'
    derived = pdln_derive(source)
    # One CTA, 256 logical threads and the original eight-warp XOR reductions.
    kernel = derived[derived.index('template <'):derived.index('// Multi-CTA path')]
    assert kernel.index('PDLTriggerSecondary<kUsePDL>();') < kernel.index('PDLWaitPrimary<kUsePDL>();')
    assert kernel.index('PDLWaitPrimary<kUsePDL>();') < kernel.index('branch_v.load')
    (OUT / 'norm-early.cuh').write_bytes((NOTICE + derived + QUERY).encode())
    names = ['norm_residual.cuh','cta.cuh','warp.cuh','math.cuh','utils.cuh','vec.cuh','type.cuh']
    hashes = {n: hashlib.sha256((UPSTREAM / n).read_text().encode()).hexdigest() for n in names}
    record = dict(upstream_commit=COMMIT, canonical_LF_hashes=hashes,
                  derived_header_sha256=hashlib.sha256(derived.encode()).hexdigest(),
                  source_with_query_sha256=hashlib.sha256((NOTICE + derived + QUERY).encode()).hexdigest(),
                  inverse_exact=True, changes='Scalar outgoing trigger moved before incoming wait; private scalar kernel and host class names.',
                  arithmetic_and_reduction_source_unchanged=True, GPU_jobs=0,
                  cluster_path_unchanged_and_unused=True)
    (OUT / 'source-binding.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
