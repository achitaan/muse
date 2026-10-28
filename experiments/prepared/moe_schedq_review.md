# MoE split-work owner estimate

The retained source is 33114871. WGS32 and six-bit scale encoding lost in full
checks; neither is inherited. This trial changes one expression in the original
WGS scheduler's wgs_cta_of, from a 64-bit integer quotient to an FP32 quotient
estimate. Both original integer correction loops and their order remain exact.
GEMMs, scales, routing, synchronization, storage, scratch and BF16/FP32 norm
boundaries are unchanged. No extra module, preparation or runtime selector.

The owner is the largest c in [0,G-1] satisfying floor(tot*c/G)<xe. Starting
from a bounded estimate, the original upward loop reaches the last satisfying
c if it starts low. If it starts high, the original downward loop reaches that
same c. Strictly positive costs and tot make this predicate monotone. Unit
starts have 0<xe<=tot. Thus the estimate's error changes correction work but
cannot change the resulting owner. All comparison arithmetic stays integer.

JMAX=88, costs<=255, at most512 stages/job and G<=160 give
tot*G<=88*512*255*160=1838284800<2^32, preserving the original unsigned products.
All float conversions are finite, positive and far from exceptional domains.
The actual host grid is smaller; using160 is a conservative scratch-capacity
bound. Empty workloads never call wgs_cta_of; this trial makes no changes to
the original empty-work scheduling behavior.

CPU audit seed20261028 covered157 distributions, grid1..160, both phases,
all job unit starts, full capacity, nearby M and repeated physical routes.
It compared82275840 owners against the independent integer oracle
ceil(xe*G/tot)-1, including +/-2ULP perturbations of the float32 divide. Every
owner matched; at most one correction was observed. This is a CPU control and
rounding-envelope check, not GPU numerical or performance evidence.

The source ledger reverses the sole CUDA expression/comment replacement and
recovers the complete original Python module AST. Numerical CUDA helpers are
byte-identical. Frozen draft SHA256:
df1871fad29b04f70aed36c7e3e1a5085066346c966aee5d07fe96e1a928a843,
179096 bytes. Ordinary load_inline source hashing separates compiler caches.

NVIDIA documents costly integer division and a maximum2ULP error for
__fdividef in this positive domain. This motivates a trial; without local
NVCC/SASS we do not claim that this exact division emits a particular helper,
nor that replacing it is faster. References:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html

One full official check, cached baseline2.365437416x, standard5tests/6benchmarks
and integrity/precision guards. Compare all WGS loads32/128/256 and unchanged
controls1/8/16. Preserve the job handle. Do not score a flat/noisy point or
start a denominator/quotient tuning sweep. Keep Goal active at phase boundaries.
