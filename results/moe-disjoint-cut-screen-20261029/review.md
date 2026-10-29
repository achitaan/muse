# Disjoint cut alignment: valid model, insufficient opportunity

This CPU-only screen binds retained MoE33114871 and CUDA1e60cf46. It changes no
CUDA source, timed dispatch or verified file. Source cost tables are extracted
and checked against that exact source. Saved synthetic routes are from the
earlier ownership screen, not judged routing or fresh GPU measurements.

The one fixed rule considers odd cut indices. A cut one stage past a unit edge
may move backward, or one stage before an edge may move forward, only when the
affected pair's maximum estimated work does not increase. Neighboring selected
cuts are separated by an untouched cut, so the affected CTA pairs are disjoint.

Let F be cumulative positive source-table stage cost and let a,x,b be the three
original boundaries. L=F(x)-F(a), R=F(b)-F(x). For a backward move of cost d,
accept R+d <= max(L,R); then the new costs L-d,R+d cannot exceed the old pair
maximum. The forward case is symmetric. Disjoint pairs make this a global
maximum-cost guarantee. Contiguous ordered half-open intervals preserve every
stage exactly once. This is a statement about the source cost model, not time.

The implementation checked 53,960 models: saved draws, empty routing, all
single-job counts 1..64 at grids 1..160, all two-job counts at four grids, and
capacity arithmetic. It checks the unit-start owner inverse and the uint32
weighted-boundary product limit. Two negative examples matter:

- Keeping the old owner is wrong: jobs [17,10,1,2,1,4,5,3,1,3], phase A, grid132,
  unit-start stage96 moves from owner2 to3. Owner2's new interval ends at96.
- Allowing adjacent independently accepted cuts is wrong: jobs [32,32,10,13],
  phase A, grid132 increases the global maximum estimated cost2370 to2528.

Across the saved normal/skewed M32/128/256 draws, median whole split units
removed are only zero to two per phase. Median partial-participant reductions
are about zero to1.64%. Those are reductions in fixup work, not in the whole
kernel. Logical atomic payload and finisher read/zero counts are recorded;
they are not measured HBM traffic. Additional scheduler and owner calculations
are unmeasured. No exposed-tail fraction is established.

Park this exact rule without a GPU check or CUDA implementation. It is distinct
from the earlier unconditional backward snap, but the small opportunity does
not support a new evaluation on its own. The historical phase diagnostic's
timing admission remains false. No source promotion or timing claim follows.

Reproduce: `python experiments/prepared/screen_moe_disjoint_cuts.py`.
The complete model, source/input binding and both counterexamples are in
`screen.json`. No challenge API is called.
