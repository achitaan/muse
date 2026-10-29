# One-stage ownership screen: parked

This CPU screen reuses the192saved synthetic routing draws from the whole-unit
screen. It changes a boundary s to s-1 only when s modulo16 equals1. It is one
fixed rule, not a threshold sweep, a GPU implementation or a timing result.

The transform is monotone, keeps both endpoints, moves at most one K-stage
backward, and therefore retains complete stage coverage. For a unit's first
stage u, the last transformed boundary <=u is exactly the last old boundary
<=u+1. The cost interval ends at the unit's second stage, so the inverse owner
can use that shifted interval without a new binary search. The CPU audit
checks131,568unit owners over these fixtures. This is a source-algebra/fixture
check, not a CUDA memory-order or barrier proof.

Normal synthetic M32 draws remove a median8split units in either phase, while
the maximum CTA cost ratio rises to a median1.0147in A and1.0303in B. Skewed
M32 draws remove a median0split units and raise the modeled maximum cost to
1.0601/1.0908. Other draws also show a tradeoff. Fewer moved boundaries do not
guarantee that a multiply split unit becomes unsplit.

Do not GPU-test this rule from the CPU screen alone. It removes only a small
part of existing fixup work and has a skewed-tail risk. First require calibrated
phase evidence of an exposed fixup-related cost, a credible load-time saving,
and a complete ownership/counter/replay proof for any eventual exact source.
There is no new runtime candidate, evaluation or publication from this screen.

Reproduce with `python experiments/prepared/screen_moe_one_stage_ownership.py`.
The parent evidence hash and complete rows are preserved in
`results/moe-one-stage-ownership-screen-20261029/screen.json`. CPU cost units
and logical fixups are not GPU nanoseconds, cache misses or measured HBM bytes.
