# Existing full-context load balancing

CPU source audit only; no implementation, GPU job or latency measurement.
Retained Attention9e922c35 and DecodeD9 have the same launch-plan behavior
after identifier/docstring normalization and exactly the same critical partition
excerpt. Source and CUDA hashes, normalized plan, markers and excerpt are in
audit.json. Reproduce with experiments/prepared/audit_attention_flat_partition.py.

Full-context1<B<=MAX_BATCH selects a flat grid. The L1 path reads actual device
sequence lengths, forms cumulative key counts, adds the retained virtual segment
cost, partitions that global interval by CTA and creates each CTA's request
segment list. It does not allocate equal work assuming every request has80K keys,
and it does not host-read lengths. This is an existing mechanism, not a new
adaptive-splits proposal. No claim applies to every fallback or L8 branch.

Any later redesign must identify an exposed cost left by this mapping, retain
all keys and precision boundaries, distinguish itself from rejected serial
partition-loop trials, and charge extra scheduling, workspace/reduction traffic
and resource costs. Generic unequal-length scheduling does not earn a GPU check.
