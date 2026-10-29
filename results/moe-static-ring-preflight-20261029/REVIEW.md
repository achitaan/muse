# MoE constant ring indices: CPU drafts held back

Retained native parent1e60cf46, Python source33114871. The two drafts change
only the WGS consumer's fixed three-slot ring indexing. Tile geometry, ring
depth, math, commit/wait/release ordering, fixups, epilogues and fallbacks
remain. V1 uses three explicit slot bodies and a parity recurrence; V2
removes the separately live parity and keeps the original seq/3 expression.
These are different source implementations, not a parameter sweep.

The idea was suggested by a [public firsthand discussion](https://guild.1nterface.ai/t/18?credible=all).
Its reported performance is not reproduced here and is not our evidence.
Our implementation was written against the retained source. No messages,
account creation, external agents or code exchange were used.

The V1 finite model checks65536 starting-sequence/segment combinations and
rejects four ordering/parity mutants. The inverse edit ledger restores the
exact parent. This does not establish GPU correctness or race freedom.

The matched offline compiler uses the same flags on all three inputs. The
parent has96registers,16stack bytes and56/124spill store/load bytes. Both
drafts have96registers and no spills, but newly trigger C7512 in both kernel
entries. Native NVCC resources can differ from this NVRTC proxy; the
retained GPU kernel is not proven faulty by the proxy's stack report.

SASS confirms the adverse change. Per kernel, the parent has512HGMMA,
256waits allowing one group in flight and16zero waits. Each draft has
1024HGMMA and1024zero waits, with no one-group waits. Static instruction
counts rise25088->38304(V1) or38064(V2). This is static generated-code
evidence, not a GPU latency measurement. Zero spills alone is insufficient
admission when the asynchronous pipeline is serialized.

Neither draft gets a Python candidate or GPU check. Preserve both failures;
do not try compiler/register flags hoping for a win. Reopening needs a
specific source mechanism that restores the pipeline with matched resource
and SASS evidence, followed by numerical/replay and complete-load gates.
