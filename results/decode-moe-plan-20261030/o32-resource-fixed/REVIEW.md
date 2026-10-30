# Header-only packaging repair

Job cd6dd28126de466e passed the retained Dense tests but did not exercise O32. The new C++ binding omitted `#include <pybind11/pybind11.h>` while explicitly disabling implicit headers. This was an implementation mistake. The CUDA translation unit emitted only an unused-variable warning; C++ failed at PYBIND11_MODULE.

The repaired payload is exactly the failed payload plus the32-byte escaped Pybind include in `_O32R_CPP`. This header matches the preceding successful projection.cpp. CUDA source/hash, Python runtime, resource queries, data probes, timings and stop conditions are byte-identical. The audit now asserts the required header and exact one-insertion payload relation. Frozen failed payload and evidence remain intact.

Only one retry of this header-only repair is admissible. If the module still fails or the private resource gate still rejects it, stop this diagnostic direction without another retry. No scored call, Decode check or publication. The earlier host-isolation hypothesis remains unconfirmed.
