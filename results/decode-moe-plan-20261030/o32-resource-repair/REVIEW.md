# Fixed O32 resource initialization repair

The preceding check 3264689ee9974ef9 did contain the exact O32 plan. Its resource query returned 42 registers, zero local bytes, but zero resident blocks at 196736 dynamic shared bytes. Consequently no O32 correctness or timing probe ran. Earlier wording that the plan itself was absent was incomplete.

The frozen twin stripped `r3_o32_o_direct_a_ol_` to generic `ol_` host names. The retained Dense OLAYER_CUDA already defines the identical `ol_launch_t<32,128,8,1,1,1,1>` host template signature with its own `static bool configured`. The private CUDA kernel has a different name, so sharing the flag could skip the private kernel's shared-memory opt-in. These duplicate host symbols are a confirmed isolation defect; their causal role in the H200 result remains a hypothesis.

GCC documents unique binding across RTLD_LOCAL modules for relevant C++ static state. The actual judge's symbol binding has not been inspected. A minimal local ABI reproducer was written but could not compile because WSL has no g++; no compiler was installed. [GCC code generation options](https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html)

The repair adds a uniquely named host launcher/map/template while retaining the entire old CUDA source as an exact prefix. The legacy host entry is called only with query=1 and null data pointers, never launched. The new host code is an inverse-exact spelling change of the old host code. All projection arithmetic, resource geometry, device wait placement, producer, probes, timing harness and gates remain unchanged.

One prepare-only diagnostic records function attributes and occupancy before initialization, after the legacy query, and after the isolated query. CUDA exposes the dynamic-memory ceiling, static memory, register/local usage and thread limit used here. [NVIDIA function attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html)

Continue only if the isolated kernel has exactly 196736 opted-in bytes, zero local bytes, legal shared-memory capacity and positive queried residency. Then measure only the previously unexercised fixed O32 question, with changing normal/zero/outlier byte probes and both cold-L2 graph arm orders. More than 3% and at least 1.5us must hold in both orders. This remains an optimistic synthetic-producer ceiling; it enables no Decode or Dense path.

If residency remains zero, reject without a launch. If bytes or either timing gate fails, close O32 without tuning or another retry. Do not rerun the five completed families. Preserve the final Decode check and all best sources. No scored submission or board publication follows from this isolated result.

This is a bounded repair of a source-isolation defect that prevented the planned probe from running, rather than a repeat of unchanged timed code. Any spacing exception must be recorded before upload. Source binding and CPU audit are adjacent; H200 compilation and the actual diagnosis remain outstanding until the job completes.
