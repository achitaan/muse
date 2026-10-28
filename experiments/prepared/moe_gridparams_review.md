# MoE parameter memory

One annotation-only CUDA trial on exact retained33114871. Do not inherit the
rejected scale-six, WGS32 or scheduler-quotient experiments. The kernel's Params
object is passed by value to moe_fused and then by const reference to numerous
device helpers, including out-of-line routing and scheduler functions. This
can require per-thread parameter copies. Marking it const __grid_constant__
allows the compiler to retain its grid-lifetime parameter-space address.

NVIDIA documents this behavior and the read-only object requirement:
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
The checked native attention source already compiles this annotation with the
same official toolkit. This establishes availability, not a MoE speedup.

The Params object contains mutable pointees (scratch, output, counters) whose
contents must continue changing. The pointer values and scalar fields inside
the Params object are not mutated during the kernel. All device references to
the object are const; there are no pointer aliases or const_cast operations on
the object. The native host still constructs and initializes its ordinary Params
before launch. No forward kernel declaration/specialization needs updating.

The build/audit script changes only the kernel declaration and a local-change
comment. It reverses the exact CUDA patch to recover the original complete
Python module AST and checks the entire kernel body/suffix byte-for-byte.
All host dispatch, CUDA arithmetic, weights, precision, routing, scratch,
barriers, graph replay behavior and fresh-output allocations remain original.
No additional compilation variants, buffers, selectors or runtime operations.

One full official check using cached baseline2.365437416x, standard5tests and
6benchmarks plus integrity/precision guard. All six loads may be affected,
because they compile the same kernel declaration under different defines.
Original WGS resource attributes96registers/0local bytes do not prove whether
parameter copies exist in an outlined call; do not promise elimination or
register savings without generated-code/runtime evidence. If timing is flat,
do not score a merely cleaner declaration or retry it for a favorable point.
