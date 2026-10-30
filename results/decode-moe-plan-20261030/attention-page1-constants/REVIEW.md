# Metadata specialization: CPU rejected

The retained producer already specializes the common page-one,small-cache address calculation at runtime. This screen tests compile-time propagation of the same metadata family: L1,Hq16,Hkv2,P1,row128bytes and caches below2^32bytes. Physical page indices remain fresh device inputs. All other shapes would require the original fallback;no Python dispatch payload was built.

Only seven metadata field expressions are substituted in the device prefix. An exact inverse recovers the original source. Arithmetic,cache writes,masks,barriers,launch plans and key coverage remain original under that eligibility contract. This differs from the failed finalizer outlining experiment.

Matched L1-flat compilation retains128registers,zero stack/spills and no local instructions. The entry shrinks3560to3400instructions,ratio0.95505618. It misses the predeclared5%footprint screen;do not move the threshold or infer a latency saving. CUDA `cbdf9f3fa7b1bfaa4a6a81ac2b3091ac93b344c299df4384d3528400f9b47c01` is rejected before GPU use.

Plain bulk-copy substitution was also screened out by source inspection: the existing cache tiles use a row-dependent16-byte XOR swizzle,and page-one rows may be physically scattered. A contiguous bulk copy is not a drop-in equivalent or a demonstrated traffic reduction. Page-table TMA staging already exists and was not reintroduced.

No GPU check,score or Decode transfer. Static instruction counts include embedded leaf code and do not measure dynamic instructions or instruction-cache pressure.
