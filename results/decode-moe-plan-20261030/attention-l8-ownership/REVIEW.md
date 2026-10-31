# Retained L8 synchronization screen

Static source review only, 2026-10-31 UTC. No candidate, compile, GPU check or measured gain.

Binding: `best/attention_checked.py` SHA256 `9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f`. Its literal `CUDA_SRC` exactly matches `../attention-nc1/parent.cu`, SHA256 `241b67fe0008bbae53435b53d1bb6d647ded0b5718118875c0ec368b718b41db`. Line references below refer to that saved CUDA source. Python `_build` passes this literal directly to the extension builder.

The L8 loop already has the main proposed reuse and overlap mechanisms:

- Lines 1581 onward describe a warpgroup per KV head and 64 query rows, covering eight tokens and eight query heads. A generic proposal to reuse historical K/V across these rows is already represented here.
- Lines 1700 onward wait for the current producer stage. In the producer path the full consumer synchronization is conditional on the initial tile or patched new rows. Removing a claimed unconditional per-tile CTA barrier would target a boundary that this path already avoids.
- Lines 1739 onward wait for previous PV completion before reusing probability registers; the code fences the associated accumulators. V transpose storage is double buffered. Lines 1790 onward release the K/V stage only after its asynchronous readers complete.
- Lines 1910 onward publish V transpose stores with an asynchronous-proxy fence and a warpgroup barrier before the PV WGMMA reads them. Later probability packing overlaps asynchronous matrix work. Generic Hopper overlap is already present.

These are source observations, not a proof that the implementation is optimal or a timing measurement. They provide no exposed barrier-tail duration and no safe redundant wait to remove. A different design would require a concrete dependency and lifetime argument, actual selected-family evidence and a recoverable-time bound. No GPU admission follows from this screen; the frozen M1 diagnostic remains next.
