# CPU-only Hopper device-code preflight

Both the retained MoE device code and candidate CUDA9d6c4354 compile to sm_90a
cubins with NVIDIA NVRTC13.0.88 on Windows. No GPU kernel was loaded or launched,
and no official check was used. The candidate's actual submitted source remains
9fc2993a; the proxy makes no edits to it or any verified source.

This is additional evidence for one unchanged paired-store candidate. It is
not a substitute for the official NVCC extension build, runtime resource gate,
whole-suite correctness, CUDA-graph replay or performance evidence.

## Setup and reproducibility

Official NVIDIA wheels were downloaded to `.tools/cuda-compiler-13.0/wheels`,
checked against the published SHA256 values and safely extracted inside the
workspace. No package was installed into Python, no global environment or PATH
was changed, and no paid service was used. `tool-provenance.json` records the
five wheel hashes and four executable/library hashes. Binaries are not stored
in Git. The runtime/compiler/header packages are13.0; the readers are13.4.92.

NVIDIA documents that [NVRTC13.0 can compile without a GPU](https://docs.nvidia.com/cuda/archive/13.0.0/nvrtc/index.html).
The compiler wrapper invokes only NVRTC's compile/output APIs. It never invokes
CUDA driver/runtime kernel-load or kernel-launch APIs. The DLL search path is
limited to that Python process. A direct builtins-DLL preload is needed on
this Windows setup; it does not change PATH.

The exact compiler options, name expressions and header text are saved in
the two successful `.json` records. Both BIG template instantiations were
explicitly requested; this is not a compile of an uninstantiated template.
Logs retain warnings, including any WGMMA serialization diagnostics.

The proxy differs from the judged compilation:

- It removes the Torch host launcher after its include boundary. It does not
  compile the actual resource query that failed in98e3538e85244c1d. The separate
  four-mutant query-binding audit covers that repair; NVRTC alone would not
  have caught that host-query bug.
- It supplies explicit virtual type_traits and stdint headers. The source uses
  only the true/false compile-time tags, and the integer declarations state
  their widths. These are recorded adaptations, not the judge's host headers.
- It uses NVRTC's default-device execution option for unannotated constexpr
  helpers/lambdas and adds a declaration for the existing `wgs_cta_of` function
  before its template caller. The retained baseline failed NVRTC lookup without
  that declaration; the earlier failure is preserved. Removing the inserted
  declaration recovers the exact original device prefix. No function body or
  arithmetic expression changes.
- Windows NVRTC, these headers/options and this compiler build are not asserted
  to reproduce the judge's Linux NVCC toolchain or its resource attributes.

## Static assembly evidence

NVIDIA's nvdisasm13.4.92 read the cubins without executing them. Complete
disassemblies are preserved as deterministic gzip files; their original hashes,
resource dumps, source-line attribution and store examples are in
`assembly-summary.json`.

Each kernel variant's retained phase-A store-helper line has108static
`STG.E.U16` instructions. The new integer store-helper line has54static
`STG.E` instructions and no `STG.E.U16`. This supports the intended replacement
of paired 16-bit stores by32-bit stores in this proxy. These counts include
the compiled tile specializations and are not dynamic per-call counts or
proof of half the epilogue time. Other opcodes attributed to the same debug
line are not independently assigned a runtime cost.

Both proxy variants report96registers,16bytes stack,LOCAL0 andSHARED1024 in
their offline dumps. Ptxas reports56/124bytes of spill stores/loads for the
retained version and60/128for the candidate. Those fields are not equated to
the official runtime `localSizeBytes`/`sharedSizeBytes` queries, and are not
measured spill traffic. No actual residency or resource admission follows.
The official gate remains unchanged: zero runtime local/static shared bytes,
the specified dynamic allocation and at least one resident block/SM.

The deliberate missing-store-helper variant fails compilation as intended.
`negative-control.json` and its log demonstrate that the preflight returns
failure for an unresolved device helper. This is not GPU correctness testing.

## Decision and commands

Keep repaired source9fc2993a unchanged for its one bounded GPU retry after
06:27:37.228843UTC. The new proxy removes one source-compilation uncertainty
and confirms the store-width mechanism; it does not establish a speedup.

Example retained compile, after obtaining and hash-checking the pinned wheels:

```text
python experiments/prepared/local_nvrtc_compile.py --source .tools/nvrtc-preflight/incumbent.cu --prefix-before-torch --minimal-std-tags --forward-wgs-owner --name "&okc::moe_fused<false>" --name "&okc::moe_fused<true>" --define OKC_NS=3 --define OKC_NSF=3 --define OKC_NW=16 --define OKC_FAST1=0 --define OKC_EPI4=0 --define OKC_WGS=1 --define OKC_PRMTSC=1 --define OKC_DQ7=1 --out .tools/nvrtc-preflight/incumbent-main4
```

The candidate uses the exact CUDA9d6c4354 from its frozen packet and namespace
`okc_epair`. Read cubins directly with the workspace nvdisasm binary and
`--print-code --print-line-info`. Run
`python experiments/prepared/summarize_nvrtc_epair.py` on the saved raw outputs.
Keep those raw tool artifacts in `.tools`; do not import this compiler or its
files into a challenge submission.
