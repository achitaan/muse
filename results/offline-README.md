# Offline SM90 compilation

Completed on 2026-10-23 using the official PyPI Triton 3.8.0 Linux cp314 wheel
and WSL Ubuntu Python 3.14.4. No Torch installation, GPU launch, or official
challenge check was used by this setup.

All 17 representative candidate specializations passed compilation through
cubin for `GPUTarget("cuda", 90, 32)`. This covers GEMV K=1024/4096/8192,
small GEMM including M=5/7 masks and unpooler transpose stride, router products
and FP32 reduction, large GEMM including M=33/130/1000 masks and split K, and
the large BF16 reduction.

The machine-readable details and compiler metadata are in
`results/offline-compile.json`. Kernel bodies are extracted without Torch host
code into `results/offline-*-kernels.py`; the AST transformation preserves their
meaning. All compiler cache files are project-local under
`.tools/offline-triton/cache`.

Reproduce from PowerShell:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/c/Users/Achita/Documents/ChatGPT/Muse && python3 scripts/offline_compile.py --suite all'
```

This confirms compiler lowering only. The official environment may have a
different Triton version. GPU launch validity, numerical correctness, real
resource limits, and performance still require the parent-owned H200 checks.
