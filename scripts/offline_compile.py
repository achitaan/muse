"""Compile candidate Triton JIT definitions for SM90 without Torch or a GPU.

Run in WSL: python3 scripts/offline_compile.py [--suite small|all]
Only checks compiler lowering; it cannot validate outputs, launches, or timing.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".tools" / "offline-triton"
sys.path.insert(0, str(LOCAL / "site-packages"))
os.environ.setdefault("TRITON_CACHE_DIR", str(LOCAL / "cache"))


def load_jit_definitions(path: Path):
    """Retain JIT bodies unchanged in meaning, excluding host Torch code."""
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and any(ast.unparse(deco).startswith("triton.jit")
                         for deco in node.decorator_list)]
    source = "import triton\nimport triton.language as tl\n\n"
    source += "\n\n".join(ast.unparse(fn) for fn in functions) + "\n"
    destination = ROOT / "results" / f"offline-{path.stem}-kernels.py"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(source)
    spec = importlib.util.spec_from_file_location("offline_" + path.stem, destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cases(suite: str):
    # Pointer signatures preserve real input/output precision; all other arguments
    # are compile-time values exactly as used by the candidate launch wrappers.
    if suite == "followup":
        for m, n, k, wn, split in [(8,8192,1024,8192,1), (16,8192,4096,8192,1),
                                   (7,4096,8192,4096,2), (5,1024,4096,8192,8)]:
            yield ("small_adaptive", "_split_k_bf16", {"X":"*bf16", "WT":"*bf16", "Partial":"*bf16" if split == 1 else "*fp32"},
                   dict(M=m,N=n,K=k,WEIGHT_N=wn,BLOCK_M=16,BLOCK_N=64,BLOCK_K=64,SPLIT_K=split),
                   {"num_warps":4,"num_stages":3})
        yield ("router_gemv", "_router_gemv", {"X":"*bf16","W":"*fp32","P":"*fp32"},
               dict(N=256,K=8192,BN=4,BK=512,SPLITS=4), {"num_warps":4})
        return
    for n, k in [(2304, 8192), (8192, 1024), (1024, 4096)]:
        yield ("small_batch", "_gemv_contiguous_k", {"X": "*bf16", "W": "*bf16", "Y": "*bf16"},
               dict(N=n, K=k, BLOCK_N=4, BLOCK_K=1024), {"num_warps": 4})
    for m, n, k, wn in [(7, 4096, 8192, 4096), (5, 1024, 4096, 8192),
                         (16, 2304, 8192, 2304), (32, 8192, 1024, 8192)]:
        split = 16 if n <= 1024 and k >= 4096 else 8
        yield ("small_batch", "_split_k_bf16", {"X": "*bf16", "WT": "*bf16", "Partial": "*fp32"},
               dict(M=m, N=n, K=k, WEIGHT_N=wn, BLOCK_M=16 if m <= 16 else 32,
                    BLOCK_N=64, BLOCK_K=64, SPLIT_K=split), {"num_warps": 4, "num_stages": 3})
    yield ("small_batch", "_finish_split_k", {"Partial": "*fp32", "Y": "*bf16"},
           dict(ELEMENTS=5*1024, SPLIT_K=16, BLOCK=256), {"num_warps": 4})
    if suite == "small":
        return
    for m in (9, 128):
        split = 16 if m <= 32 else 4
        yield ("router_only", "_router_products", {"X": "*bf16", "HI": "*bf16", "LO": "*bf16", "P": "*fp32"},
               dict(M=m, N=256, K=8192, BM=16 if m <= 16 else 32, BN=32,
                    BK=64, SPLITS=split), {"num_warps": 4, "num_stages": 2})
    yield ("router_only", "_router_reduce", {"P": "*fp32", "Y": "*fp32"},
           dict(SIZE=9*256, SPLITS=16, BLOCK=256), {"num_warps": 4})
    for m, n, k, split in [(32, 2304, 8192, 4), (33, 8192, 1024, 1),
                            (128, 4096, 8192, 1), (130, 8192, 4096, 1),
                            (1000, 8192, 4096, 1)]:
        yield ("large_batch", "_large_gemm", {"X": "*bf16", "W": "*bf16", "OUT": "*fp32" if split > 1 else "*bf16"},
               dict(M=m, N=n, K=k, XSM=k, XSK=1, WSN=k, WSK=1,
                    BM=32 if m <= 128 else 64, BN=64, BK=64,
                    SPLIT_K=split, GROUP_M=4), {"num_warps": 4, "num_stages": 3})
    yield ("large_batch", "_large_split_reduce", {"PARTIALS": "*fp32", "OUT": "*bf16"},
           dict(SIZE=32*2304, SPLIT_K=4, BLOCK=1024), {"num_warps": 4})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("small", "all", "followup"), default="all")
    args = parser.parse_args()
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    modules = {}
    results = []
    target = GPUTarget("cuda", 90, 32)
    print(f"Triton {triton.__version__}, target={target}; compile only, no GPU launch", flush=True)
    for filename, name, signature, constants, options in cases(args.suite):
        if filename not in modules:
            modules[filename] = load_jit_definitions(ROOT / "candidates" / f"{filename}.py")
        fn = getattr(modules[filename], name)
        started = time.monotonic()
        item = dict(file=filename, kernel=name, constants=constants, options=options)
        try:
            compiled = triton.compile(ASTSource(fn, signature=signature, constexprs=constants), target=target, options=options)
            item.update(status="pass", seconds=time.monotonic()-started,
                        stages=list(compiled.asm), metadata=repr(compiled.metadata))
        except Exception:
            item.update(status="fail", seconds=time.monotonic()-started, error=traceback.format_exc())
        results.append(item)
        print(f"{item['status'].upper()} {filename}.{name} {constants}", flush=True)
        if item["status"] == "fail":
            print(item["error"], flush=True)
        result_file = "offline-compile.json" if args.suite == "all" else f"offline-compile-{args.suite}.json"
        (ROOT / "results" / result_file).write_text(json.dumps(dict(triton=triton.__version__, target=repr(target), results=results), indent=2))
    return int(any(item["status"] != "pass" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
