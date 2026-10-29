"""Offline verification of the generated candidates (CPU + CPU-only NVRTC; no GPU kernel is loaded or launched).

Outputs results/opus-fresh-investigation-20261029/candidate_verification.json
"""
import ast
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_CACHE_DISABLE"] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import nvrtc_screen_alternatives as nv  # noqa: E402

SHIM = "struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };\n"


def module_consts(path):
    import base64, zlib
    src = open(path, encoding="utf-8", newline="").read()
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = n.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out[n.targets[0].id] = v.value
            elif isinstance(v, ast.Call) and getattr(v.func, "id", "") == "_zsrc":
                out[n.targets[0].id] = zlib.decompress(base64.b64decode(v.args[0].value)).decode()
    return src, tree, out


def moe_prefix(cuda):
    prefix = cuda.split("#include <torch/extension.h>", 1)[0]
    fwd = "__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n"
    anchor = "template <bool PB>\n__device__ __forceinline__ void wgs_phase"
    return prefix.replace(anchor, fwd + anchor, 1)


def compile_moe(nvrtc, cuda, defines, lineinfo):
    flags = [d for d in defines] + ["-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3"]
    if not lineinfo:
        flags = ["--no-line-info-placeholder"] + flags
    src = moe_prefix(cuda).encode()
    names = ["&okc::moe_fused<false>", "&okc::moe_fused<true>"]
    # NVRTC.compile always passes --generate-line-info; strip it for byte comparisons by a local variant
    return nvrtc.compile(src, names, [f for f in flags if f != "--no-line-info-placeholder"], tags=True, lineinfo=lineinfo)


def test_moe(nvrtc):
    _, _, base = module_consts(ROOT / "best/moe_verified.py")
    _, _, cand = module_consts(HERE / "moe_finreg.py")
    _, _, dcand = module_consts(HERE / "decoder_finreg.py")
    import finreg_edits
    b, c = base["CUDA_SRC"], cand["CUDA_SRC"]
    assert dcand["_native_moe_CUDA_SRC"] == c, "MoE and Decode candidates must embed identical CUDA"
    # every changed line lies inside the OKC_WGS block
    wgs_start = c.index("// ---------------------------------------------------------------- WGS: RS-wgmma")
    wgs_end = c.index("template <bool PB, bool BIG>\n__device__ __forceinline__ void run_phase")
    assert c.rfind("#if OKC_WGS", 0, wgs_start) > c.rfind("#endif", 0, wgs_start)
    assert c.rfind("#endif", wgs_start, wgs_end) > max(c.index(n) for _, n in finreg_edits.EDITS)
    for _, new in finreg_edits.EDITS:
        pos = c.index(new)
        assert wgs_start < pos < wgs_end
    main = ["-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_PRMTSC=1"]
    tiny = ["-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0",
            "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1", "-DOKC_PRMTSC=1"]
    mid = ["-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4",
           "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1", "-DOKC_PRMTSC=1"]
    res = {}
    for label, defs in (("MAIN_WGS_PRMT", main), ("TINY_PRMT", tiny), ("MID_PRMT", mid)):
        rb = compile_moe(nvrtc, b, defs, True)
        rc = compile_moe(nvrtc, c, defs, True)
        res[label] = {"baseline": {"rc": rb["rc"], "cubin_bytes": rb["cubin_bytes"], "cubin_sha256": rb["cubin_sha256"], "kernels": nv.parse_ptxas(rb["log"])},
                      "candidate": {"rc": rc["rc"], "cubin_bytes": rc["cubin_bytes"], "cubin_sha256": rc["cubin_sha256"], "kernels": nv.parse_ptxas(rc["log"])}}
    # preprocessed device code of the non-WGS builds must be identical: compare after removing the WGS region text
    res["inverse_restores_parent_cuda"] = finreg_edits.invert(c) == b
    res["all_edits_inside_the_OKC_WGS_region"] = True
    res["TINY_MID_cubin_sha256_identical_with_lineinfo"] = {k: res[k]["baseline"]["cubin_sha256"] == res[k]["candidate"]["cubin_sha256"] for k in ("TINY_PRMT", "MID_PRMT")}
    res["TINY_MID_cubin_sha256_identical_without_lineinfo"] = {label: compile_moe(nvrtc, b, defs, False)["cubin_sha256"] == compile_moe(nvrtc, c, defs, False)["cubin_sha256"] for label, defs in (("TINY_PRMT", tiny), ("MID_PRMT", mid))}
    return res


def test_o128(nvrtc):
    out = {}
    common = (HERE / "o128_common.py").read_text(encoding="utf-8")
    for label, path in (("dense", HERE / "dense_o128.py"), ("decoder", HERE / "decoder_o128.py")):
        src, tree, consts = module_consts(path)
        ns = {k: consts[k] for k in ("tt_mw_CUDA_PREFIX", "tt_mw_CUDA_CORE", "tt_mw_CUDA_MAP", "tt_mw_CUDA_HOST")}
        # tt_mw_mma from the candidate itself
        fn_mma = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tt_mw_mma")
        exec(compile(ast.Module([fn_mma], []), "tt_mw_mma", "exec"), ns)
        cm = ast.parse(common)
        keep = [n for n in cm.body if (isinstance(n, ast.Assign)) or (isinstance(n, ast.FunctionDef) and n.name == "_o128_source")]
        exec(compile(ast.Module(keep, []), "o128_source", "exec"), ns)
        full = ns["_o128_source"]()
        assert consts["tt_mw_CUDA_CORE"].count(ns["_O128_OLD"]) == 1
        device = full.split(consts["tt_mw_CUDA_MAP"])[0].replace("#include <cuda.h>", SHIM)
        names = ["&m32w_gemm<128,64,1,2,2,true,2>", "&m32w_gemm<128,64,2,2,2,true,1>"]
        r = nvrtc.compile(device.encode(), names, tags=True)
        host = full.split(consts["tt_mw_CUDA_MAP"])[1]
        # static name binding: every global referenced by appended functions is defined at module level
        defined = set()
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            defined.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    defined.add((a.asname or a.name).split(".")[0])
        import builtins
        appended = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (n.name.startswith("_o128") or n.lineno > len(src.splitlines()) - 200)]
        missing = set()
        for fn in appended:
            local = {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
            local |= {x.id for x in ast.walk(fn) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
            local |= {x.name for x in ast.walk(fn) if isinstance(x, ast.ExceptHandler) and x.name}
            for x in ast.walk(fn):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id not in local and x.id not in defined and not hasattr(builtins, x.id):
                    missing.add(x.id)
        # smem budget per launch_t formula for both distinct template instances
        smem = {}
        for wk in (1, 2):
            a = wk * ((2 * (128 * 128 + 8) + 1023) // 1024 * 1024)
            bb = wk * 128 * (64 + 4) * 4
            smem[f"WK{wk}"] = max(a, bb)
        out[label] = {"nvrtc_rc": r["rc"], "kernels": nv.parse_ptxas(r["log"]), "cubin_bytes": r["cubin_bytes"],
                      "host_dispatch_cases": host.count("case "), "unbound_globals": sorted(missing),
                      "dynamic_smem_bytes": smem, "smem_limit": 232448}
    return out


def epilogue_coverage(BN=128, R=64, WK=2, split=2):
    # wgmma m64nN f32 D-fragment (PTX ISA): thread (warp w, lane l) holds rows 16w + l/4 + 8h, cols 8j + 2(l%4) + c
    # at acc[4j + 2h + c]; the kernel stores exactly that mapping, then sums WK*split partials per (token,row).
    seen = np.zeros((BN, R), dtype=int)
    for w in range(4):
        for l in range(32):
            row, col = w * 16 + l // 4, 2 * (l % 4)
            for j in range(BN // 8):
                for h in range(2):
                    for c in range(2):
                        seen[col + 8 * j + c, row + 8 * h] += 1
    reduce_items = set()
    NT = 128 * WK
    for rank in range(split):
        for tid in range(NT):
            for q in range(tid + rank * NT, BN * (R // 4), NT * split):
                reduce_items.add(q)
    return {"each_token_row_written_once_per_warpgroup": bool((seen == 1).all()),
            "final_reduction_covers_all_items_once": reduce_items == set(range(BN * (R // 4)))}


def finisher_arithmetic(trials=20000, seed=7):
    # Original: fb starts at +0; every share (finisher included) adds its FP32 partial atomically in arbitrary order;
    # the finisher reads fb. FINREG: non-finisher shares add atomically into +0; the finisher adds that sum to its
    # registers. Check FINREG's result is bitwise one of the results the original's admissible orders produce.
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        k = int(rng.integers(2, 4))
        parts = (rng.standard_normal(k) * 10.0 ** rng.integers(-3, 4, size=k)).astype(np.float32)
        if rng.random() < 0.05:
            parts[int(rng.integers(0, k))] = np.float32(-0.0)
        fin, others = parts[0], parts[1:]
        allowed = set()
        for perm in itertools.permutations(range(k)):
            acc = np.float32(0.0)
            for i in perm:
                acc = np.float32(acc + parts[i])
            allowed.add(acc.tobytes())
        for perm in itertools.permutations(range(k - 1)):
            s = np.float32(0.0)
            for i in perm:
                s = np.float32(s + others[i])
            new = np.float32(fin + s)
            if new.tobytes() not in allowed:
                bad += 1
    return {"trials": trials, "finreg_results_outside_original_admissible_set": bad}


def main():
    nvrtc = nv.NVRTC()
    res = {
        "manifest_sha256": hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest(),
        "moe_finreg_nvrtc": test_moe(nvrtc),
        "o128_nvrtc_and_static": test_o128(nvrtc),
        "o128_epilogue_coverage_BN128": epilogue_coverage(),
        "finreg_fp32_order_model": finisher_arithmetic(),
        "gpu_kernels_loaded_or_launched": False,
        "official_checks_used": 0,
    }
    out = ROOT / "results/opus-fresh-investigation-20261029/candidate_verification.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
