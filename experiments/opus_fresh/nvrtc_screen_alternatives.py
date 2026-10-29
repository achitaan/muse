"""CPU-only NVRTC compilation and mathematical verification of the independent kernel alternatives.

Never loads or launches a GPU kernel, never calls okc or any network API.
Uses the pinned workspace NVRTC 13.0.88 DLLs in C:/Users/Achita/Documents/ChatGPT/Muse 2/.tools/cuda-compiler-13.0.
"""
import ctypes as c
import hashlib
import json
import os
os.environ["CUDA_CACHE_DISABLE"] = "1"
import re
import time
from pathlib import Path
import torch

WORKTREE = Path(__file__).resolve().parents[2]
MAIN_ROOT = WORKTREE.parents[2]
TOOL = MAIN_ROOT / ".tools/cuda-compiler-13.0"

TAGS = """namespace std {
template<class T, T V> struct integral_constant {
  static constexpr T value = V;
  using value_type = T;
  using type = integral_constant;
  __host__ __device__ constexpr operator value_type() const noexcept { return value; }
  __host__ __device__ constexpr value_type operator()() const noexcept { return value; }
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
}
"""
STDINT = """typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long long int64_t;
typedef unsigned long long uint64_t;
typedef int64_t intptr_t;
typedef uint64_t uintptr_t;
"""


class NVRTC:
    def __init__(self):
        self.binary = TOOL / "nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll"
        self.search = os.add_dll_directory(str(self.binary.parent))
        self.builtins = c.CDLL(str(self.binary.parent / "nvrtc-builtins64_130.dll"))
        self.lib = c.CDLL(str(self.binary))
        self.bind("nvrtcVersion", [c.POINTER(c.c_int), c.POINTER(c.c_int)])
        self.bind("nvrtcCreateProgram", [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p,
                                         c.c_int, c.POINTER(c.c_char_p), c.POINTER(c.c_char_p)])
        self.bind("nvrtcAddNameExpression", [c.c_void_p, c.c_char_p])
        self.bind("nvrtcCompileProgram", [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)])
        self.bind("nvrtcDestroyProgram", [c.POINTER(c.c_void_p)])
        for kind in ("ProgramLog", "PTX", "CUBIN"):
            self.bind("nvrtcGet" + kind + "Size", [c.c_void_p, c.POINTER(c.c_size_t)])
            self.bind("nvrtcGet" + kind, [c.c_void_p, c.c_void_p])
        self.lib.nvrtcGetErrorString.argtypes = [c.c_int]
        self.lib.nvrtcGetErrorString.restype = c.c_char_p

    def bind(self, name, args):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = args, c.c_int

    def check(self, rc):
        if rc:
            raise RuntimeError(self.lib.nvrtcGetErrorString(rc).decode())

    def output(self, prog, kind):
        size = c.c_size_t()
        self.check(getattr(self.lib, "nvrtcGet" + kind + "Size")(prog, c.byref(size)))
        buf = c.create_string_buffer(size.value)
        self.check(getattr(self.lib, "nvrtcGet" + kind)(prog, buf))
        return bytes(buf.raw)

    def compile(self, source: bytes, names, extra_flags=(), tags=True):
        prog = c.c_void_p()
        headers = (c.c_char_p * 2)(TAGS.encode(), STDINT.encode()) if tags else None
        includes = (c.c_char_p * 2)(b"type_traits", b"stdint.h") if tags else None
        self.check(self.lib.nvrtcCreateProgram(c.byref(prog), source, b"preflight.cu",
                                               2 if tags else 0, headers, includes))
        inc1 = str(TOOL / "nvidia/cu13/include")
        inc2 = str(TOOL / "nvidia/cu13/include/cccl")
        flags = [
            "--gpu-architecture=sm_90a",
            "--std=c++20",
            "--generate-line-info",
            "--device-as-default-execution-space",
            "--ptxas-options=-v",
            f"--include-path={inc1}",
            f"--include-path={inc2}",
            *extra_flags,
        ]
        try:
            for n in names:
                self.check(self.lib.nvrtcAddNameExpression(prog, n.encode()))
            opts = (c.c_char_p * len(flags))(*(f.encode() for f in flags))
            t0 = time.monotonic()
            rc = self.lib.nvrtcCompileProgram(prog, len(flags), opts)
            dt = time.monotonic() - t0
            log = self.output(prog, "ProgramLog").rstrip(b"\0").decode(errors="replace")
            cubin = self.output(prog, "CUBIN") if rc == 0 else b""
            return {"rc": rc, "log": log, "cubin_bytes": len(cubin), "seconds": round(dt, 3)}
        finally:
            self.check(self.lib.nvrtcDestroyProgram(c.byref(prog)))


def parse_ptxas(log: str):
    out = []
    entries = re.split(r"ptxas info\s*:\s*Compiling entry function '([^']+)'", log)
    for i in range(1, len(entries), 2):
        sym = entries[i]
        body = entries[i + 1]
        m_stack = re.search(r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads", body)
        m_regs = re.search(r"Used\s+(\d+)\s+registers", body)
        m_smem = re.search(r"(\d+)\s+bytes smem", body)
        warns = re.findall(r"ptxas warning\s*:\s*([^\r\n]+)", body)
        out.append({
            "symbol": sym,
            "registers": int(m_regs.group(1)) if m_regs else None,
            "stack_frame": int(m_stack.group(1)) if m_stack else 0,
            "spill_stores": int(m_stack.group(2)) if m_stack else 0,
            "spill_loads": int(m_stack.group(3)) if m_stack else 0,
            "smem_static": int(m_smem.group(1)) if m_smem else 0,
            "warnings": warns,
        })
    return out


def test_m128w(nvrtc: NVRTC):
    # Extract tt_mw_CUDA_PREFIX and tt_mw_CUDA_CORE from best/decoder_verified.py
    import ast, base64, zlib
    src_py = (WORKTREE / "best/decoder_verified.py").read_text(encoding="utf-8")
    tree = ast.parse(src_py)
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "tt_mw_CUDA_PREFIX":
                vals[name] = node.value.value
            elif name == "tt_mw_CUDA_CORE":
                vals[name] = zlib.decompress(base64.b64decode(node.value.args[0].value)).decode()

    def mma_fn(n):
        nr = n // 2
        regs = ",".join(f"%{i}" for i in range(nr))
        outs = ",".join(f'"+f"(d[{i}])' for i in range(nr))
        inputs = ",".join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
        ops = ["wgmma.fence.sync.aligned;"]
        for kk in range(4):
            ar = ",".join(f"%{nr + 4 * kk + i}" for i in range(4))
            ops.append(f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;")
        ops.append("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;")
        return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + " ".join(ops) + f'" : {outs} : {inputs} : "memory");}}'

    # Also add a dual-mma64 helper for BN=128 that reuses the same decoded a[16] across two 64-token tiles!
    core_ext = vals["tt_mw_CUDA_CORE"].replace(
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));",
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==128){mma64(acc,a,desc(base+s*SB));mma64(acc+32,a,desc(base+s*SB+8192));}"
    )
    core_128_direct = vals["tt_mw_CUDA_CORE"].replace(
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));",
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));"
    )
    # NVRTC does not include driver-API cuda.h's CUtensorMap by default; provide the 128B aligned struct
    cutensormap_shim = "struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };\n"
    prefix_nvrtc = vals["tt_mw_CUDA_PREFIX"].replace("#include <cuda.h>", cutensormap_shim)
    cu_dual = prefix_nvrtc + "\n".join(mma_fn(n) for n in (16, 32, 64)) + core_ext
    cu_direct = prefix_nvrtc + "\n".join(mma_fn(n) for n in (16, 32, 64, 128)) + core_128_direct

    names = [
        "&m32w_gemm<32, 64, 2, 2, 2, true, 2>",
        "&m32w_gemm<64, 64, 2, 2, 2, true, 2>",
        "&m32w_gemm<128, 64, 2, 2, 2, true, 1>",
        "&m32w_gemm<128, 64, 1, 2, 2, true, 2>",
    ]
    r_dual = nvrtc.compile(cu_dual.encode(), names, tags=True)
    r_direct = nvrtc.compile(cu_direct.encode(), names, tags=True)
    return {
        "dual_mma64_reuse_a16": {"rc": r_dual["rc"], "cubin_bytes": r_dual["cubin_bytes"], "seconds": r_dual["seconds"], "kernels": parse_ptxas(r_dual["log"])},
        "direct_mma128": {"rc": r_direct["rc"], "cubin_bytes": r_direct["cubin_bytes"], "seconds": r_direct["seconds"], "kernels": parse_ptxas(r_direct["log"])},
    }


def test_wgs_finisher_inreg(nvrtc: NVRTC):
    raw = (WORKTREE / "experiments/opus_fresh/scratch/moe.cu").read_text(encoding="utf-8")
    prefix = raw.split("#include <torch/extension.h>", 1)[0]
    forward = "__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n"
    anchor = "template <bool PB>\n__device__ __forceinline__ void wgs_phase"
    prefix_base = prefix.replace(anchor, forward + anchor, 1)

    # Alternative 2: in wgs_seg, when DF is active for this phase and this share is the designated finisher
    # (PB && s0 == 0), keep d[nt][0..3] in registers, wait for the non-finishers on fc, add __ldcg(fb) in registers,
    # zero fb and fc, and fall straight through to line 2087's unrolled wgs_epi_tile<PB> loop!
    old_flush_df = """  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {"""

    new_flush_df = """  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");
  pd.kind = 0;
  return;
  } else {"""

    old_seg_tail = """  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }"""

    new_seg_tail = """  if (slot >= 0) {  // a share of a unit split between CTAs
    constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
    if (DF && (PB ? s0 == 0 : s1 == 16)) {
      int* fc = wgs_cnt(p, PB) + slot * 4 + wg;
      if (wt == 0) { while (ld_acquire(fc) < 16 - (s1 - s0)) __nanosleep(32); }
      asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
      asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
        *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        d[nt][0] += v.x; d[nt][1] += v.y; d[nt][2] += v.z; d[nt][3] += v.w;
      }
      if (wt == 0) *fc = 0;
    } else {
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt)
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                     "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
      pd = WgsPend{2, u, slot, s1 - s0};
      return;
    }
  }"""

    assert old_flush_df in prefix_base, "old_flush_df not found"
    assert old_seg_tail in prefix_base, "old_seg_tail not found"
    prefix_new = prefix_base.replace(old_flush_df, new_flush_df, 1).replace(old_seg_tail, new_seg_tail, 1)

    flags = ["-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_NW=16", "-DOKC_FAST1=0",
             "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_PRMTSC=1", "-DOKC_DQ7=1"]
    names = ["&okc::moe_fused<false>", "&okc::moe_fused<true>"]
    r_base = nvrtc.compile(prefix_base.encode(), names, flags, tags=True)
    r_new = nvrtc.compile(prefix_new.encode(), names, flags, tags=True)
    return {
        "baseline": {"rc": r_base["rc"], "cubin_bytes": r_base["cubin_bytes"], "seconds": r_base["seconds"], "kernels": parse_ptxas(r_base["log"])},
        "wgs_finisher_inreg": {"rc": r_new["rc"], "cubin_bytes": r_new["cubin_bytes"], "seconds": r_new["seconds"], "kernels": parse_ptxas(r_new["log"])},
    }


def test_rf_vec4_smem(nvrtc: NVRTC):
    rf_base = (WORKTREE / "experiments/opus_fresh/scratch/rf.cu").read_text(encoding="utf-8")
    # Extract device code before void run(...)
    dev_base = rf_base.split("void run(", 1)[0]
    for h in ("#include <ATen/core/Tensor.h>", "#include <ATen/cuda/CUDAContext.h>",
              "#include <c10/cuda/CUDAException.h>", "#include <cuda_runtime.h>",
              "#include <math.h>", "#include <limits.h>"):
        dev_base = dev_base.replace(h, "")
    shim = "#define INFINITY __int_as_float(0x7f800000)\n#define INT_MAX 2147483647\n"
    dev_base = shim + dev_base

    # Alternative 4: stage x[TM, 1024] into shared memory once per CTA using float4 loads,
    # so all 8 warps read x from shared memory instead of issuing 8x redundant global loads!
    # Preserves the exact same per-lane k = ks * 1024 + j * 32 + lane and __fmaf_rn order.
    dev_smem = dev_base.replace(
        "  float a[TM], b[TM];",
        "  __shared__ float sx[TM][1024];\n"
        "  for (int idx = tid; idx < TM * 256; idx += 256) {\n"
        "    const int m = idx >> 8, c4 = (idx & 255) << 2;\n"
        "    if (m < M) *reinterpret_cast<float4*>(&sx[m][c4]) = *reinterpret_cast<const float4*>(&x[(size_t)m * 8192 + ks * 1024 + c4]);\n"
        "  }\n"
        "  __syncthreads();\n"
        "  float a[TM], b[TM];"
    ).replace(
        "const float v = x[(size_t)m * 8192 + k];",
        "const float v = sx[m][j * 32 + lane];"
    )
    names = ["&route_fp32<1>", "&route_fp32<8>"]
    r_base = nvrtc.compile(dev_base.encode(), names, tags=True)
    r_smem = nvrtc.compile(dev_smem.encode(), names, tags=True)
    return {
        "baseline": {"rc": r_base["rc"], "cubin_bytes": r_base["cubin_bytes"], "kernels": parse_ptxas(r_base["log"])},
        "smem_vec4_x": {"rc": r_smem["rc"], "cubin_bytes": r_smem["cubin_bytes"], "kernels": parse_ptxas(r_smem["log"])},
    }


def cpu_math_checks():
    # 1. Verify c12 bitwise round-trip on M=128 pooler/unpooler synthetic weights including -0.0 and outliers
    gen = torch.Generator().manual_seed(20261029)
    w = (torch.randn((256, 512), generator=gen) / 64.0).to(torch.bfloat16)
    w[0, 0] = torch.tensor(-0.0, dtype=torch.bfloat16)
    w[1, 1] = torch.tensor(0.0, dtype=torch.bfloat16)
    w[2, 2] = torch.tensor(1e-30, dtype=torch.float32).to(torch.bfloat16)
    w[3, 3] = torch.tensor(256.0, dtype=torch.bfloat16)
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(256 // 16, 2, 8, 512 // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(16, 8, 32, 32)
    # Run exact c12 pack/unpack
    sign_pos = torch.tensor((3, 19, 7, 23, 11, 27, 15, 31), dtype=torch.long)
    nib = (torch.arange(8) * 4).long()
    lo = v & 255
    sign = v >> 15
    eh = (v >> 8) & 127
    hist = torch.bincount(eh.flatten(), minlength=128)
    base = int(torch.stack([hist[b:b + 8].sum() for b in range(121)]).argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    code = ((d.view(16, 8, 32, 4, 8).long() << nib) + (sign.view(16, 8, 32, 4, 8).long() << sign_pos)).sum(-1)
    code = torch.where(code >= 2**31, code - 2**32, code).to(torch.int32).view(torch.uint8).view(16, 8, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(16, 8, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    # Unpack
    p = data.view(16, 8, 3, 32, 16)
    lo_u = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code_u = p[:, :, 2].contiguous().view(torch.int32).view(16, 8, 32, 4).long() & 0xFFFFFFFF
    d_u = ((code_u[..., None] >> nib) & 7).reshape(16, 8, 32, 32)
    sg_u = ((code_u[..., None] >> sign_pos) & 1).reshape(16, 8, 32, 32)
    val = ((sg_u << 15) | ((base + d_u) << 8)).to(torch.int32) | lo_u
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    c12_bitwise_exact = bool(torch.equal(val, v))

    # 2. Verify wgs_finisher_inreg invariants across all 192 saved synthetic draws in screen.json
    screen = json.loads((WORKTREE / "results/moe-unit-ownership-screen-20261029/screen.json").read_text())
    G = 132
    WGS_CB = [100, 104, 115, 123, 132, 144, 156, 174]
    total_phaseB_split_units = 0
    finisher_is_last_seg_count = 0
    finisher_unique_per_split_unit = True
    for row in screen["rows"]:
        jC = row["jobs"]
        nj = len(jC)
        pre = [0]
        for cnt in jC:
            ntt = min(8, max(1, (cnt + 7) >> 3))
            pre.append(pre[-1] + (WGS_CB[ntt - 1] << 8))
        tot = pre[-1]
        def stage_at(x):
            lo_i, hi_i = 0, nj
            while hi_i - lo_i > 1:
                mid = (lo_i + hi_i) >> 1
                if pre[mid] <= x:
                    lo_i = mid
                else:
                    hi_i = mid
            cs = WGS_CB[min(8, max(1, (jC[lo_i] + 7) >> 3)) - 1]
            return (lo_i << 8) + (x - pre[lo_i]) // cs
        unit_finishers = {}
        for c_idx in range(G):
            g0 = stage_at(tot * c_idx // G)
            g1 = (nj << 8) if c_idx == G - 1 else stage_at(tot * (c_idx + 1) // G)
            gs = g0
            while gs < g1:
                u = gs >> 4
                s0 = gs & 15
                s1 = min(16, s0 + (g1 - gs))
                if s1 - s0 < 16 and s0 == 0:
                    total_phaseB_split_units += 1
                    if gs + (s1 - s0) == g1:
                        finisher_is_last_seg_count += 1
                    unit_finishers[u] = unit_finishers.get(u, 0) + 1
                gs += s1 - s0
        if any(v != 1 for v in unit_finishers.values()):
            finisher_unique_per_split_unit = False

    return {
        "c12_bitwise_exact_including_neg_zero": c12_bitwise_exact,
        "c12_escapes_tested": int(er.numel()),
        "wgs_phaseB_draws_checked": len(screen["rows"]),
        "wgs_phaseB_split_units_checked": total_phaseB_split_units,
        "wgs_finisher_is_last_segment_of_cta": finisher_is_last_seg_count == total_phaseB_split_units,
        "wgs_finisher_unique_per_split_unit": finisher_unique_per_split_unit,
    }


if __name__ == "__main__":
    nvrtc = NVRTC()
    res = {
        "m128w": test_m128w(nvrtc),
        "moe_wgs_finisher_inreg": test_wgs_finisher_inreg(nvrtc),
        "rf_vec4_smem": test_rf_vec4_smem(nvrtc),
        "cpu_math_checks": cpu_math_checks(),
    }
    out_path = WORKTREE / "results/opus-fresh-investigation-20261029/nvrtc_and_cpu_verification.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))

