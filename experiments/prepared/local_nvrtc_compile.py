"""Workspace-only NVRTC device-code preflight. Never load/launch a GPU kernel."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
TOOL = ROOT / '.tools/cuda-compiler-13.0'
PINNED = {
    'nvidia_cuda_nvrtc-13.0.88-py3-none-win_amd64.whl': '6bcd4e7f8e205cbe644f5a98f2f799bef9556fefc89dd786e79a16312ce49872',
    'nvidia_cuda_runtime-13.0.88-py3-none-win_amd64.whl': '0156cf8f3453025938aab80bcca909506f4e82f1584a17736df7edeba882cc02',
    'nvidia_cuda_cccl-13.0.85-py3-none-win_amd64.whl': 'e160c6f031687b913fbe6e82e43f1788c3dd2a2a6378d3a8b15a31934b3ab285',
}
# The device source uses only std::true_type/false_type. NVRTC has no host STL.
# This explicit virtual header is a proxy adaptation, not a submitted edit.
TAGS = '''namespace std {
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
'''
STDINT = '''typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long long int64_t;
typedef unsigned long long uint64_t;
typedef int64_t intptr_t;
typedef uint64_t uintptr_t;
'''


class NVRTC:
    def __init__(self):
        self.binary = TOOL / 'nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll'
        self.search = os.add_dll_directory(str(self.binary.parent))
        self.builtins = c.CDLL(str(self.binary.parent / 'nvrtc-builtins64_130.dll'))
        self.lib = c.CDLL(str(self.binary))
        self.bind('nvrtcVersion', [c.POINTER(c.c_int), c.POINTER(c.c_int)])
        self.bind('nvrtcCreateProgram', [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p,
                                        c.c_int, c.POINTER(c.c_char_p), c.POINTER(c.c_char_p)])
        self.bind('nvrtcAddNameExpression', [c.c_void_p, c.c_char_p])
        self.bind('nvrtcCompileProgram', [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)])
        self.bind('nvrtcDestroyProgram', [c.POINTER(c.c_void_p)])
        for kind in ('ProgramLog', 'PTX', 'CUBIN'):
            self.bind('nvrtcGet' + kind + 'Size', [c.c_void_p, c.POINTER(c.c_size_t)])
            self.bind('nvrtcGet' + kind, [c.c_void_p, c.c_void_p])
        self.lib.nvrtcGetErrorString.argtypes = [c.c_int]
        self.lib.nvrtcGetErrorString.restype = c.c_char_p
        a, b = c.c_int(), c.c_int()
        self.check(self.lib.nvrtcVersion(c.byref(a), c.byref(b)))
        self.version = [a.value, b.value]
        assert self.version == [13, 0]

    def bind(self, name, args):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = args, c.c_int

    def check(self, rc):
        if rc:
            raise RuntimeError(self.lib.nvrtcGetErrorString(rc).decode())

    def output(self, prog, kind):
        size = c.c_size_t()
        self.check(getattr(self.lib, 'nvrtcGet' + kind + 'Size')(prog, c.byref(size)))
        buf = c.create_string_buffer(size.value)
        self.check(getattr(self.lib, 'nvrtcGet' + kind)(prog, buf))
        return bytes(buf.raw)

    def compile(self, source, names, flags, tags=False):
        prog = c.c_void_p()
        headers = (c.c_char_p * 2)(TAGS.encode(), STDINT.encode()) if tags else None
        includes = (c.c_char_p * 2)(b'type_traits', b'stdint.h') if tags else None
        self.check(self.lib.nvrtcCreateProgram(c.byref(prog), source, b'device-preflight.cu',
                                              2 if tags else 0, headers, includes))
        try:
            for name in names:
                self.check(self.lib.nvrtcAddNameExpression(prog, name.encode()))
            opts = (c.c_char_p * len(flags))(*(f.encode() for f in flags))
            start = time.monotonic()
            rc = self.lib.nvrtcCompileProgram(prog, len(flags), opts)
            duration = time.monotonic() - start
            log = self.output(prog, 'ProgramLog').rstrip(b'\0').decode(errors='replace')
            cubin = self.output(prog, 'CUBIN') if rc == 0 else b''
            return rc, log, cubin, duration
        finally:
            self.check(self.lib.nvrtcDestroyProgram(c.byref(prog)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--name', action='append', default=[])
    parser.add_argument('--define', action='append', default=[])
    parser.add_argument('--prefix-before-torch', action='store_true')
    parser.add_argument('--minimal-std-tags', action='store_true')
    parser.add_argument('--forward-wgs-owner', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    for name, expected in PINNED.items():
        assert hashlib.sha256((TOOL / 'wheels' / name).read_bytes()).hexdigest() == expected
    raw = Path(args.source).read_bytes() if args.source else b'extern "C" __global__ void smoke(float* p) { p[threadIdx.x] = (float)threadIdx.x; }\n'
    source = raw
    if args.prefix_before_torch:
        assert source.count(b'#include <torch/extension.h>') == 1
        source = source.split(b'#include <torch/extension.h>', 1)[0]
    forward = b'__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'
    anchor = b'template <bool PB>\n__device__ __forceinline__ void wgs_phase'
    if args.forward_wgs_owner:
        # NVRTC's lookup differs from the judged NVCC frontend here. Add only
        # the declaration of the existing unique function; no body changes.
        assert source.count(anchor) == 1 and source.count(forward) == 0
        old = source
        source = source.replace(anchor, forward + anchor)
        assert source.replace(forward, b'') == old
    flags = ['--gpu-architecture=sm_90a', '--std=c++20', '--generate-line-info',
             '--device-as-default-execution-space',
             '--ptxas-options=-v',
             '--include-path=' + str(TOOL / 'nvidia/cu13/include'),
             '--include-path=' + str(TOOL / 'nvidia/cu13/include/cccl')]
    flags += ['-D' + d for d in args.define]
    nvrtc = NVRTC()
    rc, log, cubin, duration = nvrtc.compile(source, args.name, flags, args.minimal_std_tags)
    out = Path(args.out)
    assert out.resolve().is_relative_to(ROOT.resolve()), 'Keep compiler artifacts in the workspace'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.log').write_text(log, encoding='utf-8')
    out.with_suffix('.cu').write_bytes(source)
    if rc == 0:
        out.with_suffix('.cubin').write_bytes(cubin)
    report = dict(nvrtc_version=nvrtc.version, package_hashes=PINNED, platform='Windows x86_64',
                  original_source_sha256=hashlib.sha256(raw).hexdigest(),
                  compiled_prefix_sha256=hashlib.sha256(source).hexdigest(), source_bytes=len(source),
                  host_torch_launcher_omitted=args.prefix_before_torch,
                  added_forward_declaration=forward.decode() if args.forward_wgs_owner else None,
                  virtual_type_traits_header=TAGS if args.minimal_std_tags else None,
                  virtual_stdint_header=STDINT if args.minimal_std_tags else None,
                  name_expressions=args.name, flags=flags, return_code=rc,
                  cubin_bytes=len(cubin), compile_seconds=duration, log_file=str(out.with_suffix('.log')),
                  gpu_kernels_loaded_or_launched=False, official_checks_used=0,
                  exact_judge_toolchain=False, scope='Device-code proxy preflight only; no host-extension ABI, occupancy, correctness or performance validation')
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('nvrtc_version', 'source_bytes', 'return_code', 'cubin_bytes', 'compile_seconds', 'scope')}))
    if rc:
        print(log[-7000:])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
