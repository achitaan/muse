"""CPU-only combined shader, buffer-ownership and host-flag isolation checks."""

import ast
from collections import Counter
import hashlib
import json

from pdl16_sources import HERE, M16_INST, OUT as ODIR
from pdlu_sources import OUT as UDIR, U16_INST, U32_INST
from build_pdl_chain import OUT
from check_pdl16_cpu import proxy as oproxy, instructions
from check_pdlu_cpu import proxy as uproxy
from check_pdl_diag_cpu import Capture
from check_r2_cpu import PRE
import nvrtc_screen_alternatives as nv


def buffers():
    raw = [0] * 24576
    for slot in range(2):
        for warp in range(8):
            for plane in range(3):
                for lane in range(32):
                    start = (((slot * 8 + warp) * 3 + plane) * 32 + lane) * 16
                    for q in range(start, start + 16):
                        raw[q] += 1
    assert set(raw) == {1}
    boundaries = [64 * p // 4 for p in range(5)]
    assert boundaries == [0, 16, 32, 48, 64]
    coverage = [u for a, b in zip(boundaries, boundaries[1:]) for u in range(a, b)]
    assert coverage == list(range(64))
    writers = [q for rank in range(2) for tid in range(256)
               for q in range(tid + rank * 256, 32 * 16, 256 * 2)]
    assert sorted(writers) == list(range(512))
    assert 2 * ((2 * (32 * 128 + 8) + 1023) // 1024 * 1024) == 18432
    assert 2 * 32 * (64 + 4) * 4 == 17408 <= 18432
    return dict(U16_rawring_bytes=24576, U16_reduction_bytes=8192,
                U16_dynamic_bytes=32768, rawring_one_writer_per_byte=True,
                U32_K_units=64, U32_rank_wg_boundaries=boundaries,
                U32_dynamic_bytes=18432, U32_epilogue_bytes=17408,
                U32_output_items=512, U32_one_writer_per_output_item=True,
                scope='Integer ownership/size models; no async memory or numerical GPU execution.')


def host_flags():
    # A narrow semantic check of host setter lifetime, including exceptions.
    # It is not CUDA graph validation: CUDA capture correctness remains open.
    tree = ast.parse((HERE / 'pdl_chain_runtime.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_p16_pool_route')
    class Ext:
        def __init__(self): self.depth = 0; self.calls = []; self.fail = False
        def sk2_set_pdl(self, n): self.depth = n
        def sk2f(self, *args):
            self.calls.append((self.depth, args[-2], args[-1]))
            if self.fail: raise RuntimeError('injected launch failure')
    class Torch:
        bfloat16 = 'bf16'; float32 = 'fp32'
        @staticmethod
        def empty(shape, **kwargs): return shape
    class X:
        def __init__(self, m): self.shape = (m, 8192); self.device = 'mock'
    ext = Ext()
    env = dict(torch=Torch, _fpr_get_ext=lambda: ext,
               _fpr_bfp__sk2_tokr=lambda m: m,
               _fpr_bfp_fused_pooler_router=lambda x, st: 'parent')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'host-config-only', 'exec'), env)
    state = {'fpr': {'sk2': {'pooler': (0, None, None, 0, 0, 4096), 'router': (0, None)},
                     'fused_part': None, 'fused_cnt': None}}
    for m in [16, 32]:
        env['_p16_pool_route'](X(m), state)
        assert ext.depth == 0
    assert ext.calls == [(5, 16, 16), (3, 32, 32)]
    ext.fail = True
    try:
        env['_p16_pool_route'](X(16), state)
    except RuntimeError:
        pass
    else:
        raise AssertionError('failure was swallowed')
    assert ext.depth == 0
    assert env['_p16_pool_route'](X(8), state) == 'parent'
    return dict(depths=[5, 3], reset_after_normal_and_error=True,
                small_M_parent=True, GPU_graph_correctness_proven=False)


def main():
    pieces = []
    for name, folder, make in [('o16', ODIR, oproxy), ('o32', ODIR, oproxy),
                               ('u16', UDIR, uproxy), ('u32', UDIR, uproxy)]:
        piece = make(name, (folder / (name + '-candidate.cu')).read_text())
        if pieces:
            piece = piece.removeprefix(PRE)
        if name == 'u32':
            piece = piece.replace('struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };',
                                  '// CUtensorMap already declared by O32 proxy')
        pieces.append(piece)
    code = '\n'.join(pieces)
    names = ['&pdlo16_gemv<' + ','.join(map(str, M16_INST)) + '>',
             '&pdlo32_projection<32,128,8,1,1,1,1>',
             '&pdlu16_gemv<' + ','.join(map(str, U16_INST)) + '>',
             '&pdlu32_gemm<' + ','.join(str(v).lower() for v in U32_INST) + '>']
    nvr = Capture()
    result = nvr.compile(code.encode(), names, [], tags=True, lineinfo=True)
    (OUT / 'combined-device-proxy.cu').write_bytes(code.encode())
    (OUT / 'combined-compile.log').write_text(result['log'])
    assert not result['rc'], result['log'][-2500:]
    result['kernels'] = nv.parse_ptxas(result['log'])
    assert sorted(k['registers'] for k in result['kernels']) == [42, 72, 88, 109]
    assert all(not any(k[f] for f in ['stack_frame', 'spill_stores', 'spill_loads']) for k in result['kernels'])
    result['device_proxy_sha256'] = hashlib.sha256(code.encode()).hexdigest()
    result['combined_CUDA_sha256'] = hashlib.sha256((OUT / 'combined.cu').read_bytes()).hexdigest()
    (OUT / 'combined-compile.json').write_text(json.dumps(result, indent=2) + '\n')
    math_counts = {}
    for name in ['u16', 'u32']:
        counts = []
        for arm in ['parent', 'candidate']:
            ins = instructions((UDIR / (name + '-' + arm + '.sass')).read_text())
            counts.append(Counter(t.split()[0] for a, t in ins
                                  if t.startswith(('HMMA', 'HGMMA', 'FADD', 'FFMA', 'FMUL', 'F2FP'))))
        assert counts[0] == counts[1]
        math_counts[name] = dict(counts[0])
    record = dict(GPU_jobs=0, buffers=buffers(), host_flags=host_flags(), math_instruction_counts=math_counts,
                  shader_resources=result['kernels'], limitations='CPU/device-proxy proof only; host CUDA build and H200 execution remain unverified.')
    (OUT / 'cpu-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
