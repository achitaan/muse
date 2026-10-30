
# Dense S2c (DENSE_LEDGER.md): DIAGNOSTIC ONLY, no routing change. Times the S2b kernel (BN=128, WK=3, Q=4,
# M=128 unpooler) with streams switched off at compile time (S2C_ABL bitmask): 1 = no weight global loads
# (opaque ALU values keep the decode), 2 = no x TMA loads or barrier waits, 4 = no WGMMA (decoded A folded
# into acc so the decode is kept). Masks 0..6 are timed round-robin with the incumbent (cold L2, CUDA
# graphs, 16 rounds, raw samples printed) to measure which stream binds. Ablated outputs are not used.
_S2C_INST = (128, 64, 3, 2, 4, True, 1)
_S2C_MASKS = (0, 1, 2, 4, 3, 5, 6)


def _s2c_core(core):
    core = _s2b_core(core)
    edits = [
        ('for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);', 'for(int l=0;l<NL;l++)raw[slot][l]=S2C_W(src+l*32,pw,i,l);'),
        ('    if(tid%128==0 && i<nkb) {\n', '    if(!(S2C_ABL&2) && tid%128==0 && i<nkb) {\n'),
        ('      waitbar(full+s*8,(i/Q)&1);\n', '      if constexpr(!(S2C_ABL&2))waitbar(full+s*8,(i/Q)&1);\n'),
        ('      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n',
         '      if constexpr(BN==128 && !(S2C_ABL&4))mma128(acc,a,desc(base+s*SB));\n'
         '      if constexpr((S2C_ABL&4)!=0){uint32_t z=0;\n        #pragma unroll\n        for(int q=0;q<16;q++)z^=a[q];\n'
         '        acc[0]+=__uint_as_float(z&0x3fffffffu);}\n'),
    ]
    for (old, new) in edits:
        assert core.count(old) == 1, old
        core = core.replace(old, new)
    return core


def _s2c_device_source(mask, prefix, core):
    head = (f'#define S2C_ABL {mask}\n'
            '#if S2C_ABL & 1\n#define S2C_W(p,pol,i,l) make_uint4(uint32_t(lane*7+(i)),uint32_t((i)*3+(l)),uint32_t(lane+(l)),uint32_t(i))\n'
            '#else\n#define S2C_W(p,pol,i,l) ldw(p,pol)\n#endif\n')
    return head + prefix + '\n'.join(_s2b_mma(n) for n in (16, 32, 64, 128)) + _s2c_core(core)


# --- runtime ---
import time as _s2c_time


def _s2c_build(mask):
    (bn, r, wk, pf, q, c, cap) = _S2C_INST
    params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
    src = (_s2c_device_source(mask, dd_decodediet_tt_mw_CUDA_PREFIX, dd_decodediet_tt_mw_CUDA_CORE) + dd_decodediet_tt_mw_CUDA_MAP
           + dd_decodediet_tt_mw_CUDA_HOST.replace('DISPATCH', f'case 0: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
           .replace('INFO', f'case 0:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;'))
    name = f's2c_abl{mask}_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return mask, load_inline(name, dd_decodediet_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True,
                             with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])


def _s2c_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s2c_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode='thread_local'):
        fn()
    return g


class _S2CEager:
    def __init__(self, fn):
        self.fn = fn

    def replay(self):
        self.fn()


def _s2c_prepare_timer(label, fn):
    try:
        rc = fn()
        torch.cuda.synchronize()
        print(f'S2C_EAGER {label} rc {rc}', flush=True)
        if rc:
            return None
    except Exception as ex:
        print(f'S2C_EAGER_FAIL {label} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
        return None
    try:
        return _s2c_graph(fn)
    except Exception as ex:
        print(f'S2C_CAPTURE_FAIL {label} {type(ex).__name__}: {str(ex)[:200]} -> eager timing', flush=True)
        torch.cuda.synchronize()
        return _S2CEager(fn)


def _s2c_diag(state, weights):
    t0 = _s2c_time.monotonic()
    w = weights['unpooler']
    (n, k) = w.shape
    dev = w.device
    exts = dict(_s2c_build(m) for m in _S2C_MASKS)
    print(f'S2C_BUILD {_s2c_time.monotonic() - t0:.0f}s info ' + ' '.join(f'abl{m}:{exts[m].info(0)}' for m in _S2C_MASKS), flush=True)
    pk = dd_decodediet_tt_mw_pack_weight(w, 64, _S2C_INST[2], 1, True)
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261032)
    x = torch.randn((128, k), generator=gen, device=dev).to(torch.bfloat16)
    data = {'x': x, 'config': {'m': 128, 'n': n, 'k': k, 'layer': 'unpooler'}}
    ref = _s2c_prev_kernel(data, state).double()
    torch.cuda.synchronize()
    graphs = {}
    g = _s2c_prepare_timer('incumbent', lambda: (_s2c_prev_kernel(data, state), 0)[1])
    if g is not None:
        graphs['base'] = g
    outs = {}
    for m in _S2C_MASKS:
        outs[m] = torch.empty((128, n), dtype=torch.bfloat16, device=dev)
        (ext, o) = (exts[m], outs[m])
        g = _s2c_prepare_timer(f'abl{m}', lambda ext=ext, o=o: dd_decodediet_tt_mw_launch_raw(ext, 0, pk, x, o))
        if g is not None:
            graphs[m] = g
    if 0 in graphs:
        print(f'S2C_AGREE abl0_vs_incumbent {float((outs[0].double() - ref).norm() / ref.norm()):.1e}', flush=True)
    samples = {key: [] for key in graphs}
    for rnd in range(16):
        keys = list(graphs) if rnd % 2 == 0 else list(graphs)[::-1]
        for key in keys:
            samples[key].append(_s2c_once(graphs[key], flush))
    med = {key: sorted(v)[len(v) // 2] for (key, v) in samples.items()}
    for key in graphs:
        label = 'incumbent' if key == 'base' else f'abl{key}'
        mode = 'eager' if isinstance(graphs[key], _S2CEager) else 'graph'
        print(f'S2C {label} {mode} med_us {med[key]:.2f} raw ' + ','.join(f'{v:.1f}' for v in samples[key]), flush=True)
    del graphs, flush
    torch.cuda.synchronize()
    print(f'S2C_DONE s={_s2c_time.monotonic() - t0:.1f}', flush=True)


_s2c_prev_kernel = kernel_fn
_s2c_prev_prepare = prepare


def prepare(weights):
    state = _s2c_prev_prepare(weights)
    try:
        _s2c_diag(state, weights)
    except Exception as ex:
        print(f'S2C_DIAG off {type(ex).__name__}: {str(ex)[:300]}', flush=True)
    torch.cuda.synchronize()
    return state
