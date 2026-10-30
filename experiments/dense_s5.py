
# Dense S5 (DENSE_LEDGER.md, oracle plan 3d #1 + #3): separate per-shape gates in one candidate.
# #1 router M128: the incumbent rwdl_128<64,8,8,1,2,1> launches only 64 CTAs. The same hi+lo WGMMA source
#    (symbols renamed s5rw128) is built with BN=32/S=8 (half output tile, 128 CTAs) and BN=64/S=16 (128 CTAs).
# #3 o M16/M32: cluster-free c12 v18s (S=1) with output-row parallelism, (2,4,4,1) and (1,4,4,1).
# Each shape is rerouted only after exactness checks and ABBA paired cold-L2 graph timing (median >= 1.03,
# >= 75% of 16 pairs).
_S5_RW_VARIANTS = [(32, 8), (64, 16)]
_S5_O_GEOMS = [(2, 4, 4, 1), (1, 4, 4, 1)]
_S5_O_MS = (16, 32)


def _s5_mma32():
    regs = ', '.join(f'%{i}' for i in range(16))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(16))
    return ('__device__ __forceinline__ void s5rw128_mma_32(float* d,uint64_t a,uint64_t b) { asm volatile('
            f'"wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {{{regs}}}, %16, %17, 1, 1, 1, 0, 0;" : {outs} : "l"(a),"l"(b)); }}\n'
            'template<int BN> __device__ __forceinline__ void s5rw128_mma(float* d,uint64_t a,uint64_t b) {\n'
            '  if constexpr(BN==64) s5rw128_mma_64(d,a,b); else s5rw128_mma_32(d,a,b); }\n')


def _s5_rw_source(src):
    src = src.replace('rwdl_128', 's5rw128')
    anchor = 'template<int BN,int S,int ST,int SW,int RED,int PIPE>\n__global__'
    assert src.count(anchor) == 1
    src = src.replace(anchor, _s5_mma32() + '\n' + anchor)
    assert src.count('s5rw128_mma_64(acc,') == 4
    src = src.replace('s5rw128_mma_64(acc,', 's5rw128_mma<BN>(acc,')
    old = 'case 0:return s5rw128_launch_t<64,8,8,1,2,1,64,64>(w,x,out,m,stream,query);'
    assert src.count(old) == 1
    cases = ''.join(f'case {i}:return s5rw128_launch_t<{bn},{s},8,1,2,1,64,64>(w,x,out,m,stream,query);'
                    for (i, (bn, s)) in enumerate(_S5_RW_VARIANTS))
    return src.replace(old, cases)


# --- runtime ---
import time as _s5_time


def _s5_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode='thread_local'):
        fn()
    return g


def _s5_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s5_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_s5_once(ga, flush)); tb.append(_s5_once(gb, flush))
        else:
            tb.append(_s5_once(gb, flush)); ta.append(_s5_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sum(r > 1.0 for r in ratios) / len(ratios), ta, tb


def _s5_log(tag, med, frac, ta, tb, extra=''):
    print(f'S5 {tag} ratio {med:.4f} win {frac:.2f} {extra} base_us ' + ','.join(f'{v:.1f}' for v in ta)
          + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)


def _s5_router(state, weights, flush, gen):
    w = weights['router']
    dev = w.device
    src = _s5_rw_source(rw_SOURCES[128])
    cpp = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint s5rw128_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);'
    ext = load_inline('s5rw128_' + hashlib.sha256(src.encode()).hexdigest()[:14], cpp, cuda_sources=src, functions=['s5rw128_launch'],
                      extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True,
                      with_pytorch_error_handling=False, verbose=False)
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    packed = torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)
    exact = hi.float() + lo.float()
    stream = lambda: torch.cuda.current_stream(dev).cuda_stream
    launch = lambda i, x, out: ext.s5rw128_launch(i, packed.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream(), 0)
    x = torch.randn((128, 8192), generator=gen, device=dev).to(torch.bfloat16)
    data = {'x': x, 'config': {'m': 128, 'n': 256, 'k': 8192, 'layer': 'router'}}
    gbase = _s5_graph(lambda: _s5_prev_kernel(data, state))
    eye = torch.eye(8192, dtype=torch.bfloat16, device=dev)
    xr = torch.randn((128, 8192), generator=gen, device=dev)
    xr[127].zero_()
    xr[127, 4321] = 16.0
    xr = xr.to(torch.bfloat16)
    ref = xr.double() @ w.double().T
    best = None
    for (i, (bn, s)) in enumerate(_S5_RW_VARIANTS):
        tag = f'router m128 BN{bn}/S{s}'
        try:
            occ = ext.s5rw128_launch(i, packed.data_ptr(), x.data_ptr(), 0, 128, stream(), 1)
            got = torch.empty((8192, 256), dtype=torch.float32, device=dev)
            ok = True
            for st in range(0, 8192, 128):
                if launch(i, eye[st:st + 128], got[st:st + 128]):
                    ok = False
                    break
            ok = ok and bool(torch.equal(got, exact.t()))
            out = torch.empty((128, 256), dtype=torch.float32, device=dev)
            ok = ok and launch(i, xr, out) == 0
            torch.cuda.synchronize()
            err = float((out.double() - ref).norm() / ref.norm())
            spike = float((out[127].double() - ref[127]).norm() / ref[127].norm())
            if not ok or not bool(torch.isfinite(out).all()) or err > 2.5e-5 or spike > 2.5e-4:
                print(f'S5_SKIP {tag} verify ok={ok} err={err:.1e} spike={spike:.1e} occ={occ}', flush=True)
                continue
            outb = torch.empty((128, 256), dtype=torch.float32, device=dev)
            gc = _s5_graph(lambda i=i, outb=outb: launch(i, x, outb))
            (med, frac, ta, tb) = _s5_paired(gbase, gc, flush)
            del gc
        except Exception as ex:
            print(f'S5_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
            continue
        _s5_log(tag, med, frac, ta, tb, f'onehot_exact 1 relL2 {err:.1e} max_active_clusters {occ}')
        if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
            best = (med, i, tag)
    if best is not None:
        state['s5_sel'][(128, 'router')] = ('rw', (ext, packed, best[1]))
        print(f'S5_SELECT {best[2]} {best[0]:.4f}', flush=True)
    else:
        print('S5_KEEP router m128', flush=True)


def _s5_o(state, weights, flush, gen):
    w = weights['o']
    dev = w.device
    insts = []
    for m in _S5_O_MS:
        for g in _S5_O_GEOMS:
            (wr, wk, ku, s) = g
            insts.append((1, (m + 7) // 8, 1, 2, 0, 0, wr * wk * 32, 0, 8192, 1024, wr, wk, ku, s, 6))
    ext = v18s_build(insts, 's5_v18s_o')
    ids = {inst: i for (i, inst) in enumerate(insts)}
    (packs, streams) = ({}, {})
    for m in _S5_O_MS:
        x = torch.randn((m, 1024), generator=gen, device=dev).to(torch.bfloat16)
        data = {'x': x, 'config': {'m': m, 'n': 8192, 'k': 1024, 'layer': 'o'}}
        gbase = _s5_graph(lambda: _s5_prev_kernel(data, state))
        best = None
        for g in _S5_O_GEOMS:
            (wr, wk, ku, s) = g
            inst = (1, (m + 7) // 8, 1, 2, 0, 0, wr * wk * 32, 0, 8192, 1024, wr, wk, ku, s, 6)
            tag = f'o m{m} {g}'
            try:
                plan = v18s_plan(ext, ids, packs, streams, w, 'o', ('o', 'm', m), inst, g)
                if plan is None:
                    print(f'S5_SKIP {tag} plan/verify', flush=True)
                    continue
                out = torch.empty((m, 8192), dtype=torch.bfloat16, device=dev)
                run = lambda plan=plan, out=out, g=g: v18s_launch_raw(ext, plan['inst_id'], plan, x, out, 8192, g)
                if run():
                    print(f'S5_SKIP {tag} launch', flush=True)
                    continue
                gc = _s5_graph(run)
                (med, frac, ta, tb) = _s5_paired(gbase, gc, flush)
                del gc
            except Exception as ex:
                print(f'S5_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
                continue
            _s5_log(tag, med, frac, ta, tb, 'v18s_plan_verified 1')
            if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
                best = (med, plan, g, tag)
        del gbase
        if best is not None:
            state['s5_sel'][(m, 'o')] = ('v18s', (ext, best[1], best[2]))
            print(f'S5_SELECT {best[3]} {best[0]:.4f}', flush=True)
        else:
            print(f'S5_KEEP o m{m}', flush=True)


def _s5_select(state, weights):
    t0 = _s5_time.monotonic()
    dev = weights['o'].device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261035)
    for part in (_s5_router, _s5_o):
        try:
            part(state, weights, flush, gen)
        except Exception as ex:
            print(f'S5_PART_OFF {part.__name__} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
        torch.cuda.synchronize()
    del flush
    print(f'S5_SELECTOR selected={sorted(state["s5_sel"])} s={_s5_time.monotonic() - t0:.1f}', flush=True)


_S5_SEEN = set()
_s5_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('s5_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        if x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0:
            key = (x.shape[0], cfg['layer'])
            v = sel.get(key)
            out = None
            if v is not None and v[0] == 'rw' and cfg['n'] == 256 and x.shape[1] == 8192:
                (ext, packed, i) = v[1]
                if x.device == packed.device:
                    out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
                    if ext.s5rw128_launch(i, packed.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                                          torch.cuda.current_stream(x.device).cuda_stream, 0):
                        out = None
            elif v is not None and v[0] == 'v18s' and cfg['n'] == 8192 and x.shape[1] == 1024:
                (ext, plan, g) = v[1]
                if x.device == plan['wp'].device:
                    out = torch.empty((x.shape[0], 8192), dtype=torch.bfloat16, device=x.device)
                    if v18s_launch_raw(ext, plan['inst_id'], plan, x, out, 8192, g):
                        out = None
            if out is not None:
                if key not in _S5_SEEN:
                    _S5_SEEN.add(key)
                    print(f'S5_USE {key}', flush=True)
                return out
    return _s5_prev_kernel(data, state)


_s5_prev_prepare = prepare


def prepare(weights):
    state = _s5_prev_prepare(weights)
    state['s5_sel'] = {}
    try:
        _s5_select(state, weights)
    except Exception as ex:
        state['s5_sel'] = {}
        print(f'S5_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
