
# Dense S2 (branch omos/dense-top50-20261030, DENSE_LEDGER.md S2, narrowed per review): M=128 unpooler
# (8192 x 4096) only. Compressed-weight reuse: the exact-c12 m32w_gemm WGMMA kernel with 64 or 128 tokens
# per CTA, so each decoded 64x64 weight tile feeds BN tokens. split=1: no cluster, no inter-block
# reduction; K is split only across the WK warpgroups of one CTA (their WGMMAs overlap each other's
# decode; each warpgroup keeps the kernel's own wait_group 0, so no register is reused while in flight).
# Packing is bit-exact (pack_weight compares unpacked int16 bits); one-hot exactness over all 4096
# columns and FP64 random/spike rows are checked before timing. Selection is balanced ABBA paired
# cold-L2 graph replay: median ratio >= 1.03 and >= 75% of pairs.
_S2_INSTS = [(128, 64, 2, 2, 3, True, 1), (128, 64, 4, 2, 2, True, 1)]
_S2_M = 128


def _s2_core(core):
    old = '      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n'
    assert core.count(old) == 1
    return core.replace(old, old + '      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n')


# --- runtime ---
import time as _s2_time


def _s2_build(insts):
    (cases, attrs) = ([], [])
    for (i, (bn, r, wk, pf, q, c, cap)) in enumerate(insts):
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    src = (dd_decodediet_tt_mw_CUDA_PREFIX + '\n'.join(dd_decodediet_tt_mw_mma(n) for n in (16, 32, 64, 128))
           + _s2_core(dd_decodediet_tt_mw_CUDA_CORE) + dd_decodediet_tt_mw_CUDA_MAP
           + dd_decodediet_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs)))
    name = 's2_m32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, dd_decodediet_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True,
                       with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])


def _s2_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def _s2_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s2_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_s2_once(ga, flush)); tb.append(_s2_once(gb, flush))
        else:
            tb.append(_s2_once(gb, flush)); ta.append(_s2_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sum(r > 1.0 for r in ratios) / len(ratios), ta, tb


def _s2_verify_random(ext, idx, pk, w, ms, seed=13):
    (n, k) = w.shape
    gen = torch.Generator(device=w.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
        if dd_decodediet_tt_mw_launch_raw(ext, idx, pk, x, out):
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 2.5e-3 or \
                float(rows.max()) > 1e-2 or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def _s2_select(state, weights):
    t0 = _s2_time.monotonic()
    w = weights['unpooler']
    (n, k) = w.shape
    dev = w.device
    ext = _s2_build(_S2_INSTS)
    print(f'S2_BUILD {_s2_time.monotonic() - t0:.0f}s info ' + ' '.join(f'{inst}:{ext.info(i)}' for (i, inst) in enumerate(_S2_INSTS)), flush=True)
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261031)
    x = torch.randn((_S2_M, k), generator=gen, device=dev).to(torch.bfloat16)
    data = {'x': x, 'config': {'m': _S2_M, 'n': n, 'k': k, 'layer': 'unpooler'}}
    gbase = _s2_graph(lambda: _s2_prev_kernel(data, state))
    ref = _s2_prev_kernel(data, state).double()
    (packs, best) = ({}, None)
    for (i, inst) in enumerate(_S2_INSTS):
        try:
            if ext.info(i) % 10000:
                print(f'S2_SKIP {inst} local memory {ext.info(i)}', flush=True)
                continue
            wk = inst[2]
            if wk not in packs:
                packs[wk] = dd_decodediet_tt_mw_pack_weight(w, 64, wk, 1, True)
            pk = packs[wk]
            if pk is None:
                print(f'S2_SKIP {inst} pack', flush=True)
                continue
            if not dd_decodediet_tt_mw_verify_onehot(ext, i, pk, w, _S2_M) or not _s2_verify_random(ext, i, pk, w, (_S2_M, 100)):
                print(f'S2_SKIP {inst} verify', flush=True)
                continue
            print(f'S2_PACK {inst} bitexact {pk["pack_exact"]} escapes {pk["escapes"]} max_esc {pk["max_esc"]} onehot 1 random 1', flush=True)
            out = torch.empty((_S2_M, n), dtype=torch.bfloat16, device=dev)
            run = lambda: dd_decodediet_tt_mw_launch_raw(ext, i, pk, x, out)
            if run():
                print(f'S2_SKIP {inst} launch', flush=True)
                continue
            agree = float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30))
            gc = _s2_graph(run)
            (med, frac, ta, tb) = _s2_paired(gbase, gc, flush)
            del gc
        except Exception as ex:
            print(f'S2_SKIP {inst} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
            continue
        print(f'S2 m{_S2_M} {inst} ratio {med:.4f} win {frac:.2f} vs_incumbent {agree:.1e} base_us ' + ','.join(f'{v:.1f}' for v in ta)
              + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
        if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
            best = (med, inst, i, pk)
    del gbase, flush
    if best is not None:
        state['s2_sel'][(_S2_M, 'unpooler', n, k)] = (ext, best[2], best[3])
        print(f'S2_SELECT m{_S2_M} {best[1]} {best[0]:.4f}', flush=True)
    else:
        print(f'S2_KEEP m{_S2_M}', flush=True)
    torch.cuda.synchronize()
    print(f'S2_SELECTOR selected={sorted(m for (m, _, _, _) in state["s2_sel"])} s={_s2_time.monotonic() - t0:.1f}', flush=True)


_S2_SEEN = set()
_s2_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('s2_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        if x.ndim == 2:
            key = (x.shape[0], cfg['layer'], cfg['n'], x.shape[1])
            v = sel.get(key)
            if v is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0 \
                    and x.device == v[2]['w'].device:
                (ext, idx, pk) = v
                out = torch.empty((x.shape[0], pk['n']), dtype=torch.bfloat16, device=x.device)
                if dd_decodediet_tt_mw_launch_raw(ext, idx, pk, x, out) == 0:
                    if key not in _S2_SEEN:
                        _S2_SEEN.add(key)
                        print(f'S2_USE {key}', flush=True)
                    return out
    return _s2_prev_kernel(data, state)


_s2_prev_prepare = prepare


def prepare(weights):
    state = _s2_prev_prepare(weights)
    state['s2_sel'] = {}
    try:
        _s2_select(state, weights)
    except Exception as ex:
        state['s2_sel'] = {}
        print(f'S2_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
