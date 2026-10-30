
# Dense S2b (branch omos/dense-top50-20261030, DENSE_LEDGER.md Section 3b): M=128 unpooler (8192 x 4096) only.
# Same exact-c12 m32w_gemm WGMMA as S2 (BN=128 tokens per decoded weight tile, split=1, no cluster), but
# pipelined per warpgroup: MMA helpers commit without waiting, two A-register banks alternate by step,
# and after issuing group i the warpgroup waits only for group i-1 (wait_group 1) before refilling that
# group's x stage. The decode of step i+1 overlaps group i. Lifetime schedule: DENSE_LEDGER.md 3b.
# Gates as S2: bit-exact pack, one-hot over all columns, FP64 random/spike rows, ABBA paired timing
# (median >= 1.03 and >= 75% wins).
_S2B_INSTS = [(128, 64, 3, 2, 4, True, 1), (128, 64, 2, 2, 6, True, 1)]
_S2B_M = 128


def _s2b_mma(n):
    nr = n // 2
    regs = ','.join(f'%{i}' for i in range(nr))
    outs = ','.join(f'"+f"(d[{i}])' for i in range(nr))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join(f'%{nr + 4 * kk + i}' for i in range(4))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'


def _s2b_core(core):
    fence = '      #pragma unroll\n      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");\n'
    edits = [
        ('  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {\n',
         '  static_assert(PF>=2 && Q>=3, "S2b lifetime schedule needs two A banks and three x stages");\n'
         '  uint32_t abank[PF][16];\n  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {\n'),
        ('      uint32_t a[16];\n', '      uint32_t* a=abank[p];\n'),
        ('      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n',
         '      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n'),
        ('      loadx(i+Q);\n',
         '      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");\n' + fence + '      if(i>=1)loadx(i-1+Q);\n'),
        ('  __syncthreads();\n  if(tid%128==0) {\n    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval',
         '  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n' + fence.replace('      ', '  ')
         + '  __syncthreads();\n  if(tid%128==0) {\n    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval'),
    ]
    for (old, new) in edits:
        assert core.count(old) == 1, old
        core = core.replace(old, new)
    return core


def _s2b_device_source(prefix, core):
    return prefix + '\n'.join(_s2b_mma(n) for n in (16, 32, 64, 128)) + _s2b_core(core)


# --- runtime ---
import time as _s2b_time


def _s2b_build(insts):
    (cases, attrs) = ([], [])
    for (i, (bn, r, wk, pf, q, c, cap)) in enumerate(insts):
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    src = (_s2b_device_source(dd_decodediet_tt_mw_CUDA_PREFIX, dd_decodediet_tt_mw_CUDA_CORE) + dd_decodediet_tt_mw_CUDA_MAP
           + dd_decodediet_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs)))
    name = 's2b_m32p_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, dd_decodediet_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True,
                       with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])


def _s2b_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def _s2b_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s2b_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_s2b_once(ga, flush)); tb.append(_s2b_once(gb, flush))
        else:
            tb.append(_s2b_once(gb, flush)); ta.append(_s2b_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sum(r > 1.0 for r in ratios) / len(ratios), ta, tb


def _s2b_verify_random(ext, idx, pk, w, ms, seed=13):
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


def _s2b_select(state, weights):
    t0 = _s2b_time.monotonic()
    w = weights['unpooler']
    (n, k) = w.shape
    dev = w.device
    ext = _s2b_build(_S2B_INSTS)
    print(f'S2B_BUILD {_s2b_time.monotonic() - t0:.0f}s info ' + ' '.join(f'{inst}:{ext.info(i)}' for (i, inst) in enumerate(_S2B_INSTS)), flush=True)
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261031)
    x = torch.randn((_S2B_M, k), generator=gen, device=dev).to(torch.bfloat16)
    data = {'x': x, 'config': {'m': _S2B_M, 'n': n, 'k': k, 'layer': 'unpooler'}}
    gbase = _s2b_graph(lambda: _s2b_prev_kernel(data, state))
    ref = _s2b_prev_kernel(data, state).double()
    (packs, best) = ({}, None)
    for (i, inst) in enumerate(_S2B_INSTS):
        try:
            if ext.info(i) % 10000:
                print(f'S2B_SKIP {inst} local memory {ext.info(i)}', flush=True)
                continue
            wk = inst[2]
            if wk not in packs:
                packs[wk] = dd_decodediet_tt_mw_pack_weight(w, 64, wk, 1, True)
            pk = packs[wk]
            if pk is None:
                print(f'S2B_SKIP {inst} pack', flush=True)
                continue
            if not dd_decodediet_tt_mw_verify_onehot(ext, i, pk, w, _S2B_M) or not _s2b_verify_random(ext, i, pk, w, (_S2B_M, 100)):
                print(f'S2B_SKIP {inst} verify', flush=True)
                continue
            print(f'S2B_PACK {inst} bitexact {pk["pack_exact"]} escapes {pk["escapes"]} max_esc {pk["max_esc"]} onehot 1 random 1', flush=True)
            out = torch.empty((_S2B_M, n), dtype=torch.bfloat16, device=dev)
            run = lambda: dd_decodediet_tt_mw_launch_raw(ext, i, pk, x, out)
            if run():
                print(f'S2B_SKIP {inst} launch', flush=True)
                continue
            agree = float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30))
            gc = _s2b_graph(run)
            (med, frac, ta, tb) = _s2b_paired(gbase, gc, flush)
            del gc
        except Exception as ex:
            print(f'S2B_SKIP {inst} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
            continue
        print(f'S2B m{_S2B_M} {inst} ratio {med:.4f} win {frac:.2f} vs_incumbent {agree:.1e} base_us ' + ','.join(f'{v:.1f}' for v in ta)
              + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
        if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
            best = (med, inst, i, pk)
    del gbase, flush
    if best is not None:
        state['s2b_sel'][(_S2B_M, 'unpooler', n, k)] = (ext, best[2], best[3])
        print(f'S2B_SELECT m{_S2B_M} {best[1]} {best[0]:.4f}', flush=True)
    else:
        print(f'S2B_KEEP m{_S2B_M}', flush=True)
    torch.cuda.synchronize()
    print(f'S2B_SELECTOR selected={sorted(m for (m, _, _, _) in state["s2b_sel"])} s={_s2b_time.monotonic() - t0:.1f}', flush=True)


_S2B_SEEN = set()
_s2b_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('s2b_sel')
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
                    if key not in _S2B_SEEN:
                        _S2B_SEEN.add(key)
                        print(f'S2B_USE {key}', flush=True)
                    return out
    return _s2b_prev_kernel(data, state)


_s2b_prev_prepare = prepare


def prepare(weights):
    state = _s2b_prev_prepare(weights)
    state['s2b_sel'] = {}
    try:
        _s2b_select(state, weights)
    except Exception as ex:
        state['s2b_sel'] = {}
        print(f'S2B_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
