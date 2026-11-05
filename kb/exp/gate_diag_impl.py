# Appended to the clean decode_v1 source by build_decode_gate_diag.py.
# Timing-only ablation: NEVER selected by kernel_fn. The gate is pinned to the
# incumbent's output on this fixed synthetic input, not zeroed (which would
# change routing/MoE traffic and invalidate a whole-layer ceiling).
import hashlib as _dg_hash
import statistics as _dg_stats
import time as _dg_time

_DG_PARENT_PREPARE = prepare
_DG_EXT = {}
_DG_RESULTS = []


def _dg_sk_source(narrow=False):
    if narrow:
        src = _de_source()
    else:
        lines = []
        for m in (1, 16, 32):
            cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == bfp__sk2_name('qkv_gate', m))
            _, ns = bfp__sk2_geom(cfg, bfp__sk2_tokr(m))
            _, _, _, _, nt, nwr, wsk, upw, nacc, cl = cfg
            lines.append(f'if (M == {m}) {{ launch_sk2<{2 if m == 32 else 1}, {bfp__sk2_tokr(m)}, '
                         f'{nt}, {nwr}, {wsk}, {upw}, {ns}, true, false, {nacc}, {cl}, __nv_bfloat16>'
                         '(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg); return; }')
        body = bfp_SK2_SRC.replace('@@SK2_DISPATCH@@', '\n'.join(lines)).replace('@@SK2F_DISPATCH@@', '')
        src = bfp_CUDA_SRC.replace('@@MBAR@@', bfp_MBAR_HELPERS).replace('@@BODY@@', '') + bfp_MAP_SRC + body
    # Keep N=2304 as the output stride, but use the compact stream's CTA count.
    # There is no reduction across CTAs: arithmetic/escape order is unchanged.
    for old, new in (('dim3(N / R * CL)', 'dim3(Ws.size(1) * CL)'),
                     ('kern<<<N / R,', 'kern<<<Ws.size(1),')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def _dg_build_sk(narrow=False):
    src = _dg_sk_source(narrow)
    tag = _dg_hash.sha256(src.encode()).hexdigest()[:12]
    cpp = bfp_CPP_SRC[bfp_CPP_SRC.index('void sk2('):bfp_CPP_SRC.index('void sk2f(')]
    cpp += 'void sk2_set_pdl(int64_t n);'
    _DG_EXT['narrow' if narrow else 'sk'] = load_inline(
        'gate_ceiling_sk_' + tag, '#include <torch/extension.h>\n' + cpp, cuda_sources=src,
        functions=['sk2', 'sk2_set_pdl'], no_implicit_headers=True,
        extra_cflags=['-O3', '-std=c++20'],
        extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)


_DG_SUPPORT = r'''
// One large block per SM. All blocks rendezvous before any is allowed to
// return, so each of the N-cap surviving blocks really reserves a DISTINCT SM.
// A five-second watchdog prevents a failed host-side diagnostic from wedging
// the CUDA context. Stop is written by DMA, not a kernel needing an idle SM.
__global__ void gate_reserve(int* state, int cap, int nsm) {
  extern __shared__ volatile unsigned char bytes[];
  if (threadIdx.x == 0) {
    bytes[0] = 0;
    unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    state[3 + blockIdx.x] = sm;
    atomicAdd(state, 1);
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    while (atomicAdd(state, 0) != nsm) {
      unsigned long long now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (now - t > 5000000000ull) { atomicExch(state + 2, 1); return; }
      __nanosleep(256);
    }
    if ((int)blockIdx.x < cap) return;
    int stop = 0;
    while (!stop) {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(stop) : "l"(state + 1) : "memory");
      unsigned long long now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (now - t > 5000000000ull) { atomicExch(state + 2, 1); break; }
      __nanosleep(256);
    }
  }
}
void gate_block(torch::Tensor state, int64_t cap, int64_t nsm) {
  cudaFuncSetAttribute(gate_reserve, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  gate_reserve<<<nsm,32,200*1024,at::cuda::getCurrentCUDAStream()>>>(state.data_ptr<int>(),cap,nsm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int L, int MODE> int64_t gate_occ_impl() {
  auto k = attn_kernel<L,false,MODE>;
  const int sm = sizeof(Smem<L,false>) + 1024;
  cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,k,Cfg<L,false>::NTX,sm));
  return occ;
}
int64_t gate_occ(int64_t L, int64_t mode) {
  if (L == 8) return gate_occ_impl<8,15>();
  if (mode == 13) return gate_occ_impl<1,13>();
  return gate_occ_impl<1,1>();
}
'''


def _dg_build_support():
    # Instrumented attention is used ONLY for a one-replay placement survey;
    # all timings use the untouched incumbent attention extension.
    src = '#include <c10/cuda/CUDAException.h>\n' + ATT7_CUDA_SRC
    needle = 'attn_kernel(const __grid_constant__ Params p) {'
    assert src.count(needle) == 1
    src = src.replace(needle, needle + '\n' +
        '  if (p.dbg && threadIdx.x == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); '
        'p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 15] = sm + 1; }\n', 1)
    src += _DG_SUPPORT
    cpp = ATT7_CPP_SRC + '\nvoid gate_block(torch::Tensor state,int64_t cap,int64_t nsm);\nint64_t gate_occ(int64_t L,int64_t mode);'
    tag = _dg_hash.sha256(src.encode()).hexdigest()[:12]
    _DG_EXT['support'] = load_inline('gate_ceiling_att_' + tag, cpp, cuda_sources=src,
        functions=['run', 'gate_block', 'gate_occ'],
        extra_cflags=['-O3', '-std=c++20'],
        extra_cuda_cflags=['-DOKC_QDIRECT=0', '-DOKC_PK2=0', '-DOKC_W0ENABLE=1', '-O3',
                           '-DOKC_SM90=1', '--expt-relaxed-constexpr', '-std=c++20',
                           '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)


def _dg_build_rq(st):
    ps = [p for (name, m), p in st.get('chen_rq', {}).items() if name == 'qkv']
    if not ps:
        return
    # p.i indexes the PROD subset, NOT the full RQV list. Preserve those case
    # numbers while pruning unused instantiations from the diagnostic build.
    import re
    prod = [_CHEN['RQV'][_CHEN['RQI'][nm]] for nm in _CHEN['RQPROD']]
    src = _CHEN['_rqsrc'](prod)
    wanted = {p['i'] for p in ps}
    src = re.sub(r'    case (\d+): return rq_go<[^\n]+;',
                 lambda m: m.group(0) if int(m.group(1)) in wanted else '', src)
    old = 'out.size(1) != W.size(0) * 128'
    assert src.count(old) == 1
    src = src.replace(old, 'out.size(1) < W.size(0) * 128')
    tag = _dg_hash.sha256(src.encode()).hexdigest()[:12]
    _DG_EXT['rq'] = load_inline('gate_ceiling_rq_' + tag,
        cpp_sources='int64_t rqrun(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,torch::Tensor,torch::Tensor,int64_t);',
        cuda_sources=src, functions=['rqrun'], extra_cflags=['-O3', '-std=c++20'],
        extra_cuda_cflags=_CHEN['RQFLAGS'] + ['-std=c++20'], verbose=False)


def _dg_short_state(st, T, qkv):
    ss = dict(st)
    ss['_dg_qkv'] = qkv.clone()
    if T <= 32:
        name = bfp__sk2_name('qkv_gate', T)
        ci, stream, tab, pmax, b4, nrows = st['bfp']['sk2'][name]
        # R=24 crosses the qkv/g boundary. Keep its last mixed tile (16 gate
        # rows): only 1008/1024 gate rows removed, a conservative ceiling.
        ncta = (Q_DIM + 2 * NUM_KV_HEADS * HEAD_DIM + 23) // 24
        ss['_dg_sk'] = (ci, stream[:, :ncta].contiguous(), tab[:ncta].contiguous(), pmax, b4, nrows)
        ss['_dg_route'] = 'sk2_narrow' if 2 <= T <= 8 else 'sk2'
        return ss
    p = st.get('chen_rq', {}).get(('qkv', T))
    if p is not None:
        nb = (Q_DIM + 2 * NUM_KV_HEADS * HEAD_DIM) // 128
        oldnb, ks = p['lay'].shape[0], p['KS']
        esc = p['esc']
        cap = (esc.numel() - 32) // (ks * oldnb * 8)
        small = dict(p, lay=p['lay'][:nb].contiguous(), mod=_DG_EXT['rq'])
        small['esc'] = torch.cat((esc[:-32].view(ks, oldnb, 8, cap)[:, :nb].contiguous().flatten(), esc[-32:]))
        if p.get('esc2') is not None:
            small['esc2'] = p['esc2'].view(ks, oldnb, 8, -1)[:, :nb].contiguous().flatten()
        small['cnt'] = torch.zeros_like(p['cnt'])
        ss['_dg_rq'] = small
        ss['_dg_route'] = 'rq'
        return ss
    plan = st['graft_lt'][('qkv', T)]
    lt = plan['lt']
    n = Q_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
    mode, batch = ('tn', 16) if T == 128 else ('tn', 1)
    a, b, _ = _graft_layouts(lt, T, n, HIDDEN, mode, batch, False)
    c = _GP()
    lt.call('cublasLtMatrixLayoutCreate', _C.byref(c), 14, n // batch, T, QKV_ROWS)
    for attr, value in ((5, _GI(batch)), (6, _C.c_int64(n // batch))):
        lt.call('cublasLtMatrixLayoutSetAttribute', c, attr, _C.byref(value), _C.sizeof(value))
    heur = _GraftHeur()
    lt.call('cublasLtMatmulAlgoCheck', lt.handle, lt.desc, a, b, c, c, _C.byref(plan['algo']), _C.byref(heur))
    if heur.workspaceSize > lt.workspace.numel():
        raise RuntimeError('short Lt workspace insufficient')
    ss['_dg_lt'] = dict(plan, layouts=(a, b, c))
    ss['_dg_route'] = 'Lt'
    return ss


def _dg_qkv(data, st, short=False):
    x, T = data['attn_input'], data['config']['tokens']
    if short:
        out = st['_dg_qkv']
        if '_dg_sk' in st:
            ci, stream, tab, pmax, b4, _ = st['_dg_sk']
            ext = _DG_EXT['narrow' if 2 <= T <= 8 else 'sk']
            _bfp_pdl(T, 'qkv')
            ext.sk2_set_pdl(bfp_SK2_PDL)
            ext.sk2(stream, x, out, tab, pmax, b4, T, QKV_ROWS, HIDDEN, ci, bfp__sk2_tokr(T), 0, st['bfp']['dummy'])
        elif '_dg_rq' in st:
            code = _CHEN['_rqcall'](st['chen_st'], st['_dg_rq'], x, out)
            if code:
                raise RuntimeError(f'short rq status {code}')
        else:
            plan = st['_dg_lt']
            lt = plan['lt']
            a, b, c = plan['layouts']
            lt.call('cublasLtMatmul', lt.handle, lt.desc, _C.byref(lt.alpha), _GP(plan['weight'].data_ptr()), a,
                    _GP(x.data_ptr()), b, _C.byref(lt.beta), _GP(out.data_ptr()), c, _GP(out.data_ptr()), c,
                    _C.byref(plan['algo']), _GP(lt.workspace.data_ptr()), lt.workspace.numel(),
                    _GP(torch.cuda.current_stream().cuda_stream))
        return out
    pick = st.get('mm_pick', {}).get((T, 'qkv'), True)
    plan = st.get('graft_lt', {}).get(('qkv', T)) if T in (128, 256) else None
    if plan is not None and pick is True:
        q = _rq_gemm(st, 'qkv', x, T)
        return q if q is not None else _graft_lt_run(x, plan)
    if pick == 'bfp' and T <= BFP_MAX_T and st.get('bfp') is not None:
        return _bfp_gemm(st, 'qkv', x, T, QKV_ROWS, HIDDEN)
    if pick is True or pick == 'bfp':
        return torch.mm(x, st['qkv'].t())
    p, s = dense(x, st['qkv'], T)
    return (p[0] if s == 1 else p.sum(0)).to(torch.bfloat16)


def _dg_layer(data, st, short=False):
    qkv = _dg_qkv(data, st, short)
    o_in = _decode_attention7(data, st, qkv)
    return _after_attention(data, st, o_in, None, data['config']['tokens'])


def _dg_capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    return graph, output


def _dg_cache(data):
    return tuple(data[k].view(torch.uint8).view(-1, 2, 64).index_select(0, data['cache_loc'])
                 for k in ('k_cache', 'v_cache'))


def _dg_reset(data, before):
    for key, value in zip(('k_cache', 'v_cache'), before):
        data[key].view(torch.uint8).view(-1, 2, 64).index_copy_(0, data['cache_loc'], value)


def _dg_pair(graphs, flush, sink, restore, blocks=12, deadline=None):
    ts, ds, rs = [[], []], [], [[], []]
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for block in range(blocks):
        bt = [[], []]
        for i in ((0, 1, 1, 0) if block % 2 == 0 else (1, 0, 0, 1)):
            if deadline is not None and _dg_time.monotonic() >= deadline:
                raise TimeoutError('gate prepare timing budget exhausted')
            restore()
            torch.sum(flush, dim=0, out=sink)
            a.record()
            graphs[i].replay()
            b.record()
            b.synchronize()
            bt[i].append(a.elapsed_time(b) * 1000)
        for i in (0, 1):
            ts[i].extend(bt[i])
        ds.append((sum(bt[0]) - sum(bt[1])) / 2)
        rs[block % 2].append(sum(bt[1]) / sum(bt[0]))
    med = _dg_stats.median
    return dict(base=med(ts[0]), short=med(ts[1]), save=med(ds), ratio=med(ts[1]) / med(ts[0]),
                paired=med(rs[0] + rs[1]), abba=med(rs[0]), baab=med(rs[1]))


def _dg_attention(data, st, qkv, flush, sink, before, label):
    cfg = data['config']
    B, L = cfg['batch'], cfg['q_len']
    ad = dict(data, qkv=qkv)
    plan = dict(_plan7(B, 1, L, cfg['window'], data['page_table'].shape[1], 1, qkv.device))
    plan.update(_TUNED7.get((B, L, True, 1), {}))
    grid = plan['n_splits'] * (1 if plan['flat'] else B)
    mode = 13 if B == 1 and L == 1 else 15 if L == 8 else 1
    ext = _DG_EXT['support']
    occ = ext.gate_occ(L, mode)
    nsm = st['nsm']
    # Survey actual SM placement without timing the instrumented binary.
    dbg = torch.zeros((grid, 64), dtype=torch.int64, device=qkv.device)
    _launch7(ad, dbg=dbg, _mod=ext, pdl=False)
    torch.cuda.synchronize()
    # DBGSTRIDE defaults to 16 unless segment instrumentation is enabled.
    stride = 64 if '#define OKC_SEGDBG 1' in ATT7_CUDA_SRC else 16
    smids = dbg.flatten()[15:grid * stride:stride].cpu().tolist()
    counts = {}
    for sm in smids:
        if sm:
            counts[sm - 1] = counts.get(sm - 1, 0) + 1
    # Small native grids spread over one SM per CTA. Preserve that footprint;
    # do not pretend that packing two CTAs per SM has no throughput cost.
    cap = min(nsm, grid) if grid < nsm else min(nsm, (grid + occ - 1) // occ)
    g, _ = _dg_capture(lambda: _launch7(ad, pdl=False))
    restore = lambda: _dg_reset(data, before)
    full = _dg_pair((g, g), flush, sink, restore, blocks=8)
    if cap == nsm:
        restricted = full
        actual = nsm
        valid = True
    else:
        state = torch.zeros(nsm + 3, dtype=torch.int32, device=qkv.device)
        stop = torch.ones(1, dtype=torch.int32, pin_memory=True)
        reserve = torch.cuda.Stream()
        reserve.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(reserve):
            ext.gate_block(state, cap, nsm)
        begin = _dg_time.monotonic()
        while int(state[0]) != nsm and _dg_time.monotonic() - begin < 1.0:
            pass
        try:
            if int(state[0]) != nsm:
                raise RuntimeError('SM reservation did not rendezvous')
            actual = len(set(state[3:].cpu().tolist()))
            if actual != nsm:
                raise RuntimeError(f'SM reservation has {actual}/{nsm} distinct SMs')
            restricted = _dg_pair((g, g), flush, sink, restore, blocks=8)
        finally:
            state[1:2].copy_(stop, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            reserve.synchronize()
        valid = not bool(int(state[2]))
    row = (f'GATE_ATT shape={label} grid={grid} occ_cta_per_sm={occ} surveyed_sms={len(counts)} '
           f'surveyed_max_cta_per_sm={max(counts.values(), default=0)} '
           f'full_sms={nsm} cap_sms={cap} full_us={full["base"]:.3f} cap_us={restricted["base"]:.3f} '
           f'valid={int(valid)} note=raw_graph_events_includes_launch_overhead')
    _DG_RESULTS.append(row)
    print(row, flush=True)


def _dg_shape(st, B, L, sliding, flush, sink):
    label = f'{"d" if L == 1 else "v"}{B}{"S" if sliding else "F"}'
    T = B * L
    data, refresh = _dg_inputs(st, B, L, sliding, 20261104 + T + 1000 * sliding)
    refresh()
    before = _dg_cache(data)
    qkv = _dg_qkv(data, st)
    torch.cuda.synchronize()
    ss = _dg_short_state(st, T, qkv)
    _dg_qkv(data, ss, True)
    torch.cuda.synchronize()
    nongate = Q_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
    exact = torch.equal(qkv[:, :nongate], ss['_dg_qkv'][:, :nongate])
    gates = torch.equal(qkv[:, nongate:], ss['_dg_qkv'][:, nongate:])
    iso = [_dg_capture(lambda: _dg_qkv(data, st))[0], _dg_capture(lambda: _dg_qkv(data, ss, True))[0]]
    base, b_out = _dg_capture(lambda: _dg_layer(data, st))
    short, s_out = _dg_capture(lambda: _dg_layer(data, ss, True))
    # Routing is pinned by using the actual incumbent gate, not a zero gate.
    base.replay()
    busiest = int(torch.bincount((b_out[2].long() // LOCAL_EXPERTS).flatten(), minlength=8).argmax())
    data['local_start'].fill_(busiest * LOCAL_EXPERTS)
    restore = lambda: _dg_reset(data, before)
    for turn in range(2):
        restore()
        base.replay()
        b_ids = b_out[2].clone()
        restore()
        short.replay()
        torch.cuda.synchronize()
        ids = torch.equal(b_ids, s_out[2])
        qi = _dg_pair(iso, flush, sink, lambda: None)
        layer = _dg_pair((base, short), flush, sink, restore)
        # A/A isolates measurement/layout noise from the gate-row ablation.
        ctl = _dg_pair((base, base), flush, sink, restore, blocks=4)
        row = (f'GATE_CEIL shape={label} pass={turn + 1} route={ss["_dg_route"]} '
               f'qkv_us={qi["base"]:.3f} qkv_short_us={qi["short"]:.3f} qkv_save_us={qi["save"]:.3f} '
               f'layer_us={layer["base"]:.3f} short_us={layer["short"]:.3f} save_us={layer["save"]:.3f} '
               f'ratio={layer["ratio"]:.5f} abba={layer["abba"]:.5f} baab={layer["baab"]:.5f} '
               f'control_us={ctl["save"]:.3f} qkv_exact={int(exact)} gate_exact={int(gates)} ids_exact={int(ids)}')
        _DG_RESULTS.append(row)
        print(row, flush=True)
    if 'support' in _DG_EXT and sliding and (B, L) in ((1, 1), (1, 8), (16, 1), (32, 1)):
        _dg_attention(data, st, qkv, flush, sink, before, label)
    torch.cuda.synchronize()


def prepare(weights):
    global MFAB_LAYER
    start, old = _dg_time.monotonic(), MFAB_LAYER
    errors = []
    def build(fn):
        try:
            fn()
        except Exception as ex:
            errors.append(f'{fn.__name__}: {str(ex)[-1400:]}')
    threads = [threading.Thread(target=build, args=(fn,)) for fn in
               (_dg_build_sk, lambda: _dg_build_sk(True), _dg_build_support)]
    for th in threads:
        th.start()
    try:
        MFAB_LAYER = False  # omit only an unrelated write-flush diagnostic
        st = _DG_PARENT_PREPARE(weights)
    finally:
        MFAB_LAYER = old
    for th in threads:
        th.join()
    try:
        for error in errors:
            # An OPTIONAL placement-survey build must never discard all twelve
            # independent projection/whole-layer timing measurements.
            print(f'GATE_BUILD_ERR {error}', flush=True)
        _dg_build_rq(st)
        print(f'GATE_INFO base_and_build_s={_dg_time.monotonic()-start:.1f} '
              'baseline=clean_decode_v1 timing_only=1 read_flush_bytes=536870912 skipped_gate_rows_small=1008/1024', flush=True)
        flush = torch.ones((512 << 20) // 4, dtype=torch.float32, device=st['qkv'].device)
        sink = torch.empty((), dtype=torch.float32, device=flush.device)
        for B, L in ((1, 1), (1, 8), (16, 1), (32, 1), (16, 8), (32, 8)):
            for sliding in (True, False):
                if _dg_time.monotonic() - start > 272:
                    print(f'GATE_ERR shape={B},{L},{sliding} prepare_budget_exhausted', flush=True)
                    continue
                try:
                    with torch.no_grad():
                        _dg_shape(st, B, L, sliding, flush, sink)
                except Exception as ex:
                    print(f'GATE_ERR shape={B},{L},{sliding} {type(ex).__name__}: {str(ex)[-900:]}', flush=True)
                    torch.cuda.synchronize()
        _check_guards()
    except Exception as ex:
        print(f'GATE_ERR diagnostic_aborted {type(ex).__name__}: {str(ex)[-1500:]}', flush=True)
    finally:
        # Tail-safe copy: the job API retains only the final 16000 characters.
        print('GATE_SUMMARY_BEGIN', flush=True)
        for row in _DG_RESULTS:
            print(row, flush=True)
        print(f'GATE_SUMMARY_END total_prepare_s={_dg_time.monotonic()-start:.1f} production=incumbent adopted=0', flush=True)
    return st
