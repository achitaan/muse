# DEC_POOL: stage-major FP32 router stream. Only copy addresses/count change;
# the original GLE consumer, escapes, FMA sequence and reductions stay verbatim.
import statistics as _dp_stats
import time as _dp_time

_DP_PREPARE = prepare
_DP_KERNEL = kernel_fn
_DP_FRT = _gle_frt
_DP_EXT = None
_DP_ERROR = []


def _dp_source():
    source = _gle_source()
    old = '''          for (int r = 0; r < 8; ++r)
            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);'''
    new = '''          // [router stage j][128 CTAs][8 rows][512 FP32 columns].
          // One 16-KiB transaction, identical destination bytes and consumers.
          const size_t wp = ((size_t)((kb & 2047) >> 9) * 128 + cta) * 4096;
          bulk_g2s(s0, reinterpret_cast<const uint8_t*>(ep.w32 + wp), 16384, full0 + 8 * slot);'''
    assert source.count(old) == 1
    return source.replace(old, new)


def _dp_build():
    global _DP_EXT
    try:
        source = _dp_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace('sk2_kernel', 'sk2_kernel_poolpack_' + tag)
                  .replace('launch_sk2', 'launch_sk2_poolpack_' + tag)
                  .replace('void sk2(', 'void sk2_poolpack_' + tag + '(')
                  .replace('  sk2(Ws,', '  sk2_poolpack_' + tag + '(Ws,')
                  .replace('void sk2f(', 'void sk2f_poolpack_' + tag + '(')
                  .replace('  sk2f(Ws,', '  sk2f_poolpack_' + tag + '(Ws,'))
        # Separate module, so the incumbent's host globals and compiled source
        # cannot change. No runtime source modification of the incumbent.
        cpp = '''void gle_pr(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, int64_t);
void sk2_set_pdl(int64_t);'''
        _DP_EXT = load_inline('decode_pooler_pack_' + tag,
            '#include <torch/extension.h>\n' + cpp, cuda_sources=source,
            functions=['gle_pr', 'sk2_set_pdl'], no_implicit_headers=True,
            extra_cflags=_dp_host_flags(['-O3']), extra_cuda_cflags=_dp_nvcc_flags(['-O3']), verbose=False)
    except Exception as ex:
        _DP_ERROR.append(f'{type(ex).__name__}: {str(ex)[-1200:]}')


def _dp_pack(weight):
    # Original [tile,row,quarter,j,k] -> [j,tile,quarter,row,k]. The flattened
    # (tile,quarter) is exactly cta = 4*tile+quarter in the unchanged kernel.
    return weight.view(32, 8, 4, 4, 512).permute(3, 0, 2, 1, 4).contiguous().view(256, 8192)


def _dp_unpack(weight):
    return weight.view(4, 32, 4, 8, 512).permute(1, 3, 2, 0, 4).contiguous().view(256, 8192)


def _gle_frt(xb, x32, state):
    if not state.get('_dp_active', False):
        result = _DP_FRT(xb, x32, state)
    else:
        bst, m = state['bfp'], xb.shape[0]
        _, stream, tab, pmax, b4, _ = bst['sk2']['pooler']
        if 'fused_part' not in bst:
            bst['fused_part'] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=xb.device)
            bst['fused_cnt'] = torch.zeros(32, dtype=torch.int32, device=xb.device)
        pooled = torch.empty((m, 4096), device=xb.device, dtype=torch.bfloat16)
        logits = torch.empty((m, 256), device=xb.device, dtype=torch.float32)
        _bfp_pdl(m, 'pool')
        _DP_EXT.sk2_set_pdl(bfp_SK2_PDL)
        _DP_EXT.gle_pr(stream, bst['sk2']['router'][1], xb, pooled, logits, tab, pmax, b4,
                       bst['fused_part'], bst['fused_cnt'], m, x32, state['_dp_weight'], int(GLE_ILV))
        result = pooled, logits
    observer = state.get('_dp_observe')
    if observer is not None:
        observer[:] = result
    return result


def kernel_fn(data, state):
    cfg = data['config']
    key = (cfg['batch'], cfg['q_len'], bool(cfg['window']))
    if state.get('dp_pick', {}).get(key, False):
        # Do not mutate shared state: capture and fresh-host-thread calls use
        # exactly the same incumbent layer, with only its router stream changed.
        return _DP_KERNEL(data, dict(state, _dp_active=True))
    return _DP_KERNEL(data, state)


def _dp_inputs(st, B, L, sliding, seed):
    # Public input distribution, independent seeds. Static addresses, fresh
    # input/KV contents at both validation passes. Reset only actual written KV.
    dev, T, per = st['qkv'].device, B * L, (80000 if B == 1 else 86000) + L
    total = B * per
    gen = torch.Generator(device=dev).manual_seed(seed)
    cfg = dict(batch=B, q_len=L, tokens=T, hidden=8192, num_q_heads=16, num_kv_heads=2,
               head_dim=64, page_size=1, window=2048 if sliding else 0,
               rope_theta=500000.0 if sliding else 0.0, qk_scale_factor=5.4730064863875,
               sm_scale=0.125, kv_scale=1.0, n_experts=256, top_k=8, local_experts=32)
    data = dict(config=cfg, stream=torch.empty((T, 8192), device=dev),
        attn_input=torch.empty((T, 8192), dtype=torch.bfloat16, device=dev),
        seq_lens=torch.empty(B, dtype=torch.int32, device=dev),
        positions=torch.empty(T, dtype=torch.int64, device=dev),
        cache_loc=torch.empty(T, dtype=torch.int64, device=dev),
        cache_seqlens=torch.empty(B, dtype=torch.int32, device=dev),
        cu_seqlens_q=torch.arange(0, T + 1, L, dtype=torch.int32, device=dev),
        scheduler_metadata=None, page_table=torch.empty((B, per), dtype=torch.int32, device=dev),
        k_cache=torch.empty((total, 1, 2, 64), dtype=torch.float8_e4m3fn, device=dev),
        v_cache=torch.empty((total, 1, 2, 64), dtype=torch.float8_e4m3fn, device=dev),
        k_scale=torch.ones((), device=dev), v_scale=torch.ones((), device=dev),
        local_start=torch.zeros(1, dtype=torch.int32, device=dev))
    mean = torch.randn(8192, generator=gen, device=dev)
    shared = torch.randn((4, 8192), generator=gen, device=dev)
    draft = torch.ones((B, L, 1), device=dev)
    if L == 8:
        draft[:, 1:] = 0.6
    draft = draft.view(T, 1)
    chunk = torch.empty((min(total, 1 << 18), 2, 64), device=dev)

    def refresh():
        seq = (torch.full((1,), 80000, dtype=torch.int64, device=dev) if B == 1 else
               torch.randint(74000, 86001, (B,), generator=gen, device=dev))
        data['seq_lens'].copy_(seq)
        data['cache_seqlens'].copy_(seq + L)
        data['page_table'].copy_(torch.randperm(total, generator=gen, device=dev).view(B, per))
        positions = seq[:, None] + torch.arange(L, device=dev)
        data['positions'].copy_(positions.flatten())
        data['cache_loc'].copy_(data['page_table'].gather(1, positions).flatten())
        coef = torch.randn((T, 4), generator=gen, device=dev)
        stream = (draft * (0.7 * mean + (0.9 / 4) ** 0.5 * (coef @ shared))
                  + (1.0 - 0.9 * draft.square()).sqrt()
                  * torch.randn((T, 8192), generator=gen, device=dev))
        data['stream'].copy_(stream)
        data['attn_input'].copy_(stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + 1e-5))
        u = torch.randn((2, 64), generator=gen, device=dev)
        u *= torch.rsqrt(u.square().mean(-1, keepdim=True))
        lo, hi = (0.3, 0.75) if sliding else (0.5, 0.62)
        c2 = lo + (hi - lo) * torch.rand((2, 1), generator=gen, device=dev)
        for key in ('k_cache', 'v_cache'):
            dst = data[key].view(total, 2, 64)
            for at in range(0, total, chunk.shape[0]):
                x = chunk[:min(chunk.shape[0], total - at)]
                x.normal_(generator=gen)
                if key == 'k_cache':
                    x.mul_((1.0 - c2).sqrt()).add_(c2.sqrt() * u)
                dst[at:at + x.shape[0]].copy_(x)
        data['local_start'].zero_()
    return data, refresh


def _dp_cache(data):
    return tuple(data[k].view(torch.uint8).view(-1, 2, 64).index_select(0, data['cache_loc'])
                 for k in ('k_cache', 'v_cache'))


def _dp_reset(data, before):
    for key, value in zip(('k_cache', 'v_cache'), before):
        data[key].view(torch.uint8).view(-1, 2, 64).index_copy_(0, data['cache_loc'], value)


def _dp_capture(fn):
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


def _dp_pair(graphs, flush, sink, reset, deadline, rounds=12):
    times, paired, orders = [[], []], [], [[], []]
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(rounds + 2):
        bt = [[], []]
        for i in ((0, 1, 1, 0) if r % 2 == 0 else (1, 0, 0, 1)):
            if _dp_time.monotonic() >= deadline:
                raise TimeoutError('DEC_POOL incomplete timing discarded')
            reset()
            torch.sum(flush, dim=0, out=sink)  # 512 MiB READ, outside interval
            a.record()
            graphs[i].replay()
            b.record()
            b.synchronize()
            us = a.elapsed_time(b) * 1000.0
            if not 0 < us < 1e7:
                raise RuntimeError('invalid CUDA event interval')
            bt[i].append(us)
        if r >= 2:
            for i in (0, 1):
                times[i].extend(bt[i])
            ratio = sum(bt[1]) / sum(bt[0])
            paired.append(ratio)
            orders[r % 2].append(ratio)
    med = _dp_stats.median
    return dict(old_us=med(times[0]), new_us=med(times[1]), ratio=med(times[1]) / med(times[0]),
                paired=med(paired), abba=med(orders[0]), baab=med(orders[1]))


def _dp_null_ok(result):
    return all(.995 <= result[k] <= 1.005 for k in ('ratio', 'paired', 'abba', 'baab'))


def _dp_admit(exact, ids_equal, timings):
    return bool(exact and ids_equal and len(timings) == 2 and all(
        _dp_null_ok(r['null']) and all(r[k] <= .992 for k in ('ratio', 'paired', 'abba', 'baab'))
        for r in timings))


def _dp_shape(st, B, L, sliding, flush, sink, deadline):
    label, key = f'{"d" if L == 1 else "v"}{B}{"S" if sliding else "F"}', (B, L, sliding)
    data, refresh = _dp_inputs(st, B, L, sliding, 617000 + 100 * B + 10 * L + sliding)
    refresh()
    base_state = dict(st, _dp_active=False, _dp_observe=[])
    cand_state = dict(st, _dp_active=True, _dp_observe=[])
    base, bo = _dp_capture(lambda: _DP_KERNEL(data, base_state))
    cand, co = _dp_capture(lambda: _DP_KERNEL(data, cand_state))
    exact, ids_equal, timings = True, True, []
    for turn in range(2):
        refresh()
        before = _dp_cache(data)
        reset = lambda: _dp_reset(data, before)
        reset()
        base.replay()
        busiest = int(torch.bincount((bo[2].long() // 32).flatten(), minlength=8).argmax())
        data['local_start'].fill_(32 * busiest)
        reset()
        base.replay()
        # Strict bit equality is deliberately stronger than judge tolerance.
        ref = tuple(t.clone() for t in (*bo, *base_state['_dp_observe'], *_dp_cache(data)))

        def check(output, state):
            actual = (*output, *state['_dp_observe'], *_dp_cache(data))
            same = len(actual) == len(ref) and len(state['_dp_observe']) == 2
            same &= all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(actual, ref))
            return same, torch.equal(output[2], ref[2])

        for graph, output, state in ((base, bo, base_state), (cand, co, cand_state), (cand, co, cand_state)):
            reset()
            graph.replay()
            torch.cuda.synchronize()
            same, ids = check(output, state)
            exact &= same
            ids_equal &= ids
        # Reverse null placement on pass 2 to expose monotone drift.
        if turn == 0:
            null = _dp_pair((base, base), flush, sink, reset, deadline)
        result = _dp_pair((base, cand), flush, sink, reset, deadline)
        same, ids = check(co, cand_state)  # final measured BAAB replay is candidate
        exact &= same
        ids_equal &= ids
        if turn == 1:
            null = _dp_pair((base, base), flush, sink, reset, deadline)
        result['null'] = null
        timings.append(result)
        admitted = turn == 1 and _dp_admit(exact, ids_equal, timings)
        print(f'DEC_POOL shape={label} T={B*L} pass={turn+1} old_us={result["old_us"]:.3f} '
              f'new_us={result["new_us"]:.3f} ratio={result["ratio"]:.5f} paired={result["paired"]:.5f} '
              f'null={null["ratio"]:.5f} exact={int(exact)} ids_equal={int(ids_equal)} admitted={int(admitted)} '
              f'abba={result["abba"]:.5f} baab={result["baab"]:.5f} '
              f'null_paired={null["paired"]:.5f} null_abba={null["abba"]:.5f} null_baab={null["baab"]:.5f}', flush=True)
    if _dp_admit(exact, ids_equal, timings):
        st['dp_pick'][key] = True


def prepare(weights):
    global MFAB_LAYER, _WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR
    start = _dp_time.monotonic()
    thread = threading.Thread(target=_dp_build, daemon=True)
    thread.start()
    old = MFAB_LAYER
    try:
        MFAB_LAYER = False  # unrelated incumbent diagnostics, never production
        st = _DP_PREPARE(weights)
    finally:
        MFAB_LAYER = old
    st['dp_pick'] = {}
    saved = (_WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR)
    _WS7, _CL7, _GUARDS, _GUARDED_IDS = {}, {}, [], set()
    try:
        thread.join(timeout=max(0.0, start + 230 - _dp_time.monotonic()))
        if thread.is_alive() or _DP_EXT is None:
            raise RuntimeError('optional build unavailable ' + ' '.join(_DP_ERROR))
        st['_dp_weight'] = _dp_pack(st['router_fp32'])
        if not torch.equal(_dp_unpack(st['_dp_weight']).view(torch.uint8), st['router_fp32'].view(torch.uint8)):
            raise RuntimeError('router pack not bit exact')
        _, stream, tab, pmax, _, _ = st['bfp']['sk2']['pooler']
        payload = stream.numel() * stream.element_size() + tab.numel() * tab.element_size() + st['_dp_weight'].numel() * 4
        print(f'DEC_POOL_BYTES weights_plus_escapes={payload} pooler={stream.numel()*stream.element_size()} '
              f'router={st["_dp_weight"].numel()*4} pmax={pmax} ctas=128 sms={st["nsm"]} '
              f'weight_floor_us={payload/4.8e6:.4f} router_weight_copies_per_cta=32->4', flush=True)
        with torch.no_grad():
            flush = torch.ones((512 << 20) // 4, device=st['qkv'].device, dtype=torch.float32)
            sink = torch.empty((), device=flush.device, dtype=torch.float32)
            for L in (1, 8):
                for sliding in (True, False):
                    label = f'{"d" if L == 1 else "v"}1{"S" if sliding else "F"}'
                    try:
                        _dp_shape(st, 1, L, sliding, flush, sink, start + 280)
                        _check_guards()
                    except Exception as ex:
                        st['dp_pick'].pop((1, L, sliding), None)
                        print(f'DEC_POOL shape={label} old_us=nan new_us=nan ratio=nan paired=nan null=nan '
                              f'exact=0 ids_equal=0 admitted=0 error={type(ex).__name__}:{str(ex)[-600:]}', flush=True)
            # The already-contiguous BF16 hi/lo router is explicitly untouched.
            # Do not select a no-op at T16/32 based on timer noise.
            for T in (16, 32):
                for suffix in ('S', 'F'):
                    print(f'DEC_POOL shape=d{T}{suffix} T={T} old_us=nan new_us=nan ratio=1 paired=1 null=nan '
                          'exact=1 ids_equal=1 admitted=0 reason=already_contiguous_incumbent', flush=True)
    except Exception as ex:
        st['dp_pick'].clear()
        print(f'DEC_POOL_ABORT {type(ex).__name__}: {str(ex)[-900:]} fallback=incumbent', flush=True)
    finally:
        torch.cuda.synchronize()
        (_WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR) = saved
        bfp__get_ext().sk2_set_pdl(bfp_SK2_PDL)
        if _GLE_EXT is not None:
            _GLE_EXT.sk2_set_pdl(bfp_SK2_PDL)
        if _DE_EXT is not None:
            _DE_EXT.sk2_set_pdl(bfp_SK2_PDL)
        print(f'DEC_POOL_DONE prepare_s={_dp_time.monotonic()-start:.2f} admitted={len(st["dp_pick"])} '
              'default=incumbent read_flush_MiB=512 rounds=12 warm_rounds=2 passes=2', flush=True)
    return st
