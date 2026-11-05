# Scheduling-only gate branch; admission and ceiling diagnostics run in prepare.
# Eligible native attention: one request, page 1, T=1/8 only. Never adopted
# without exact component/downstream checks and two cold ABBA timing passes.
import hashlib as _gg_hash
import statistics as _gg_stats
import time as _gg_time

_GG_PREPARE = prepare
_GG_KERNEL = kernel_fn
_GG_EXT = {}
_GG_ROWS = 1296                  # 54 * 24: retain the boundary tile's 16 gate rows
_GG_FLAG_SLOT = 32767           # unused by native one-request attention
_GG_SUMMARY = []


def _gg_replace(src, old, new, count=1):
    assert src.count(old) == count, (old[:100], src.count(old), count)
    return src.replace(old, new)


def _gg_sk_source(narrow):
    src = _gg_original_sk_source(narrow)
    src = _gg_replace(src, 'void sk2_set_pdl(int64_t n) {',
        'static int* g_gate_flags = nullptr;\nstatic int g_gate_branch = 0, g_gate_offset = 0;\n'
        'void gate_setup(at::Tensor flags,int64_t branch,int64_t offset) {\n'
        ' g_gate_flags=flags.numel()?flags.data_ptr<int>():nullptr; g_gate_branch=branch; g_gate_offset=offset;\n}\n'
        'void sk2_set_pdl(int64_t n) {')
    src = _gg_replace(src, 'int* __restrict__ rcnt, int pdl) {',
        'int* __restrict__ rcnt, int pdl, int* gate_flags, int gate_branch, int gate_offset) {')
    src = _gg_replace(src, '  const int dgv = SK2_DIAG ? dg : 0;',
        '  if (gate_branch) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
        '  const int dgv = SK2_DIAG ? dg : 0;')
    # The gate consumes the original layer input x, NOT the prefix output.
    # It must not wait on the prefix. Its entry trigger releases attention only
    # after every gate CTA has started (so no gate CTA can starve behind it).
    src = _gg_replace(src, 'if (pdl) asm volatile("griddepcontrol.wait;',
        'if (pdl && !gate_branch) asm volatile("griddepcontrol.wait;', count=2)
    src = _gg_replace(src, 'wsr, outr, part, rcnt, pdl));',
        'wsr, outr, part, rcnt, pdl, g_gate_flags, g_gate_branch, g_gate_offset));')
    src = _gg_replace(src, 'wsr, outr, part, rcnt, 0);',
        'wsr, outr, part, rcnt, 0, g_gate_flags, g_gate_branch, g_gate_offset);')
    src = _gg_replace(src, 'OutT* o = out + (size_t)tt * N + cta * R + row;',
        'OutT* o = out + (size_t)tt * N + gate_offset + cta * R + row;')
    marker = '  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32)'
    # All NC consumers finish their stores before leader release. Every CTA's
    # RMW is acq_rel: observing count=54 publishes ALL prefix output stores.
    release = r'''
  if (!gate_branch && gate_flags) {
    named_bar(1, NC);
    if (tid == 0) {
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                   : "=r"(old) : "l"(gate_flags) : "memory");
    }
  }
'''
    return _gg_replace(src, marker, release + marker)


def _gg_build_sk(narrow):
    src = _gg_sk_source(narrow)
    tag = _gg_hash.sha256(src.encode()).hexdigest()[:12]
    cpp = bfp_CPP_SRC[bfp_CPP_SRC.index('void sk2('):bfp_CPP_SRC.index('void sk2f(')]
    cpp += 'void sk2_set_pdl(int64_t n);\nvoid gate_setup(at::Tensor flags,int64_t branch,int64_t offset);'
    # The declaration TU uses torch::Tensor, unlike the CUDA TU's at::Tensor.
    cpp = cpp.replace('at::Tensor flags', 'torch::Tensor flags')
    _GG_EXT['narrow' if narrow else 'sk'] = load_inline('gate_overlap_sk_' + tag,
        '#include <torch/extension.h>\n' + cpp, cuda_sources=src,
        functions=['sk2', 'sk2_set_pdl', 'gate_setup'], no_implicit_headers=True,
        extra_cflags=['-O3', '-std=c++20'],
        extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)


def _gg_att_source():
    src = ATT7_CUDA_SRC
    marker = '__device__ __forceinline__ void pdl_wait() {'
    wait = r'''
__device__ __forceinline__ void gate_qkv_wait(const Params& p) {
  if (threadIdx.x == 0) {
    int ready;
    SPIN_T0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ready) : "l"(p.ws_cnt + 32767) : "memory");
      if (ready == 54) break;
      SPIN_CHECK;
      __nanosleep(32);
    }
  }
  // Eligible kernels are NON-FLAT (NP=0): every CTA thread reaches this.
  __syncthreads();
}
'''
    src = _gg_replace(src, marker, wait + marker)
    # QKV, not the gate, is the only prerequisite for query/KV work. Retain
    # launch_dependents and the entire original attention arithmetic.
    old = '  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel\'s output; nothing above reads or writes it'
    src = _gg_replace(src, old, '  gate_qkv_wait(p);   // prefix release, not gate completion')
    # On a speculative sequence-length miss, a CTA with NO real keys must not
    # reach the ready wait after the real active splits have reset it. Reject
    # the speculation before staging qkv; the caller then retries the real
    # length, and its existing empty-split return avoids the wait entirely.
    # This also makes every active split consume the ready count exactly once.
    marker = '  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;'
    src = _gg_replace(src, marker,
        '  if constexpr (!FLAT && (MODE & 8) != 0) {\n'
        '    if (seq_real_ != seq) return false;   // before any qkv/async work\n'
        '  }\n' + marker)
    # Gate reads happen only in finalizers on eligible non-cluster kernels.
    # A true PDL wait here joins the gate branch and publishes its stores.
    lines = []
    gate_loads = 0
    for line in src.splitlines(keepends=True):
        if 'graw' in line and 'p.qkv +' in line:
            lines.append(' ' * (len(line) - len(line.lstrip())) + 'pdl_wait();   // gate branch completion\n')
            gate_loads += 1
        lines.append(line)
    assert gate_loads == 4, gate_loads
    src = ''.join(lines)
    # No memset/join kernel. Native merge counters already prove ALL active
    # splits have passed gate_qkv_wait. Reset our one counter at that existing
    # single-leader reset, before the next layer invocation can start.
    src = _gg_replace(src, '  if (tid == 0) p.ws_cnt[bp] = 0;',
        '  if (tid == 0) { p.ws_cnt[bp] = 0; p.ws_cnt[32767] = 0; }')
    src = _gg_replace(src, '        p.ws_cnt[24576 + bp] = ep + 1;',
        '        p.ws_cnt[32767] = 0;\n        p.ws_cnt[24576 + bp] = ep + 1;')
    # These branches are compiled out by today's OKC_PARITY=1, SENSE=0 flags;
    # keep reset correct if the incumbent changes its cooperative barrier.
    src = _gg_replace(src, '        p.ws_cnt[bp] = 0;\n        asm volatile("st.release.gpu.global.s32',
        '        p.ws_cnt[bp] = 0; p.ws_cnt[32767] = 0;\n        asm volatile("st.release.gpu.global.s32')
    src = _gg_replace(src, '          p.ws_cnt[16384 + bp] = 0;',
        '          p.ws_cnt[16384 + bp] = 0; p.ws_cnt[32767] = 0;')
    return src


def _gg_build_att():
    src = _gg_att_source()
    tag = _gg_hash.sha256(src.encode()).hexdigest()[:12]
    _GG_EXT['att'] = load_inline('gate_overlap_att_' + tag, ATT7_CPP_SRC, cuda_sources=src,
        extra_cflags=['-O3', '-std=c++20'],
        functions=['run'], extra_cuda_cflags=['-DOKC_QDIRECT=0', '-DOKC_PK2=0', '-DOKC_W0ENABLE=1', '-O3',
        '-DOKC_SM90=1', '--expt-relaxed-constexpr', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)


def _gg_supported(data, state):
    cfg = data['config']
    T = cfg['tokens']
    return (cfg['batch'] == 1 and cfg['q_len'] in (1, 8) and cfg['page_size'] == 1 and T in (1, 8)
            and cfg['num_q_heads'] == NUM_Q_HEADS and cfg['num_kv_heads'] == NUM_KV_HEADS
            and state.get('att_ok') and _supported7(cfg) and _att_fits7(data, cfg) and _pdl(T)
            and state.get('mm_pick', {}).get((T, 'qkv')) == 'bfp' and state.get('bfp') is not None
            and T in state.get('gate_streams', {}))


def _gg_pack(st):
    out = {}
    for T in (1, 8):
        ci, stream, tab, pmax, b4, nrows = st['bfp']['sk2'][bfp__sk2_name('qkv_gate', T)]
        assert stream.shape[1] == 96 and nrows == QKV_ROWS
        out[T] = (ci, stream[:, :54].contiguous(), stream[:, 54:].contiguous(),
                  tab[:54].contiguous(), tab[54:].contiguous(), pmax, b4)
    st['gate_streams'] = out


def _gg_project(data, state):
    T = data['config']['tokens']
    x = data['attn_input']
    ci, prefix, gate, ptab, gtab, pmax, b4 = state['gate_streams'][T]
    out = torch.empty((T, QKV_ROWS), device=x.device, dtype=torch.bfloat16)
    _, _, cnt = _workspace7(T * 16 * 8, x.device)
    flag = cnt[_GG_FLAG_SLOT:]
    ext = _GG_EXT['narrow' if T == 8 else 'sk']
    _bfp_pdl(T, 'qkv')
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gate_setup(flag, 0, 0)
    ext.sk2(prefix, x, out, ptab, pmax, b4, T, QKV_ROWS, HIDDEN, ci, 8, 0, state['bfp']['dummy'])
    ext.gate_setup(flag, 1, _GG_ROWS)
    ext.sk2(gate, x, out, gtab, pmax, b4, T, QKV_ROWS, HIDDEN, ci, 8, 0, state['bfp']['dummy'])
    return out


def _gg_prefix_only(data, state):
    # Prepare-only ceiling: no gate branch and, critically, no ready-counter
    # publication (there is no attention consumer to reset it in this graph).
    T, x = data['config']['tokens'], data['attn_input']
    ci, prefix, _, ptab, _, pmax, b4 = state['gate_streams'][T]
    out = torch.empty((T, QKV_ROWS), device=x.device, dtype=torch.bfloat16)
    ext = _GG_EXT['narrow' if T == 8 else 'sk']
    _bfp_pdl(T, 'qkv')
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gate_setup(state['bfp']['dummy'][:0], 0, 0)
    ext.sk2(prefix, x, out, ptab, pmax, b4, T, QKV_ROWS, HIDDEN, ci, 8, 0, state['bfp']['dummy'])
    return out


def _gg_attention(data, state, qkv):
    global _PF7
    cfg = data['config']
    T = cfg['tokens']
    _ps = state['bfp']['sk2']['pooler'][1]
    _PF7 = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))
    try:
        out = _launch7(dict(data, qkv=qkv), _mod=_GG_EXT['att'], pdl=True)
    finally:
        _PF7 = None
    return out.view(T, Q_DIM)


def _gg_layer(data, state):
    qkv = _gg_project(data, state)
    o_in = _gg_attention(data, state, qkv)
    observer = state.get('_gg_observe')
    if observer is not None:
        observer.update(qkv=qkv, o_in=o_in)
    return _after_attention(data, state, o_in, None, data['config']['tokens'])


def kernel_fn(data, state):
    cfg = data['config']
    key = (cfg['batch'], cfg['q_len'], bool(cfg['window']))
    if state.get('gate_pick', {}).get(key) and _gg_supported(data, state):
        _glue_bind_context(data['attn_input'])  # same fresh-host-thread contract as the incumbent
        return _gg_layer(data, state)
    return _GG_KERNEL(data, state)


def _gg_default(data, st):
    T = data['config']['tokens']
    qkv = _bfp_gemm(st, 'qkv', data['attn_input'], T, QKV_ROWS, HIDDEN)
    o_in = _decode_attention7(data, st, qkv)
    st['_gg_observe'].update(qkv=qkv, o_in=o_in)
    return _after_attention(data, st, o_in, None, T)


def _gg_snapshot(output, observer, data):
    return tuple(t.clone() for t in (observer['qkv'], observer['o_in'], *output, *_dg_cache(data)))


def _gg_same(output, observer, data, reference):
    actual = (observer['qkv'], observer['o_in'], *output, *_dg_cache(data))
    return tuple(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(actual, reference))


def _gg_null_ok(timing):
    return all(.995 <= timing[k] <= 1.005 for k in ('ratio', 'paired', 'abba', 'baab'))


def _gg_admit(exact, timings):
    return (exact and len(timings) == 2 and all(
        _gg_null_ok(t['null']) and all(t[k] <= .992 for k in ('ratio', 'paired', 'abba', 'baab'))
        for t in timings))


def _gg_ceiling(data, st, label, flush, sink, deadline):
    T = data['config']['tokens']
    full, fo = _dg_capture(lambda: _bfp_gemm(st, 'qkv', data['attn_input'], T, QKV_ROWS, HIDDEN))
    prefix, po = _dg_capture(lambda: _gg_prefix_only(data, st))
    torch.cuda.synchronize()
    exact = torch.equal(fo[:, :_GG_ROWS].contiguous().view(torch.uint8),
                        po[:, :_GG_ROWS].contiguous().view(torch.uint8))
    for turn in range(2):
        result = _dg_pair((full, prefix), flush, sink, lambda: None, blocks=8, deadline=deadline)
        row = (f'GATE_CEIL shape={label} pass={turn+1} qkv_us={result["base"]:.3f} '
               f'qkv_prefix_us={result["short"]:.3f} save_us={result["save"]:.3f} '
               f'ratio={result["ratio"]:.5f} paired={result["paired"]:.5f} exact={int(exact)} '
               'skipped_gate_rows=1008/1024 diagnostic_only=1')
        _GG_SUMMARY.append(row)
        print(row, flush=True)


def _gg_shape(st, B, L, sliding, flush, sink, deadline):
    label, key = f'{"d" if L == 1 else "v"}{B}{"S" if sliding else "F"}', (B, L, sliding)
    data, refresh = _dg_inputs(st, B, L, sliding, 535353 + 100 * L + sliding)
    refresh()
    if not _gg_supported(data, st):
        _GG_SUMMARY.append(f'GATE_ARM shape={label} old_us=nan new_us=nan ratio=nan paired=nan null=nan exact=0 unsupported=1')
        return
    # Always measure the projection ceiling, including when attention failed
    # to build or the full overlap arm is subsequently rejected by admission.
    if ('narrow' if L == 8 else 'sk') not in _GG_EXT:
        _GG_SUMMARY.append(f'GATE_ARM shape={label} old_us=nan new_us=nan ratio=nan paired=nan null=nan exact=0 sk_build=0')
        return
    _gg_ceiling(data, st, label, flush, sink, deadline)
    if 'att' not in _GG_EXT:
        _GG_SUMMARY.append(f'GATE_ARM shape={label} old_us=nan new_us=nan ratio=nan paired=nan null=nan exact=0 att_build=0')
        return
    # The captured addresses, state and observer objects remain strongly owned
    # until both validation/timing passes have completed.
    sb, sc = dict(st, _gg_observe={}), dict(st, _gg_observe={})
    _, _, cnt = _workspace7(L * 16 * 8, st['qkv'].device)
    cnt[_GG_FLAG_SLOT:].zero_()
    base, bo = _dg_capture(lambda: _gg_default(data, sb))
    candidate, co = _dg_capture(lambda: _gg_layer(data, sc))
    valid, timings = True, []
    all_checks = [True] * 7
    try:
        for turn in range(2):
            if _gg_time.monotonic() >= deadline:
                raise TimeoutError('gate prepare validation budget exhausted')
            refresh()
            before = _dg_cache(data)
            _dg_reset(data, before)
            base.replay()
            ids = bo[2]
            busiest = int(torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=8).argmax())
            data['local_start'].fill_(busiest * LOCAL_EXPERTS)
            _dg_reset(data, before)
            base.replay()
            reference = _gg_snapshot(bo, sb['_gg_observe'], data)
            # Repeated default must itself be bit-repeatable. Do NOT weaken
            # the user's strict gate when upstream atomics are nondeterministic.
            for graph, output, ss in ((base, bo, sb), (candidate, co, sc), (candidate, co, sc)):
                _dg_reset(data, before)
                graph.replay()
                torch.cuda.synchronize()
                checks = _gg_same(output, ss['_gg_observe'], data, reference)
                all_checks = [a and b for a, b in zip(all_checks, checks)]
                valid &= all(checks)
            flag_zero = int(cnt[_GG_FLAG_SLOT]) == 0
            valid &= flag_zero
            restore = lambda: _dg_reset(data, before)
            # Both arms see 512 MiB READ-flush, outside the event interval.
            # The A/A control has the same sample count and alternating orders.
            # Alternate control placement too, rather than always testing it last.
            if turn == 0:
                null = _dg_pair((base, base), flush, sink, restore, blocks=12, deadline=deadline)
            result = _dg_pair((base, candidate), flush, sink, restore, blocks=12, deadline=deadline)
            # Twelve blocks end with BAAB, hence the last replay is candidate.
            # Check its KV bytes BEFORE a later null arm can overwrite them.
            checks = _gg_same(co, sc['_gg_observe'], data, reference)
            all_checks = [a and b for a, b in zip(all_checks, checks)]
            flag_zero = int(cnt[_GG_FLAG_SLOT]) == 0
            valid &= all(checks) and flag_zero
            if turn == 1:
                null = _dg_pair((base, base), flush, sink, restore, blocks=12, deadline=deadline)
            result['null'] = null
            timings.append(result)
            row = (f'GATE_ARM shape={label} pass={turn+1} old_us={result["base"]:.3f} new_us={result["short"]:.3f} '
                   f'ratio={result["ratio"]:.5f} paired={result["paired"]:.5f} null={null["ratio"]:.5f} '
                   f'exact={int(valid)} abba={result["abba"]:.5f} baab={result["baab"]:.5f} '
                   f'null_paired={null["paired"]:.5f} null_abba={null["abba"]:.5f} null_baab={null["baab"]:.5f} '
                   f'null_ok={int(_gg_null_ok(null))} bits={"".join(str(int(v)) for v in all_checks)} flag_zero={int(flag_zero)}')
            _GG_SUMMARY.append(row)
            print(row, flush=True)
    finally:
        torch.cuda.synchronize()
    admitted = _gg_admit(valid, timings)
    if admitted:
        st['gate_pick'][key] = True
    _GG_SUMMARY.append(f'GATE_PICK shape={label} adopted={int(admitted)} exact={int(valid)} fallback=incumbent')


def prepare(weights):
    global MFAB_LAYER
    start, old = _gg_time.monotonic(), MFAB_LAYER
    _GG_SUMMARY.clear()
    deadline = start + 280  # leave a 20 s margin below the judge's 300 s limit
    errors = []
    def build(fn):
        try:
            fn()
        except Exception as ex:
            errors.append(f'{type(ex).__name__}: {str(ex)[-1200:]}')
    threads = [threading.Thread(target=build, args=(fn,), daemon=True) for fn in
               (lambda: _gg_build_sk(False), lambda: _gg_build_sk(True), _gg_build_att)]
    for th in threads:
        th.start()
    try:
        MFAB_LAYER = False
        st = _GG_PREPARE(weights)
    finally:
        MFAB_LAYER = old
    st['gate_pick'] = {}
    for th in threads:
        th.join(timeout=max(0., start + 240 - _gg_time.monotonic()))
    try:
        for error in errors:
            print(f'GATE_BUILD_ERR {error}', flush=True)
        if any(th.is_alive() for th in threads):
            print('GATE_BUILD_ERR optional build budget exhausted', flush=True)
        _gg_pack(st)
        flush = torch.ones((512 << 20) // 4, device=st['qkv'].device, dtype=torch.float32)
        sink = torch.empty((), device=flush.device, dtype=torch.float32)
        for L in (1, 8):
            for sliding in (True, False):
                label = f'{"d" if L == 1 else "v"}1{"S" if sliding else "F"}'
                if _gg_time.monotonic() >= deadline:
                    _GG_SUMMARY.append(f'GATE_ARM shape={label} old_us=nan new_us=nan ratio=nan paired=nan null=nan exact=0 budget=1')
                    continue
                try:
                    with torch.no_grad():
                        _gg_shape(st, 1, L, sliding, flush, sink, deadline)
                except Exception as ex:
                    # A shape has not been admitted until BOTH passes complete.
                    st['gate_pick'].pop((1, L, sliding), None)
                    _GG_SUMMARY.append(f'GATE_ARM shape={label} old_us=nan new_us=nan ratio=nan paired=nan null=nan exact=0 aborted=1')
                    print(f'GATE_ERR shape={label} {type(ex).__name__}: {str(ex)[-900:]} fallback=incumbent', flush=True)
                    torch.cuda.synchronize()
                finally:
                    # Rejected/aborted diagnostic graphs must not poison a
                    # previously admitted shape's first production invocation.
                    ws = _WS7.get(str(st['qkv'].device))
                    if ws is not None:
                        ws[2][_GG_FLAG_SLOT:].zero_()  # prepare only
        _check_guards()
    except Exception as ex:
        st['gate_pick'] = {}
        print(f'GATE_ERR admission_aborted {type(ex).__name__}: {str(ex)[-1400:]} fallback=incumbent', flush=True)
        torch.cuda.synchronize()
    print('GATE_FINAL_BEGIN', flush=True)
    for row in _GG_SUMMARY:
        print(row, flush=True)
    print(f'GATE_FINAL_END prepare_s={_gg_time.monotonic()-start:.1f} adopted={len(st["gate_pick"])}/4', flush=True)
    return st
