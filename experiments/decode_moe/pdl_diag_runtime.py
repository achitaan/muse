"""Prepare-only diagnostic appended by build_pdl_diag.py; no production dispatch edits."""

_PDLD_EXT = {}
_PDLD_ERRORS = {}


def _pdld_build():
    for name, src in _PDLD_SOURCES.items():
        fused = name.startswith('r')
        cpp = _R2_CPP if fused else _PDLD_O_CPP
        cpp += '\nint pdl_info(int, int);\n'
        fn = 'r2_launch' if fused else 'pdl_o_launch'
        try:
            tag = hashlib.sha256(src.encode()).hexdigest()[:12]
            _PDLD_EXT[name] = load_inline('okc_pdld_' + name + '_' + tag, cpp,
                cuda_sources=src, functions=[fn, 'pdl_info'], extra_cuda_cflags=['-O3', '-lineinfo'],
                no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
            print('PDL_DIAG BUILD', name, tag, flush=True)
        except Exception as e:
            _PDLD_ERRORS[name] = repr(e)[-1800:]
            print('PDL_DIAG BUILD_ERROR', name, _PDLD_ERRORS[name], flush=True)


_pdld_thread = _okh_bg(_pdld_build)


def _pdld_rc(rc, what):
    if rc:
        raise RuntimeError('PDL diagnostic ' + what + ' rc=' + str(rc))


def _pdld_o(x, plan, ext):
    m = x.shape[0]
    out = torch.empty((m, 8192), dtype=torch.bfloat16, device=x.device)
    wr, wk, ku, s = plan['cfg']
    _pdld_rc(ext.pdl_o_launch(plan['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(),
        m, 8192, 1024, wr, wk, ku, s, plan['t01'], plan['u01'], torch.cuda.current_stream().cuda_stream), 'O')
    return out


def _pdld_r(xb, xr, plan, ext, scratch, state):
    m, dev = xb.shape[0], xb.device
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
    ow = torch.empty((m, 8), dtype=torch.float32, device=dev)
    oi = torch.empty((m, 8), dtype=torch.int32, device=dev)
    wr, wk, ku, s = plan['cfg']
    _pdld_rc(ext.r2_launch(plan['wp'].data_ptr(), xb.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(),
        m, 4096, 8192, wr, wk, ku, s, plan['t01'], plan['u01'], torch.cuda.current_stream().cuda_stream,
        xr.data_ptr(), state['rf']['weight'].data_ptr(), state['rf']['bias'].data_ptr(),
        scratch['parts'].data_ptr(), scratch['counter'].data_ptr(), ow.data_ptr(), oi.data_ptr(), 0), 'R2f')
    return out, ow, oi


def _pdld_equal(a, b):
    if a is None or b is None or not _okh_same(a, b):
        return False
    if not all(bool(torch.isfinite(x).all()) for x in a + tuple(b)):
        return False
    # torch.equal alone treats +0 and -0 alike. Compare their storage bits too.
    return _okh_same(tuple(x.contiguous().view(torch.uint8) for x in a),
                     tuple(x.contiguous().view(torch.uint8) for x in b))


def _pdld_probe(fns, inputs, gen):
    captures = {}
    for lab, fn in fns.items():
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = fn()
        captures[lab] = (gr, out)
    ok = True
    for _ in range(3):
        for x in inputs:
            x.normal_(generator=gen)
        reference = None
        for lab, (gr, out) in captures.items():
            gr.replay()
            torch.cuda.synchronize()
            if lab == 'normal':
                reference = tuple(x.clone() for x in out)
            else:
                ok = ok and _pdld_equal(reference, out)
    del captures
    return ok


def _pdld_select(state, t0):
    dev = state['rf']['weight'].device
    dense = state['achita_dense']
    print('PDL_DIAG CONFIG production_enabled=False spin_ns=5000 reps_per_order=30 '
          'orders=normal,pdl;pdl,normal bytewise=True', flush=True)
    for m in (1, 8):
        if not _okh_budget(t0, 278.0):
            print('PDL_DIAG BUDGET_SKIP', m, flush=True)
            break
        oname, rname = 'o' + str(m), 'r' + str(m)
        if m not in state.get('r2sel', ()) or not all(k in _PDLD_EXT for k in (oname, rname)):
            print('PDL_DIAG SKIP', m, 'r2sel=', state.get('r2sel'), 'built=', sorted(_PDLD_EXT), flush=True)
            continue
        osub = _r2_find(dense, 'v12_o1' if m == 1 else 'v18') or {}
        op = osub.get('plans', {}).get(('o', m))
        rp = state['r2'][m][1]
        if op is None:
            print('PDL_DIAG SKIP', m, 'missing O plan', flush=True)
            continue
        for name, plan in ((oname, op), (rname, rp)):
            inst = _PDLD_INST[name]
            if plan['fmt'] != inst[0] or tuple(plan['cfg']) != tuple(inst[10:14]):
                raise RuntimeError('PDL plan/source mismatch: ' + name)
            info = _PDLD_EXT[name].pdl_info(m, 2)
            occ = _PDLD_EXT[name].pdl_info(m, 1)
            if info < 0 or occ < 1:
                raise RuntimeError('PDL resources invalid ' + name + ' ' + str((info, occ)))
            print('PDL_DIAG RESOURCE', name, 'regs_local=', info, 'blocks_per_sm=', occ,
                  'source=', _PDLD_HASHES[name], 'instance=', inst, flush=True)
        ext_o, ext_r = _PDLD_EXT[oname], _PDLD_EXT[rname]
        scratch = {'parts': torch.empty_like(state['rf']['parts']),
                   'counter': torch.zeros_like(state['r2cnt'])}
        gen = torch.Generator(device=dev).manual_seed(57001 + m)
        xq = torch.randn(m, 8192, device=dev, generator=gen).to(torch.bfloat16)
        xo = torch.randn(m, 1024, device=dev, generator=gen).to(torch.bfloat16)
        xb = torch.randn(m, 8192, device=dev, generator=gen).to(torch.bfloat16)
        xr = torch.randn(m, 8192, device=dev, generator=gen)

        def spin():
            _pdld_rc(_PDLD_EXT['o1'].pdl_spin_launch(5000, torch.cuda.current_stream().cuda_stream), 'spin')

        def ofn(pdl):
            spin()
            o = _pdld_o(xo, op, ext_o) if pdl else _achita_dense_project(xo, 'o', state)
            return (o,)

        def rfn(pdl):
            spin()
            return (_pdld_r(xb, xr, rp, ext_r, scratch, state) if pdl else _r2_pool_route(xb, xr, state))

        def chain(pdl):
            # QKV is real; spins stand in for attention and the boundary norm.
            # Projection inputs are independent probes, not a full Decode step.
            q = _achita_dense_project(xq, 'qkv_gate', state)
            o = ofn(pdl)[0]
            r = rfn(pdl)
            return None if r is None else (q, o) + tuple(r)

        savings = {}
        for kind, fn in (('spin_o', ofn), ('spin_r2f', rfn), ('synthetic_chain', chain)):
            if not _okh_budget(t0, 278.0):
                print('PDL_DIAG BUDGET_SKIP', m, kind, flush=True)
                break
            fns = {'normal': lambda: fn(False), 'pdl': lambda: fn(True)}
            ok = _pdld_probe(fns, (xq, xo, xb, xr), gen)
            print('PDL_DIAG BITWISE', m, kind, ok, 'changing_input_graph_replays=3', flush=True)
            if not ok:
                savings[kind] = None
                continue
            by_order = []
            for order in (('normal', 'pdl'), ('pdl', 'normal')):
                if not _okh_budget(t0, 278.0):
                    break
                med = _okh_time({lab: fns[lab] for lab in order}, dev, reps=30)
                delta = med['normal'] - med['pdl']
                by_order.append(delta)
                print('PDL_DIAG TIME M=' + str(m) + ' kind=' + kind + ' order=' + ','.join(order) +
                      ' normal_us=%.6f pdl_us=%.6f saving_us=%.6f ratio=%.6f pick=%s' %
                      (med['normal'], med['pdl'], delta, med['pdl'] / med['normal'],
                       _okh_pick(med, 'normal', .97)), flush=True)
            savings[kind] = min(by_order) if len(by_order) == 2 else None
        oo, rr, cc = (savings.get(k) for k in ('spin_o', 'spin_r2f', 'synthetic_chain'))
        eligible = all(x is not None for x in (oo, rr, cc)) and oo >= 1.5 and rr >= 1.5 and cc >= 3.0
        print('PDL_DIAG DECISION M=' + str(m), 'conservative_savings_us=', savings,
              'combined_ceiling_pass=', eligible, 'production_enabled=False', flush=True)
        del scratch


_pdld_prev_prepare = prepare


def prepare(weights):
    t0 = _okh_clock.perf_counter()
    state = _pdld_prev_prepare(weights)
    _pdld_thread.join(max(0.0, 274.0 - (_okh_clock.perf_counter() - t0)))
    if _pdld_thread.is_alive() or not _okh_budget(t0, 278.0) or state.get('rf') is None or 'o1' not in _PDLD_EXT:
        print('PDL_DIAG SKIP_PREPARE', _okh_clock.perf_counter() - t0, sorted(_PDLD_EXT), flush=True)
        return state
    try:
        _pdld_select(state, t0)
    except Exception as e:
        print('PDL_DIAG ERROR', repr(e)[-1200:], flush=True)
    torch.cuda.synchronize()
    print('PDL_DIAG DONE elapsed_s=', _okh_clock.perf_counter() - t0,
          'production_enabled=False', flush=True)
    return state
