
# Dense S3d (DENSE_LEDGER.md): DIAGNOSTIC ONLY, no routing change. Hypothesis H32: at M=32 the GEMV kernels are
# limited by per-warp x reloads. v18s proxies of the incumbent splits (qkv (1,4,8,4), pooler (1,8,16,1), FMT 1,
# MT 4) are timed full (DIAG 0, verified by v18s_plan) and without x loads (DIAG 2, same packed stream), each
# against the incumbent dispatch as ABBA paired cold-L2 CUDA-graph replays, 16 pairs, raw samples printed.
_S3D_CASES = {
    'qkv_gate': ((1, 4, 8, 4), 2, 2304, 8192),
    'pooler': ((1, 8, 16, 1), 0, 4096, 8192),
}
_S3D_DIAGS = (0, 2)


def _s3d_inst(layer, diag):
    ((wr, wk, ku, s), red, n, k) = _S3D_CASES[layer]
    return (1, 4, 1, 1, red, diag, wr * wk * 32, 0, n, k, wr, wk, ku, s, 6), (wr, wk, ku, s)


# --- runtime ---
import time as _s3d_time


def _s3d_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode='thread_local'):
        fn()
    return g


def _s3d_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s3d_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_s3d_once(ga, flush)); tb.append(_s3d_once(gb, flush))
        else:
            tb.append(_s3d_once(gb, flush)); ta.append(_s3d_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sorted(ta)[len(ta) // 2], sorted(tb)[len(tb) // 2], ta, tb


def _s3d_diag(state, weights):
    t0 = _s3d_time.monotonic()
    insts = [_s3d_inst(l, d)[0] for l in _S3D_CASES for d in _S3D_DIAGS]
    ext = v18s_build(insts, 's3d_v18s_m32_diag')
    ids = {inst: i for (i, inst) in enumerate(insts)}
    print(f'S3D_BUILD {_s3d_time.monotonic() - t0:.0f}s', flush=True)
    dev = weights['pooler'].device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261033)
    (packs, streams) = ({}, {})
    for layer in _S3D_CASES:
        (_, _, n, k) = _S3D_CASES[layer]
        x = torch.randn((32, k), generator=gen, device=dev).to(torch.bfloat16)
        data = {'x': x, 'config': {'m': 32, 'n': n, 'k': k, 'layer': layer}}
        try:
            ref = _s3d_prev_kernel(data, state).double()
            gbase = _s3d_graph(lambda: _s3d_prev_kernel(data, state))
            (inst0, cfg) = _s3d_inst(layer, 0)
            plan = v18s_plan(ext, ids, packs, streams, weights[layer], layer, (layer, 'm', 32), inst0, cfg)
            if plan is None:
                print(f'S3D_SKIP {layer} plan/verify failed', flush=True)
                continue
            print(f'S3D_PLAN {layer} cfg {cfg} verified (one-hot + FP64) max_esc {plan["max_esc"]}', flush=True)
            for diag in _S3D_DIAGS:
                inst = _s3d_inst(layer, diag)[0]
                out = torch.empty((32, n), dtype=torch.bfloat16, device=dev)
                run = lambda inst=inst, out=out: v18s_launch_raw(ext, ids[inst], plan, x, out, n, cfg)
                rc = run()
                torch.cuda.synchronize()
                if rc:
                    print(f'S3D_SKIP {layer} diag{diag} rc {rc}', flush=True)
                    continue
                if diag == 0:
                    print(f'S3D_AGREE {layer} diag0_vs_incumbent {float((out.double() - ref).norm() / ref.norm()):.1e}', flush=True)
                g = _s3d_graph(run)
                (med, bmed, cmed, ta, tb) = _s3d_paired(gbase, g, flush)
                print(f'S3D {layer} m32 diag{diag} ratio_base_over_cand {med:.4f} base_med {bmed:.2f} cand_med {cmed:.2f} base_us '
                      + ','.join(f'{v:.1f}' for v in ta) + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
                del g
            del gbase
        except Exception as ex:
            print(f'S3D_FAIL {layer} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
            torch.cuda.synchronize()
    del flush
    torch.cuda.synchronize()
    print(f'S3D_DONE s={_s3d_time.monotonic() - t0:.1f}', flush=True)


_s3d_prev_kernel = kernel_fn
_s3d_prev_prepare = prepare


def prepare(weights):
    state = _s3d_prev_prepare(weights)
    try:
        _s3d_diag(state, weights)
    except Exception as ex:
        print(f'S3D_DIAG off {type(ex).__name__}: {str(ex)[:300]}', flush=True)
    torch.cuda.synchronize()
    return state
