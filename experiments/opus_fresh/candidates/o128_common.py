
# Opus candidate O128 (CPU-prepared; GPU correctness and speed unmeasured).
# Lossless 12-bit RS-WGMMA m32w_gemm at BN=128 for exactly M=128/256 qkv_gate/pooler/unpooler.
# Per (M, layer) it is enabled only after prepare-time bitwise one-hot and random-input checks and a
# paired cold-L2 graph-timing gate (new < 0.97 x incumbent); any failure keeps the incumbent path.
import threading as _o128_threading
import time as _o128_time
_O128_SHAPES = ((128, 'qkv_gate'), (128, 'pooler'), (128, 'unpooler'), (256, 'qkv_gate'), (256, 'pooler'), (256, 'unpooler'))
_O128_VARIANTS = ((1, 2, 2), (1, 4, 2), (2, 2, 1))
_O128_OLD = 'if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));'


def _o128_source():
    core = tt_mw_CUDA_CORE.replace(_O128_OLD, _O128_OLD + '\n      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));')
    assert core.count('mma128(') == 1
    cases, attrs = [], []
    for i, (wk, split, cap) in enumerate(_O128_VARIANTS):
        p = f'128,64,{wk},2,2,true,{cap}'
        cases.append(f'case {i}: return launch_t<{p}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{p}>);break;')
    host = tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))
    return tt_mw_CUDA_PREFIX + '\n'.join(tt_mw_mma(n) for n in (16, 32, 64, 128)) + core + tt_mw_CUDA_MAP + host


def _o128_build(box):
    try:
        src = _o128_source()
        box['ext'] = load_inline('o128_' + hashlib.sha256(src.encode()).hexdigest()[:12], tt_mw_CPP, cuda_sources=src,
                                 functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False,
                                 extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])
    except Exception as e:
        box['error'] = repr(e)


def _o128_start():
    box = {}
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
        box['thread'] = _o128_threading.Thread(target=_o128_build, args=(box,), daemon=True)
        box['thread'].start()
    return box


def _o128_launch(ext, i, pk, x):
    out = torch.empty((x.shape[0], pk['n']), device=x.device, dtype=torch.bfloat16)
    if tt_mw_launch_raw(ext, i, pk, x, out):
        raise RuntimeError('o128 launch failed')
    return out


def _o128_exact(ext, i, pk, w, gen):
    if not all(tt_mw_verify_onehot(ext, i, pk, w, m) for m in (128, 256)):
        return False
    x = torch.randn((256, w.shape[1]), device=w.device, generator=gen).to(torch.bfloat16)
    x[3].zero_()
    y = _o128_launch(ext, i, pk, x).double()
    ref = x.double() @ w.double().T
    row = (y - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(0.1 * ref.norm(dim=1).mean())
    return bool(torch.isfinite(y).all()) and float((y - ref).norm() / ref.norm()) < 2e-3 and float(row.max()) < 1e-2


def _o128_select(box, weights, base_fn, started, limit=235.0):
    if 'thread' not in box:
        return {}
    box['thread'].join()
    if 'ext' not in box or _o128_time.monotonic() - started > limit:
        print('O128_SKIP', box.get('error'), round(_o128_time.monotonic() - started, 1), flush=True)
        return {}
    ext, dev = box['ext'], weights['qkv_gate'].device
    ok = [i for i in range(len(_O128_VARIANTS)) if ext.info(i) % 10000 == 0]
    print('O128_RESOURCES', [ext.info(i) for i in range(len(_O128_VARIANTS))], flush=True)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    flush = torch.ones(64 * 1024 * 1024, device=dev, dtype=torch.float32)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    choices = {}
    with torch.cuda.stream(stream):
        for layer in ('qkv_gate', 'pooler', 'unpooler'):
            w = weights[layer].contiguous()
            n, k = w.shape
            packs = []
            for i in ok:
                wk, split, _ = _O128_VARIANTS[i]
                pk = tt_mw_pack_weight(w, 64, wk, split, True)
                if pk is not None and _o128_exact(ext, i, pk, w, gen):
                    packs.append((i, pk))
                else:
                    print('O128_REJECT', layer, _O128_VARIANTS[i], flush=True)
            for m in (128, 256):
                if (m, layer) not in _O128_SHAPES or not packs or _o128_time.monotonic() - started > limit:
                    continue
                x = torch.randn((m, k), device=dev, generator=gen).to(torch.bfloat16)
                base = lambda: base_fn(x, layer)
                bt = _achita_wg_time(base, flush, stream, 17)
                best = None
                for i, pk in packs:
                    t = _achita_wg_time(lambda: _o128_launch(ext, i, pk, x), flush, stream, 17)
                    print('O128_CANDIDATE', m, layer, _O128_VARIANTS[i], t, 'baseline', bt, flush=True)
                    if best is None or t < best[0]:
                        best = (t, i, pk)
                bt = min(bt, _achita_wg_time(base, flush, stream, 17))
                if best is not None and best[0] < 0.97 * bt:
                    choices[m, layer, k] = (ext, best[1], best[2])
                    print('O128_SELECT', m, layer, _O128_VARIANTS[best[1]], best[0], 'baseline', bt, flush=True)
                else:
                    print('O128_KEEP', m, layer, 'baseline', bt, flush=True)
    torch.cuda.current_stream(dev).wait_stream(stream)
    torch.cuda.synchronize(dev)
    return choices


def _o128_run(choices, x, layer, n):
    ch = choices.get((x.shape[0], layer, x.shape[1])) if choices else None
    if ch is None or ch[2]['n'] != n or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    return _o128_launch(*ch, x)
