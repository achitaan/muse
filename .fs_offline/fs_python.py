# FS1 appended to the exact moe_b13 source. Original builds and selector stay intact.
# As in CPP_SRC: host-only declarations; CUDA types stay in the CUDA TU.
FS_CPP = r'''
#include <torch/extension.h>
void fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid);
void fs_set_pdl(bool x);
void okc_set_pdl(int64_t x);
std::vector<int64_t> fs_info();
'''
_FS_MOD = None
_fs_parent_prepare = prepare
_fs_parent_kernel = kernel_fn


def _fs_ext():
    global _FS_MOD
    if _FS_MOD is None:
        import hashlib
        tag = hashlib.sha256(FS_CUDA.encode()).hexdigest()[:12]
        _FS_MOD = load_inline(name='okc_moe_fs1_'+tag, cpp_sources=FS_CPP, cuda_sources=FS_CUDA,
            functions=['fs_run', 'fs_set_pdl', 'okc_set_pdl', 'fs_info'],
            # Inherit Torch's standard (C++20 for Torch2.14 ATen), as the parent does.
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'],
            verbose=False)
        _FS_MOD.okc_set_pdl(int(PDL))
    return _FS_MOD


def _fs_pack_b(codes, scales):
    """Lossless K16 tasks: [k16, codes(rowblock,lane)32KiB + scales(row)4KiB]."""
    rows = torch.arange(4096, device=codes.device)
    words = _pack_codes_nat(codes, rows).view(256, 8, 8, 32, 4)
    code_bytes = words.permute(1, 2, 4, 0, 3).contiguous().view(torch.uint8).view(256, 32768)
    scale_bytes = scales.view(torch.uint8).T.contiguous()
    return torch.cat((code_bytes, scale_bytes), dim=1)


def _fs_state(weights, state):
    dev = state['w'][0].device
    assert torch.cuda.get_device_capability(dev) == (9, 0), 'Hopper required'
    assert state['cnt'] == 32 and state['grid'] >= 8, 'EP8 / resident grid required'
    assert state['sfx'] == ('-DOKC_PRMTSC=1',), 'positive normal E4M3 scales required'
    lo = state['lo']
    for key, shape, dtype in (('w13',(8192,2048),torch.uint8), ('w2',(4096,2048),torch.uint8),
                             ('w13_sf',(8192,256),torch.float8_e4m3fn), ('w2_sf',(4096,256),torch.float8_e4m3fn)):
        assert tuple(weights[key].shape[1:]) == shape and weights[key].dtype == dtype, key
        assert weights[key].shape[0] >= lo+32, key
    assert all(bool(torch.isfinite(weights[k][lo:lo+32]).all()) for k in ('w13_gs','w2_gs'))
    assert bool(torch.isfinite(state['w'][7]).all())
    packed = torch.empty((32,256,3,36864), dtype=torch.uint8, device=dev)
    packed[:,:,0:2].copy_(state['w'][10].view(32,256,2,36864))
    for e in range(32):
        packed[e,:,2].copy_(_fs_pack_b(weights['w2'][lo+e], weights['w2_sf'][lo+e]))
    assert packed.data_ptr() % 128 == 0
    return dict(packed=packed, aux=[state['w'][2],state['w'][5],state['w'][7]],
        scratch=[torch.zeros(3,dtype=torch.int32,device=dev),
                 torch.zeros((2,32,8,4096),dtype=torch.float32,device=dev)], ext=_fs_ext())


def _fs_run(data, state, fs):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].to(torch.float32).contiguous()
    assert x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == 4096
    assert 1 <= x.shape[0] <= 8 and ids.shape == tw.shape == (x.shape[0],8)
    assert x.device == ids.device == tw.device == fs['packed'].device
    assert data['config']['local_start'] == state['lo'] and data['config']['local_count'] == 32
    if x.data_ptr() % 16: x = x.clone()
    out = torch.empty_like(x)
    fs['ext'].fs_run(fs['packed'],fs['aux'],fs['scratch'],x,ids,tw,out,state['lo'],state['grid'])
    return out


def _fs_metrics(out, ref):
    """muse_moe_w4a16/{reference,checks}.py: tol=.015, row factor5, floor.1."""
    a,b = out.float(),ref.float()
    d = a-b
    rn = b.norm(dim=-1)
    row = (d.norm(dim=-1)/torch.maximum(rn, (.1*rn.mean()).clamp_min(1e-12))).max()
    rel = d.norm()/b.norm().clamp_min(1e-12)
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    ab = float(d.abs().max()); mr = float((d.abs()/b.abs().clamp_min(1e-12)).max())
    return dict(max_abs=ab,max_rel=mr,rel_L2=float(rel),row_L2=float(row),
                ok=finite and float(rel)<=.015 and float(row)<=.075)


def _fs_reference(data, weights, *, marlin=False):
    """Independent E2M1 decode and task reference; optional parent BF16 boundaries.

    Formula: muse_moe_w4a16/reference.py:93-107; E2M1 grid: nvfp4.py:24.
    No packed FS data is read. FP32 matmul (TF32 disabled by caller).
    """
    x = data['hidden_states'].float()
    ids,tw = data['topk_ids'],data['topk_weights']
    lo = data['config']['local_start']; hi = lo+data['config']['local_count']
    out = torch.zeros_like(x)
    grid = torch.tensor([0,.5,1,1.5,2,3,4,6],device=x.device)
    def dequant(q, sf):
        n = torch.stack((q&15,q>>4),-1).flatten(-2)
        w = grid[(n&7).long()] * torch.where((n&8)!=0,-1.,1.)
        return w * sf.float().repeat_interleave(16,dim=-1)
    for e in torch.unique(ids).tolist():
        if not lo<=e<hi: continue
        tok,slot = (ids==e).nonzero(as_tuple=True)
        wa = dequant(weights['w13'][e],weights['w13_sf'][e])
        ga = weights['w13_gs'][e]
        h = x[tok] @ (wa if marlin else wa*ga).T
        if marlin: h = (h.to(torch.bfloat16)*ga.to(torch.bfloat16)).float()
        inter = torch.nn.functional.silu(h[:,:4096])*h[:,4096:]
        if marlin: inter = inter.to(torch.bfloat16).float()
        del wa,h
        wb = dequant(weights['w2'][e],weights['w2_sf'][e])
        gb = weights['w2_gs'][e]
        y = inter @ (wb if marlin else wb*gb).T
        if marlin: y = (y.to(torch.bfloat16)*gb.to(torch.bfloat16)).float()
        nw = 1+weights['post_norm_w'].float()
        if marlin: nw = nw.to(torch.bfloat16).float()
        v = y*nw
        v *= torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-8)
        out.index_add_(0,tok,v*tw[tok,slot,None])
    return out.to(torch.bfloat16)


def _fs_probe(weights, state, fs):
    dev=fs['packed'].device
    gen=torch.Generator(device=dev).manual_seed(20261101)
    accepted=True
    saved_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        for M in (1,2,4,8):
            for draw in range(2):
                x=torch.randn((M,4096),generator=gen,device=dev).to(torch.bfloat16)
                ids,tw=_moe_routes(M,gen,dev)
                ids=(ids+state['lo'])%256
                if draw==1:
                    # One expert spans all tokens: mandatory upper4 hybrid path.
                    ids.fill_((state['lo']+32)%256);ids[:,0]=state['lo']+7
                    if 1<M<8: ids[-1,1]=state['lo']+7  # keep BASE entries<=8
                    x[:,::113]=(x[:,::113].float()*8).to(torch.bfloat16)
                data=dict(hidden_states=x,topk_ids=ids,topk_weights=tw,
                          config=dict(local_start=state['lo'],local_count=32))
                # Guarantee nonempty BASE input (its inherited zero-job path is unsafe).
                ids[0,0]=state['lo']+7
                base=_fs_parent_kernel(data,state)
                got=_fs_run(data,state,fs)
                for label,ref in [('BASE',base)]:
                    m=_fs_metrics(got,ref);accepted &= m['ok']
                    print(f'FS_PROBE M={M} draw={draw} ref={label} {m}',flush=True)
                if draw==1 and M in (1,8):
                    for marlin in (False,True):
                        ref=_fs_reference(data,weights,marlin=marlin)
                        m=_fs_metrics(got,ref);accepted &= m['ok']
                        print(f'FS_FORMULA M={M} marlin={int(marlin)} {m}',flush=True)
                    if M==8:
                        # Nine routes/eight unique tokens exceeds TINY2's entry
                        # capacity, so compare duplicate-route FS to the formula,
                        # NEVER launch BASE with that unsupported routing.
                        dup=dict(data,topk_ids=ids.clone())
                        dup['topk_ids'][-1,1]=state['lo']+7
                        ref=_fs_reference(dup,weights,marlin=True)
                        m=_fs_metrics(_fs_run(dup,state,fs),ref);accepted &= m['ok']
                        print(f'FS_DUPLICATE M=8 routes=9 {m}',flush=True)
                # Replay after a different routing, without host scratch resets.
                zero=dict(data,topk_ids=torch.full_like(ids,(state['lo']+32)%256))
                z=_fs_run(zero,state,fs)
                accepted &= bool((z==0).all())
                for _ in range(3):
                    replay=_fs_run(data,state,fs)
                    accepted &= _fs_metrics(replay,base)['ok']
        print(f'FS_REPLAY zero_and_reuse={int(accepted)}',flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32=saved_tf32
    return accepted


def _fs_error_lines(exc):
    """Bound compiler diagnostics, not the trailing ninja invocation."""
    lines = [s.strip() for s in str(exc).splitlines() if 'error' in s.lower()
             and not s.lstrip().lower().startswith('error building extension')]
    return ' | '.join(lines)[:600] or 'no compiler error lines captured'


def prepare(weights):
    state=_fs_parent_prepare(weights)
    state['fs']=None
    try:
        fs=_fs_state(weights,state)
        print(f'FS_BUILD resources={list(fs["ext"].fs_info())} packed_bytes={fs["packed"].numel()}',flush=True)
        if _fs_probe(weights,state,fs):
            state['fs']=fs
            print('FS_ENABLE M=1..8 four_slots hybrid_upper4',flush=True)
        else:
            print('FS_FALLBACK correctness_probe',flush=True)
    except Exception as exc:
        print(f'FS_FALLBACK {type(exc).__name__}: {_fs_error_lines(exc)}',flush=True)
    return state


def kernel_fn(data,state):
    fs=state.get('fs')
    if fs is not None and 1<=data['hidden_states'].shape[0]<=8:
        return _fs_run(data,state,fs)
    return _fs_parent_kernel(data,state)
