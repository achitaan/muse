# L2 weight prefetch while attention/boundary leave HBM idle (omos/decode-fresh-20261029).
# A side-stream kernel issues prefetch.global.L2 for the buffers the next projections
# read (O, router, pooler head); cache hints only, retained arithmetic unchanged.
_PB = 40 << 20
_PR = None

@triton.jit
def _pfk(T, W: tl.constexpr):
 p = tl.program_id(0).to(tl.int64)
 s = tl.num_programs(0).to(tl.int64) * W
 e = tl.arange(0, W * 32)
 l = (e % 32).to(tl.int64)
 w = (e // 32).to(tl.int64)
 b0 = tl.load(T)
 b1 = tl.load(T + 1)
 b2 = tl.load(T + 2)
 b3 = tl.load(T + 3)
 c1 = tl.load(T + 4)
 c2 = tl.load(T + 5)
 c3 = tl.load(T + 6)
 n = tl.load(T + 7)
 for i in range(0, (n + s - 1) // s):
  c = tl.minimum(i * s + p * W + w, n - 1)
  a = tl.where(c < c1, b0 + c * 4096, tl.where(c < c2, b1 + (c - c1) * 4096, tl.where(c < c3, b2 + (c - c2) * 4096, b3 + (c - c3) * 4096)))
  tl.inline_asm_elementwise('prefetch.global.L2 [$1];prefetch.global.L2 [$1+1024];prefetch.global.L2 [$1+2048];prefetch.global.L2 [$1+3072];mov.u32 $0,0;', '=r,l', [a + l * 32], dtype=tl.int32, is_pure=False, pack=1)

def _pfn(t):
 if _PR is not None and isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous() and t.numel() * t.element_size() >= 1 << 20:
  _PR.append((t.data_ptr(), t.numel() * t.element_size()))

def _pwd(f):
 def g(*a, **k):
  if _PR is not None:
   for x in a:
    if isinstance(x, dict):
     _pfn(x.get('wp'))
     _pfn(x.get('w'))
  return f(*a, **k)
 return g

def _pwo(f):
 def g(x, st, i):
  if _PR is not None:
   q = st['configs'][i][4]
   _pfn(st.get('tiled32') if q == 2 else st.get('packed') if q == 1 else st.get('weight'))
  return f(x, st, i)
 return g

def _pw2(f):
 def g(x, w, *a, **k):
  _pfn(w)
  return f(x, w, *a, **k)
 return g

def _pwr(f):
 def g(x, st, a):
  if _PR is not None:
   _pfn(st.get('packed'))
  return f(x, st, a)
 return g
(sm_launch_raw, m32_launch_raw, cps32_launch_raw, cpssm_launch_raw, cps_launch, tt_mw_launch_raw, tt_v10_sm2_smallm2_launch_raw, tt_v10_warps_warps_launch_raw, tt_v10_m32x_m32x_launch_raw, tt_v10_m32x__w_launch_raw, tt_v10_m32x__q_launch_raw, tt_v12_smtrace1_smtrace_launch_raw, tt_v13_smtrace8_smtrace_launch_raw, dd_decodediet_dd_cap2_b_cpssm_launch_raw, dd_decodediet_cps_launch, dd_decodediet_tt_mw_launch_raw, mt__launch, b2p_tt_v10_m32x_m32x_launch_raw, b2p_tt_v10_m32x__w_launch_raw, b2p_tt_v10_m32x__q_launch_raw, v18s_launch_raw) = map(_pwd, (sm_launch_raw, m32_launch_raw, cps32_launch_raw, cpssm_launch_raw, cps_launch, tt_mw_launch_raw, tt_v10_sm2_smallm2_launch_raw, tt_v10_warps_warps_launch_raw, tt_v10_m32x_m32x_launch_raw, tt_v10_m32x__w_launch_raw, tt_v10_m32x__q_launch_raw, tt_v12_smtrace1_smtrace_launch_raw, tt_v13_smtrace8_smtrace_launch_raw, dd_decodediet_dd_cap2_b_cpssm_launch_raw, dd_decodediet_cps_launch, dd_decodediet_tt_mw_launch_raw, mt__launch, b2p_tt_v10_m32x_m32x_launch_raw, b2p_tt_v10_m32x__w_launch_raw, b2p_tt_v10_m32x__q_launch_raw, v18s_launch_raw))
(run_olayer_candidate, b2o_run_olayer_candidate) = map(_pwo, (run_olayer_candidate, b2o_run_olayer_candidate))
(tt_v10_tune_run_candidate, _gemm) = map(_pw2, (tt_v10_tune_run_candidate, _gemm))
rw_run_routerwide_candidate = _pwr(rw_run_routerwide_candidate)

def _pfrec(f, *a):
 global _PR
 _PR = []
 try:
  f(*a)
  torch.cuda.synchronize()
 finally:
  (r, _PR) = (_PR, None)
 return list(dict.fromkeys(r))

def _pfsp(t):
 return (t.data_ptr(), t.numel() * t.element_size()) if isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous() else None

def _pfplan(st, m, d):
 z = lambda k: torch.zeros((m, k), dtype=torch.bfloat16, device=d)
 s = _pfrec(_achita_dense_project, z(1024), 'o', st)
 p = []
 if 8 < m <= 32 and st.get('fpr') is not None:
  k = st['fpr']['sk2']
  r = [_pfsp(k['router'][1])]
  p = [_pfsp(k['pooler'][1])]
 elif m <= ROUTER_FP32_MAX_TOKENS:
  r = [_pfsp(st['router'])]
 else:
  r = _pfrec(_achita_wide_router, z(HIDDEN), st) or [_pfsp(st['router_hi_t']), _pfsp(st['router_lo_t'])]
 if not p:
  p = _pfrec(_achita_dense_project, z(HIDDEN), 'pooler', st)
 (g, u, left) = ([], set(), _PB)
 for x in s + r + p:
  if x is None or x[0] in u or x[0] % 16 or len(g) == 4:
   continue
  c = min(x[1], left) // 4096
  if c > 0:
   g.append((x[0], c))
   u.add(x[0])
   left -= c * 4096
 if not g:
  return None
 (b, c, t) = ([x for (x, _) in g], [], 0)
 for (_, k) in g:
  t += k
  c.append(t)
 b += [b[0]] * (4 - len(b))
 c += [t] * (4 - len(c))
 return (torch.tensor(b + c[:3] + [t], dtype=torch.int64, device=d), min(32, (t + 1) // 2), [k * 4096 for (_, k) in g])

def _pfl(v):
 _pfk[v[1],](v[0], W=2, num_warps=2)

def _pfprobe(st, P, d):
 f = torch.empty(256 << 20, dtype=torch.uint8, device=d)
 R = {}
 for m in (1, 8):
  for (nm, k) in (('o', 1024), ('pooler', HIDDEN)):
   x = torch.randn((m, k), device=d).to(torch.bfloat16)
   for w in (0, 1):
    t = []
    for _ in range(7):
     f.sum()
     if w:
      _pfl(P[m])
     torch.cuda.synchronize()
     (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
     e0.record()
     _achita_dense_project(x, nm, st)
     e1.record()
     e1.synchronize()
     t.append(e0.elapsed_time(e1) * 1000.0)
    R[f"m{m}_{nm}_{('pf' if w else 'cold')}"] = round(sorted(t)[3], 2)
 del f
 torch.cuda.synchronize()
 return R

def _pfprep(st):
 import time
 t0 = time.monotonic()
 d = st['router'].device
 P = {}
 for m in range(1, 257):
  try:
   v = _pfplan(st, m, d)
   if v:
    P[m] = v
  except Exception as x:
   print(f'[l2pf] m{m} skip {x!r}', flush=True)
 if not P:
  return None
 _pfl(P[min(P)])
 torch.cuda.synchronize()
 S = {m: [round(b / 2 ** 20, 1) for b in P[m][2]] for m in (1, 8, 16, 32, 128, 256) if m in P}
 print(f'[l2pf] plans={len(P)} MiB={S} s={time.monotonic() - t0:.2f}', flush=True)
 try:
  print(f'[l2pf] probe={_pfprobe(st, P, d)}', flush=True)
 except Exception as x:
  print(f'[l2pf] probe skip {x!r}', flush=True)
 return (P, torch.cuda.Stream(d))

def _pfbeg(state, cfg, tokens):
 pf = state.get('pf')
 v = pf[0].get(tokens) if pf and (cfg['window'] or cfg['batch'] == 1) else None
 if v is None:
  return None
 (main, side) = (torch.cuda.current_stream(), pf[1])
 side.wait_stream(main)
 with torch.cuda.stream(side):
  _pfl(v)
 return (main, side)

def kernel_fn(data, state):
 cfg = data['config']
 (batch, length, tokens) = (cfg['batch'], cfg['q_len'], cfg['tokens'])
 (hq, hkv, d) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'])
 qkv = _achita_dense_project(data['attn_input'], 'qkv_gate', state)
 pf = _pfbeg(state, cfg, tokens)
 attention_data = dict(data, qkv=qkv.reshape(batch, length, -1))
 o = _native_attn_kernel_fn(attention_data).reshape(tokens, hq * d)
 attn_output = _achita_dense_project(o, 'o', state)
 split_router = tokens > ROUTER_FP32_MAX_TOKENS
 (ffn_input, stream, router_input) = state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
 (main_stream, side_stream) = (torch.cuda.current_stream(), state['side_stream'])
 if 8 < tokens <= 32 and state['fpr'] is not None:
  (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
  (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
 else:
  side_stream.wait_stream(main_stream)
  expert_input = _achita_dense_project(ffn_input, 'pooler', state)
  with torch.cuda.stream(side_stream):
   if not split_router and tokens > 0 and (state['rf'] is not None):
    (topk_weights, topk_ids) = _rf_route(router_input, state['rf'])
   else:
    if split_router:
     logits = _achita_wide_router(ffn_input, state)
     if logits is None:
      logits = torch.mm(ffn_input, state['router_hi_t'], out_dtype=torch.float32)
      logits += torch.mm(ffn_input, state['router_lo_t'], out_dtype=torch.float32)
    else:
     logits = F.linear(router_input, state['router'])
    (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
  main_stream.wait_stream(side_stream)
  if not torch.cuda.is_current_stream_capturing():
   for tensor in (topk_weights, topk_ids):
    tensor.record_stream(main_stream)
 expert_input, local_ids = _pn_expert_inputs(expert_input, topk_ids, data['local_start'], state)
 expert_output = _experts(expert_input, topk_weights, local_ids, state)
 if tokens <= SLICED_UNPOOL_MAX_TOKENS:
  unpooled = _achita_dense_project(expert_output, 'unpooler', state, UNPOOL_SLICE)
  next_attn_input, stream = _sfn_boundary(stream, unpooled, state)
 else:
  ffn_output = _achita_dense_project(expert_output, 'unpooler', state)
  next_attn_input, stream, _ = state['ffn_boundary'](stream, ffn_output, branch_dtype=torch.bfloat16)
 if pf is not None:
  pf[0].wait_stream(pf[1])
 return (next_attn_input, stream, topk_ids)
_pfp0 = prepare

def prepare(weights):
 state = _pfp0(weights)
 state['pf'] = None
 if torch.cuda.get_device_capability()[0] == 9:
  try:
   state['pf'] = _pfprep(state)
  except Exception as x:
   print(f'[l2pf] off {x!r}', flush=True)
 return state
