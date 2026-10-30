def _native_attn__launch(data, dbg=None, _mod=None, pdl=None, **override):
 cfg = data['config']
 (B, L) = (cfg['batch'], cfg['q_len'])
 (hq, hkv, d, P) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'], cfg['page_size'])
 qkv = data['qkv']
 device = qkv.device
 pairs = hkv // 2
 window = cfg['window']
 page_table = data['page_table']
 out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
 plan = _native_attn__plan(B, pairs, L, window, page_table.shape[1], P, device)
 tuned = _native_attn__TUNED.get((B, L, bool(window), pairs))
 if tuned and (not override):
  override = tuned
 if override:
  plan = dict(plan, **override)
  if 'cs' in override and override['cs'] > 1:
   plan['flat'] = 0
   if 'ncl' not in override:
    plan['ncl'] = max(1, plan['n_splits'] // override['cs'])
   plan['n_splits'] = plan['cs'] * plan['ncl']
   if 'nc' not in override:
    plan['nc'] = _native_attn__copies(plan['ncl'], 4)
  elif 'n_splits' in override and 'nc' not in override:
   plan['nc'] = 1 if plan['flat'] else _native_attn__copies(plan['n_splits'], 4)
 (n_splits, flat, cs, ncl, nc) = (plan['n_splits'], plan['flat'], plan['cs'], plan['ncl'], plan['nc'])
 if flat:
  coop = int(plan.get('coop', int(L == 8 and n_splits * pairs <= 2 * _native_attn__num_sms(device))))
 else:
  coop = int(plan.get('coop', 0 if cs > 1 or n_splits < 64 else int(n_splits * B * pairs <= 2 * _native_attn__num_sms(device))))
 use_pdl = bool(pdl)
 tma = int(plan.get('tma', 0)) | coop << _native_attn_COOP_BIT | int(use_pdl) << _native_attn_PDL_BIT
 if flat or cs <= 1:
  (cs, ncl) = (0, 1)
 (ws_acc, ws_l, ws_cnt) = _native_attn__workspace(B * pairs * nc * L * _native_attn_Q_HEADS_PER_PAIR, device, (B, pairs, L, window, P, n_splits, flat, cs, nc, coop))
 scale_log2 = cfg['sm_scale'] * cfg['qk_scale_factor'] * _native_attn_LOG2E
 (_mod or _native_attn__module()).run(qkv, data['seq_lens'], data['cache_loc'], data['k_cache'], data['v_cache'], page_table, data['k_scale'], data['v_scale'], _native_attn__inv_freq(cfg['rope_theta'], device), out, ws_acc, ws_l, ws_cnt, B, L, hq, hkv, P, window, 1 if cfg['rope_theta'] else 0, n_splits, flat, nc, scale_log2, dbg, cs, ncl, tma)
 return out
