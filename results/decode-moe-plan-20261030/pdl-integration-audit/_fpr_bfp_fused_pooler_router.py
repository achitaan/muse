def _fpr_bfp_fused_pooler_router(x, state):
 m = x.shape[0]
 assert m <= 32 and state.get('sm90') and ('pooler' in state['sk2']) and ('router' in state['sk2'])
 (ci, stream, tab, pmax, b4, n_rows) = state['sk2']['pooler']
 rstream = state['sk2']['router'][1]
 if 'fused_part' not in state:
  state['fused_part'] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
  state['fused_cnt'] = torch.zeros(32, dtype=torch.int32, device=x.device)
 out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
 outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
 ext = _fpr_get_ext()
 ext.sk2_set_pdl(_fpr_PDL)
 ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state['fused_part'], state['fused_cnt'], m, _fpr_bfp__sk2_tokr(m))
 return (out, outr)
