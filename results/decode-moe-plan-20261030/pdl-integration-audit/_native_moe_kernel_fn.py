def _native_moe_kernel_fn(data, state):
 cfg = data['config']
 x = data['hidden_states'].contiguous()
 M = x.shape[0]
 assert cfg['local_start'] == state['lo'] and cfg['local_count'] == state['cnt']
 ids = data['topk_ids'].to(torch.int32).contiguous()
 if ids.data_ptr() & 15:
  ids = ids.clone()
 tw = data['topk_weights'].to(torch.float32).contiguous()
 out = torch.empty((M, _native_moe_HIDDEN), dtype=torch.bfloat16, device=x.device)
 sfx = state.get('sfx', ())
 if _native_moe_TINY_DEFINES and M <= _native_moe_TINY_MAX_M:
  ext = _native_moe__ext(tuple(_native_moe_TINY_DEFINES) + sfx)
 elif _native_moe_TINY2_DEFINES and M <= _native_moe_TINY2_MAX_M and (sfx or '-DOKC_WGS=1' not in tuple(_native_moe_TINY2_DEFINES)):
  ext = _native_moe__ext(tuple(_native_moe_TINY2_DEFINES) + sfx)
 elif _native_moe_MID_DEFINES and M <= _native_moe_MID_MAX_M:
  ext = _native_moe__ext(tuple(_native_moe_MID_DEFINES) + sfx)
 else:
  ext = _native_moe__ext(tuple(_native_moe__main_defines(sfx)) + sfx)
 for m0 in range(0, M, _native_moe_CHUNK_TOKENS):
  m1 = min(M, m0 + _native_moe_CHUNK_TOKENS)
  ext.okc_run(state['w'], state['scratch'], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1], state['lo'], state['cnt'], state['grid'])
 return out
