def kernel_fn(data, state):
 cfg = data['config']
 (batch, length, tokens) = (cfg['batch'], cfg['q_len'], cfg['tokens'])
 (hq, hkv, d) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'])
 dense = state['dense']
 qkv = _achita_dense_project(data['attn_input'], 'qkv_gate', state)
 attention_data = dict(data, qkv=qkv.reshape(batch, length, -1))
 o = _native_attn_kernel_fn(attention_data).reshape(tokens, hq * d)
 attn_output = _achita_dense_project(o, 'o', state)
 split_router = tokens > ROUTER_FP32_MAX_TOKENS
 (ffn_input, stream, router_input) = state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
 (main_stream, side_stream) = (torch.cuda.current_stream(), state['side_stream'])
 if 8 < tokens <= 32 and state['fpr'] is not None:
  (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
  (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
 elif not split_router and tokens in state.get('r2sel', ()) and (r2 := _r2_pool_route(ffn_input, router_input, state)) is not None:
  (expert_input, topk_weights, topk_ids) = r2
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
 return (next_attn_input, stream, topk_ids)
