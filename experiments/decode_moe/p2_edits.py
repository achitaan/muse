"""Exact kernel_fn edits for P2 (indentation matches the parent's one-space style)."""

PY_EDITS = [
    (""" (main_stream, side_stream) = (torch.cuda.current_stream(), state['side_stream'])
""", """ (main_stream, side_stream) = (torch.cuda.current_stream(), state['side_stream'])
 (p2, p2_logits) = (None, None)
"""),
    ("""  (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
  (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
""", """  (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
  p2 = _p2_route_pn(logits, expert_input, data['local_start'], state)
  if p2 is None:
   (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
"""),
    ("""     logits = F.linear(router_input, state['router'])
    (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
  main_stream.wait_stream(side_stream)
  if not torch.cuda.is_current_stream_capturing():
   for tensor in (topk_weights, topk_ids):
    tensor.record_stream(main_stream)
 expert_input, local_ids = _pn_expert_inputs(expert_input, topk_ids, data['local_start'], state)
""", """     logits = F.linear(router_input, state['router'])
    if split_router and state.get('p2'):
     p2_logits = logits
    else:
     (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
  main_stream.wait_stream(side_stream)
  if not torch.cuda.is_current_stream_capturing():
   for tensor in (p2_logits,) if p2_logits is not None else (topk_weights, topk_ids):
    tensor.record_stream(main_stream)
  if p2_logits is not None:
   p2 = _p2_route_pn(p2_logits, expert_input, data['local_start'], state)
   if p2 is None:
    (topk_weights, topk_ids) = state['routing'](None, p2_logits, TOP_K, True)
 if p2 is not None:
  (topk_weights, topk_ids, expert_input, local_ids) = p2
 else:
  (expert_input, local_ids) = _pn_expert_inputs(expert_input, topk_ids, data['local_start'], state)
"""),
]
