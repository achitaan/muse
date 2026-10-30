def tt_v12_smtrace1_run_smtrace(x, layer, n, st):
 if (layer, x.shape[0]) not in tt_v12_smtrace1_SMTRACE_ROUTES:
  return None
 return tt_v12_smtrace1_smtrace_run_core(x, layer, n, st)
