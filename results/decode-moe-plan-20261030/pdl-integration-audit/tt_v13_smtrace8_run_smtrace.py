def tt_v13_smtrace8_run_smtrace(x, layer, n, st):
 if (layer, x.shape[0]) not in tt_v13_smtrace8_SMTRACE_ROUTES:
  return None
 return tt_v13_smtrace8_smtrace_run_core(x, layer, n, st)
