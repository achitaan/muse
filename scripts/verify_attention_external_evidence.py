"""Exact-source full16/12/integrity and causal-gain controller guard."""
from attention_external_evidence import source,PARENT,validate_packet,bound_result,full_result,verify_trial

raw,sha=source()
full_result(bound_result(sha))
if sha==PARENT:
    print('Cached exact9e full16/12/integrity; no GPU baseline job.')
else:
    assert raw==validate_packet()
    verify_trial(bound_result(sha))
    print('External-finalizer exact source/full16/12/integrity/native resources/execution/materiality passed.')
