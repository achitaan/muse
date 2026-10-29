"""Full source-bound correctness, resources, actual execution and gain guard."""
from attention_two_score_evidence import source,PARENT,validate_packet,bound_result,full_result,verify_trial

raw,sha=source()
full_result(bound_result(sha))
if sha==PARENT:
    print('Cached exact9e full16/12/integrity; no GPU baseline job.')
else:
    assert raw==validate_packet()
    verify_trial(bound_result(sha))
    print('Two-score pipeline exact source/full16/12/integrity/native resources/execution/materiality passed.')
