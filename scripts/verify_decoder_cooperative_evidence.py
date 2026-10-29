"""Controller guard: exact baseline or fully admitted candidate evidence."""
from decoder_cooperative_evidence import source,PARENT,validate_packet,bound_result,full_result,verify_trial

raw,sha=source()
full_result(bound_result(sha))
if sha==PARENT:
    print('Cached exact D9 full13/12/integrity; no new GPU baseline job.')
else:
    assert raw==validate_packet()
    verify_trial(bound_result(sha))
    print('Exact cooperative Decode source/full13/12/integrity/resources/execution/materiality passed.')
