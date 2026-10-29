"""Source-bound complete-layer correctness, execution and material gain gate."""
from decoder_cluster_norm_dual_order_evidence import source,PARENT,bound_result,full_result,validate_packet,verify_trial

raw,sha=source()
full_result(bound_result(sha))
if sha==PARENT:print('Cached exactD9 full13/12/integrity; no baseline GPU job.')
else:
    assert raw==validate_packet()
    verify_trial(bound_result(sha))
    print('Cluster norm complete-layer correctness/native execution/weighted materiality passed.')
