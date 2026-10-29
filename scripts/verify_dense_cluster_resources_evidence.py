"""Verify complete coverage, exact source, real native use and material gain."""
from dense_cluster_resources_evidence import source,PARENT,bound_result,full_result,validate_packet,verify_trial

raw,sha=source();full_result(bound_result(sha))
if sha==PARENT:print('Cached exact EDC complete9/30/integrity and weighted loads; no GPU baseline.')
else:
    assert raw==validate_packet()
    verify_trial(bound_result(sha))
    print('Dense resource-bound correctness/native use/weighted materiality passed.')
