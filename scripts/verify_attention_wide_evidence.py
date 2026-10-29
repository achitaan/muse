"""Source-bound wide-tile full-suite and causal performance guard."""
from attention_wide_evidence import source,PARENT,validate_packet,bound_result,full_result,verify_trial

raw,sha=source()
full_result(bound_result(sha))
if sha==PARENT:
    print('Cached exact Attention9e full16/12/integrity; no GPU baseline job.')
else:
    assert raw==validate_packet()
    verify_trial(bound_result(sha))
    print('Wide-tile exact source/full16/12/integrity/resources/path/materiality passed.')
