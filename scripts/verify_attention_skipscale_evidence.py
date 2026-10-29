"""Fixed full-suite and source-bound exact identity-rescale guard."""
from attention_skipscale_evidence import source,PARENT,validate_packet,bound_result,full_result,verify_trial

raw,sha = source()
full_result(bound_result(sha))
if sha == PARENT:
    print('Cached exact Attention9e922c35 full16/12/integrity; no fresh baseline job.')
else:
    assert raw == validate_packet()
    verify_trial(bound_result(sha))
    print('Exact source/full16/12/integrity/resources/actual path/material gain passed.')
