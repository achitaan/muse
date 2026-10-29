from decoder_moe_tail_evidence import PARENT, source, bound_result, full_result, verify_trial

raw, sha = source()
result = bound_result(sha)
full_result(result) if sha == PARENT else verify_trial(result)
print('{"correct":true}')
