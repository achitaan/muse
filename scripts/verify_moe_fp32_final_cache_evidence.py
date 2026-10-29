"""Fixed full-suite, exact-source, actual-path and material-gain guard."""
from moe_fp32_final_cache_evidence import source, PARENT, validate_packet, bound_result, full_result, verify_trial


def main():
    raw, sha = source()
    result = bound_result(sha)
    full_result(result)
    if sha == PARENT:
        print('Exact cached MoE33114871: full5/6/integrity; no fresh baseline run.')
        return
    assert raw == validate_packet()
    verify_trial(result)
    print('FP32 final-cache source/full5/6/integrity/actual-path/material-gain guard passed.')


if __name__ == '__main__':
    main()
