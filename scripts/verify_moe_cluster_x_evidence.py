from moe_cluster_x_evidence import source, PARENT, bound_result, full_result, validate_packet, verify_trial


def main():
    raw,sha=source()
    result=bound_result(sha)
    full_result(result)
    if sha==PARENT:
        print('Exact cached33114871 full5/6/integrity; no baseline GPU rerun.')
        return
    assert raw==validate_packet()
    verify_trial(result)
    print('Cluster2 full-suite/source/actual-two-load-path/material-gain guard passed.')


if __name__=='__main__':
    main()
