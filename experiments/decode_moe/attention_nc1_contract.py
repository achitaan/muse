"""Metadata-only eligibility for two screened Attention entry points."""


def family(B, L, P, window, max_pages, n_splits, flat, nc, cs, flags):
    # Flags0 excludes cooperative and PDL launches, preserving the parent's
    # effective cooperative decision independently of new register counts.
    if (L != 1 or nc != 1 or flags != 0 or B < 1 or P < 1 or max_pages < 1
            or n_splits < 1 or cs > 1 or window < 0):
        return None
    if flat:
        return 'flat_L1'
    if B != 1 or P != 1:
        return None
    kmax = min(window + L - 1, max_pages * P) if window else max_pages * P
    if n_splits >= (kmax + 63) // 64:
        return 'spec_one_L1'
    return None


def run_family(args):
    """Use the actual 28-argument native call after plans/tuning are applied."""
    if len(args) != 28:
        return None
    return family(args[13], args[14], args[17], args[18], args[5].shape[1],
                  args[20], args[21], args[22], args[25], args[27])
