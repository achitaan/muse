import numpy as np

rng = np.random.default_rng(0)
for (n, k) in ((2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096)):
    w = (rng.standard_normal((256, k), dtype=np.float32) / np.sqrt(k)).astype(np.float32)
    bits = (w.view(np.uint32) >> 16).astype(np.uint32)
    e = (bits >> 7) & 0xFF
    hi7 = e >> 1
    vals, cnt = np.unique(hi7, return_counts=True)
    frac = cnt / cnt.sum()
    order = np.argsort(vals)
    vals, frac = vals[order], frac[order]
    def best(win):
        b = 0
        for i in range(len(vals)):
            s = frac[(vals >= vals[i]) & (vals < vals[i] + win)].sum()
            b = max(b, s)
        return b
    ent = -(frac * np.log2(frac)).sum()
    print(f'k={k}: hi7 entropy {ent:.2f} bits; esc rate win8 {1 - best(8):.5f} win4 {1 - best(4):.5f} win2 {1 - best(2):.4f}; '
          f'per 8192-weight chunk: win8 {8192 * (1 - best(8)):.1f}, win4 {8192 * (1 - best(4)):.1f}')
