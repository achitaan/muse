"""MoE small-batch phase diagnostic (timing only): instrumented TINY / TINY2 / MID builds at M = 1, 8, 16.

Parent: board_best/moe_m3_scored_2.3797.py (MoE board best, sha256 23008c1b...). Output moe_tail_diag.py.
The timed path is the parent's unchanged. prepare() additionally compiles the three small-batch builds with
OKC_DEBUG = 1 (the kernel's existing per-CTA globaltimer stamps: 0 entry, 1 after routing, 2 after phase A,
3 after phase B, 4 after the grid barrier, 5 exit), runs them on skewed synthetic routings (the parent's own
_moe_routes) and prints per-phase intervals, the tail after the latest phase-B CTA, and paired graph times of
the original and instrumented launches.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

APPEND = r'''

# ---- MoE small-batch phase diagnostic (timing only; the timed path above is unchanged)
import threading as _tg_thr
import time as _tg_time
import statistics as _tg_stats
_TG_EXT = {}
_TG_SFX = ("-DOKC_PRMTSC=1",)


def _tg_build(name, defines):
    try:
        defines = tuple(defines)
        if DQ7 and "-DOKC_DQ7=1" not in defines:
            defines = defines + ("-DOKC_DQ7=1",)
        src = CUDA_SRC.replace("#define OKC_DEBUG 0", "#define OKC_DEBUG 1", 1)
        assert src != CUDA_SRC
        tag = hashlib.sha256((src + repr(defines)).encode()).hexdigest()[:12]
        ns = [] if any(d.startswith("-DOKC_NS=") for d in defines) else [f"-DOKC_NS={MAIN_NS}"]
        nsf = [] if any(d.startswith("-DOKC_NSF=") for d in defines) else ["-DOKC_NSF=3"]
        _TG_EXT[name] = load_inline(
            name=f"okc_moe_taildiag_{name.lower()}_{tag}", cpp_sources=CPP_SRC, cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", *ns, *nsf, *defines],
            verbose=False)
        _TG_EXT[name].okc_set_pdl(1 if PDL else 0)
    except Exception as e:
        print("MOE_TAIL_BUILD_ERROR", name, repr(e)[-1500:], flush=True)


import hashlib
_tg_threads = [_tg_thr.Thread(target=_tg_build, args=(n, tuple(d) + _TG_SFX), daemon=True)
               for n, d in (("TINY", TINY_DEFINES), ("TINY2", TINY2_DEFINES), ("MID", MID_DEFINES)) if d]
for _t in _tg_threads:
    _t.start()


def _tg_case(M, seed, name, state):
    dev = state["w"][0].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((M, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
    ids, tw = _moe_routes(M, gen, dev)
    local = (ids >= state["lo"]) & (ids < state["lo"] + state["cnt"])
    counts = torch.bincount((ids[local] - state["lo"]).long(), minlength=state["cnt"]).cpu().tolist()
    data = dict(hidden_states=x, topk_ids=ids, topk_weights=tw, config=dict(local_start=state["lo"], local_count=state["cnt"]))
    ext = _TG_EXT[name]
    grid = state["grid"]
    dbg = torch.zeros((grid, 8), dtype=torch.int64, device=dev)
    scratch = _make_scratch(dev) + [dbg]
    out_i = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=dev)
    outs = [None]

    def orig():
        outs[0] = kernel_fn(data, state)

    def inst():
        ext.okc_run(state["w"], scratch, x, ids, tw, out_i, state["lo"], state["cnt"], grid)

    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    graphs = []
    for fn in (orig, inst):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graphs.append(g)
    ts = ([], [])
    for r in range(12):
        for k in ((0, 1) if r % 2 == 0 else (1, 0)):
            flush.sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[k].replay()
            e1.record()
            e1.synchronize()
            ts[k].append(e0.elapsed_time(e1) * 1000.0)
    ref, got = outs[0].float(), out_i.float()
    err = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
    tr = dbg.cpu().tolist()
    del graphs, flush
    rows = [r for r in tr if r[0] > 0 and r[5] >= r[0]]
    base = min(r[0] for r in rows)
    iv = {"route": (0, 1), "A": (1, 2), "B": (2, 3), "join": (3, 4), "fin": (4, 5)}
    s = {k: [(r[b] - r[a]) / 1000.0 for r in rows] for k, (a, b) in iv.items()}
    q = lambda v: "%.1f/%.1f/%.1f" % (min(v), _tg_stats.median(v), max(v))
    ev = lambda c: [(r[c] - base) / 1000.0 for r in rows]
    span = max(r[5] for r in rows) - base
    print(f"MOE_TAIL M={M} seed={seed} build={name} experts={sum(c > 0 for c in counts)} routes={sum(counts)} "
          f"maxtok={max(counts)} orig={_tg_stats.median(ts[0]):.2f}us inst={_tg_stats.median(ts[1]):.2f}us err={err:.1e} "
          f"span={span / 1000.0:.1f} " + " ".join(f"{k}={q(v)}" for k, v in s.items())
          + f" startspread={max(ev(0)) - min(ev(0)):.1f} Aend={q(ev(2))} Bend={q(ev(3))} join_rel={q(ev(4))}"
          f" tailB={(max(r[5] for r in rows) - max(r[3] for r in rows)) / 1000.0:.1f} ctas={len(rows)}", flush=True)


_tg_prev_prepare = prepare


def prepare(weights):
    t0 = _tg_time.monotonic()
    state = _tg_prev_prepare(weights)
    for t in _tg_threads:
        t.join(max(0.0, 270.0 - (_tg_time.monotonic() - t0)))
    if tuple(state.get("sfx", ())) != _TG_SFX:
        print("MOE_TAIL_SKIP sfx", state.get("sfx"), flush=True)
        return state
    for M, name in ((1, "TINY"), (8, "TINY2"), (16, "MID")):
        for seed in (20261101 + M, 20261201 + M):
            if _tg_time.monotonic() - t0 > 285.0 or name not in _TG_EXT:
                print("MOE_TAIL_SKIP", M, seed, name, f"{_tg_time.monotonic() - t0:.1f}s", flush=True)
                continue
            try:
                _tg_case(M, seed, name, state)
            except Exception as e:
                print("MOE_TAIL_ERROR", M, seed, repr(e)[:300], flush=True)
    torch.cuda.synchronize()
    print(f"MOE_TAIL_DONE {_tg_time.monotonic() - t0:.1f}s", flush=True)
    return state
'''


def main():
    import subprocess
    raw = subprocess.run(['git', 'cat-file', 'blob', 'HEAD:board_best/moe_m3_scored_2.3797.py'], cwd=ROOT,
                         check=True, capture_output=True).stdout
    src = raw.decode('utf-8')
    assert hashlib.sha256(src.encode()).hexdigest().startswith('23008c1b')
    out = src.rstrip('\n') + '\n' + APPEND
    ast.parse(out)
    b = out.encode('utf-8')
    (HERE / 'moe_tail_diag.py').write_bytes(b)
    print(json.dumps({'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}))


if __name__ == '__main__':
    main()
