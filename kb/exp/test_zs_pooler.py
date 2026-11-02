"""achitaan: CPU source/layout/bit-decoder tests; no torch or GPU required."""
import ast
from pathlib import Path
import numpy as np
from zs_codec import pack_fragments, self_test

HERE = Path(__file__).resolve().parent


def load_helpers():
    base = ast.parse((HERE / "bf16_v2.py").read_text(encoding="utf-8"))
    factories = {n.name: n for n in base.body if isinstance(n, ast.FunctionDef)}
    def constant(factory, name):
        for n in factories[factory].body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                return ast.literal_eval(n.value)
        raise AssertionError(name)
    ns = {"np": np, "_K": {"chenwainuo": {"zs_gv_source": constant("_make_chenwainuo", "GVCU")},
                           "vjeux": {"zs_cg_source": constant("_make_vjeux", "RGEMV_SRC")}}}
    code = ast.parse((HERE / "zs_pooler.py").read_text(encoding="utf-8"))
    selected = [n for n in code.body if
                isinstance(n, ast.FunctionDef) and n.name in ("_zs_replace", "_zs_cuda_source", "_zs_fragments")
                or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ZS_DECODE" for t in n.targets)]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<ZS helpers>", "exec"), ns)
    return ns


def perm(a, b, selectors):
    ab = np.asarray(a, dtype=np.uint64) | (np.asarray(b, dtype=np.uint64) << np.uint64(32))
    selectors = np.asarray(selectors, dtype=np.uint64)
    out = np.zeros(np.broadcast_shapes(ab.shape, selectors.shape), dtype=np.uint32)
    for i in range(4):
        ix = (selectors >> np.uint64(4*i)) & np.uint64(7)
        out |= (((ab >> (np.uint64(8)*ix)) & np.uint64(255)) << np.uint64(8*i)).astype(np.uint32)
    return out


def gpu_decode_model(pk):
    # Independent emulation of fixed nibble positions / PRMT / LOP3 / scan.
    # Expected values in tests are the original BF16 integers, not this model.
    p = pk["planes"]
    u = len(p)
    ms = pk["sign_mantissa"].view(np.uint32).reshape(u, 2, 32, 4).transpose(0, 2, 1, 3).reshape(u, 32, 8)
    mask = ~(p[:, 0] | p[:, 1] | p[:, 2])
    cnt = np.unpackbits(mask.copy().view(np.uint8).reshape(u, 32, 4), axis=-1).sum(axis=-1, dtype=np.uint32)
    scan = cnt.copy()
    for delta in (1, 2, 4, 8, 16):
        old = scan.copy()
        scan[:, delta:] += old[:, :-delta]
    base = pk["offsets"][:-1, None] + scan - cnt
    words = np.empty((u, 32, 16), dtype=np.uint32)
    t0 = int.from_bytes(pk["table"][:4].tobytes(), "little")
    t1 = int.from_bytes(pk["table"][4:].tobytes(), "little")
    for pos in range(0, 32, 4):
        sel = np.zeros((u, 32), dtype=np.uint32)
        shift = (pos//16)*16 + (pos % 16)//4
        for bit in range(3):
            v = p[:, bit] >> (shift-bit) if shift >= bit else p[:, bit] << (bit-shift)
            sel |= v & (0x1111 << bit)
        e = perm(t0, t1, sel)
        for pair, selector in enumerate((0x1100, 0x3322)):
            a = ((perm(e, 0, selector) << 7) & 0x7f807f80) | (perm(ms[:, :, pos//4], 0, selector) & 0x807f807f)
            for half in (0, 1):
                at = pos + 2*pair + half
                at = (at//16)*16 + (at % 16)//4 + (at % 4)*4
                need = (mask & np.uint32(1 << at)) != 0
                prefix_mask = mask & np.uint32((1 << at)-1)
                rank = np.unpackbits(prefix_mask.copy().view(np.uint8).reshape(u, 32, 4), axis=-1).sum(axis=-1, dtype=np.uint32)
                addr = base + rank
                replacement = pk["escapes"][addr[need]].astype(np.uint32)
                keep_mask = np.uint32(0xffff0000 if half == 0 else 0xffff)
                a[need] = (a[need] & keep_mask) | (replacement << (16*half))
            words[:, :, pos//2+pair] = a
    values = words.reshape(u, 32, 16, 1).view(np.uint16).reshape(u, 32, 32)
    return values.reshape(u, 32, 4, 8).transpose(0, 2, 1, 3).reshape(pk["shape"])


def test_source(ns):
    src = ns["_zs_cuda_source"]()
    assert "@@CASES@@" not in src
    assert "case 0: return gvl<6,1,1,2,16,8,1,1,0,3>" in src
    assert "case 1: return gvl<0,1,1,2,16,8,1,1,0,3>" in src
    assert "cgemv_kernel<2,512,8192,0><<<2048,544" in src
    assert "cgemv_kernel<2,512,8192,6><<<2048,544" in src
    # Unchanged arithmetic text is the source of the raw-twin invariant.
    original_cg = ns["_K"]["vjeux"]["zs_cg_source"]
    old = original_cg[original_cg.index("        const float* xg = xf + 8 * g;"):]
    old = old[:old.index("\nvoid cgemv(")]
    assert old in src
    original_gv = ns["_K"]["chenwainuo"]["zs_gv_source"]
    old = original_gv[original_gv.index("__global__ void __launch_bounds__", original_gv.index("// Predicated format1 repair")):]
    old = old[old.index("    if constexpr (((V >> 25) & 1)"):]
    old = old[:old.index("\n__global__ void p1zero")]
    assert old in src
    # No other family or shape was added to the independently built extension.
    assert src.count("case 0: return gvl<") == 1 and src.count("case 1: return gvl<") == 1
    original = ast.parse((HERE / "bf16_v2.py").read_text(encoding="utf-8"))
    built = ast.parse((HERE / "bf16_zs.py").read_text(encoding="utf-8"))
    for old_node, new_node in zip(original.body, built.body):
        if isinstance(old_node, ast.FunctionDef) and old_node.name in ("_make_chenwainuo", "_make_vjeux"):
            # These only export the existing CUDA strings for independent builds.
            assert ast.dump(ast.Module(body=old_node.body[:-1], type_ignores=[])) == ast.dump(ast.Module(body=new_node.body[:-1], type_ignores=[]))
        else:
            assert ast.dump(old_node) == ast.dump(new_node)
    print("CUDA source construction PASS: exact two geometries and preserved math/reduction text")


def test_fragment_layouts(ns):
    # Independent dense row/K indexing reference: reproducible arbitrary bits.
    bits = np.random.default_rng(0xAC41725).integers(0, 65536, size=(4096, 8192), dtype=np.uint16)
    cg = ns["_zs_fragments"](bits, "cg")
    groups = cg.reshape(-1, 4, 32, 8).transpose(0, 2, 1, 3).reshape(-1, 32, 32)
    restored = groups.reshape(2048, 16, 32, 2, 16).transpose(0, 3, 1, 2, 4).reshape(4096, 8192)
    assert np.array_equal(restored, bits)
    del cg, groups, restored
    gv = ns["_zs_fragments"](bits, "gv")
    tile = np.arange(256)[:, None, None, None]
    chunk = np.arange(128)[None, :, None, None]
    step = np.arange(4)[None, None, :, None]
    lane = np.arange(32)[None, None, None, :]
    for v in range(8):
        row = 16*tile + lane//4 + 8*((v//2) % 2)
        col = 64*chunk + 16*(lane % 4) + 4*step + v % 2 + 2*(v//4)
        assert np.array_equal(gv[..., v], bits[row, col])
    print("Both full pooler fragment layouts PASS against independent row/K indices")


def main():
    ns = load_helpers()
    test_source(ns)
    self_test()
    values = np.arange(65536, dtype=np.uint16).reshape(8, 8, 4, 32, 8)
    assert np.array_equal(gpu_decode_model(pack_fragments(values, permuted=True)), values)
    rng = np.random.default_rng(20261102)
    for shape in ((1, 1, 4, 32, 8), (3, 7, 4, 32, 8)):
        values = rng.integers(0, 65536, size=shape, dtype=np.uint16)
        assert np.array_equal(gpu_decode_model(pack_fragments(values, permuted=True)), values)
    # No escapes: scan zero counts and no empty-array dereference.
    values = np.full((1, 1, 4, 32, 8), 0xbf80, dtype=np.uint16)
    assert np.array_equal(gpu_decode_model(pack_fragments(values, permuted=True)), values)
    print("PRMT/LOP3/warp-scan/popcount decoder model PASS: exhaustive encodings and random/boundary patterns")
    test_fragment_layouts(ns)


if __name__ == "__main__":
    main()
