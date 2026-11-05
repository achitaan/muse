"""Reproduce the single-file decode VMS32 candidate from pinned local git sources."""
import ast
import io
from pathlib import Path
import subprocess
import tokenize

ROOT = Path(__file__).resolve().parents[2]


def source(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT).decode()


def definitions(text, names):
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    return "\n\n".join("".join(lines[n.lineno - 1:n.end_lineno]) for n in tree.body
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))
                         and ((isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
                              or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)))


def compact(text):
    # Leave embedded CUDA sources, including strings used by replace(), byte-identical.
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                for i in range(first.lineno - 1, first.end_lineno):
                    lines[i] = "\n"
                n.body = n.body[1:]
    tokens = [t for t in tokenize.generate_tokens(io.StringIO("".join(lines)).readline) if t.type != tokenize.COMMENT]
    uncommented = tokenize.untokenize(tokens)
    protected = set()
    for token in tokenize.generate_tokens(io.StringIO(uncommented).readline):
        if token.type == tokenize.STRING and token.start[0] != token.end[0]:
            protected.update(range(token.start[0], token.end[0] + 1))
    compacted = "\n".join(line if i in protected else line.rstrip()
                          for i, line in enumerate(uncommented.splitlines(), 1)
                          if line.strip() or i in protected) + "\n"
    assert ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(compacted), include_attributes=False), "compaction changed semantics"
    return compacted


def flags(text):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets, at = [], 0
    for line in lines:
        offsets.append(at)
        at += len(line)
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in ("load_inline", "_tvm_load_inline"):
            continue
        seen = set()
        for kw in node.keywords:
            if kw.arg not in ("extra_cflags", "extra_cuda_cflags"):
                continue
            seen.add(kw.arg)
            value = kw.value
            a, b = offsets[value.lineno - 1] + value.col_offset, offsets[value.end_lineno - 1] + value.end_col_offset
            wrapper = "_dv_host_flags" if kw.arg == "extra_cflags" else "_dv_nvcc_flags"
            edits.append((a, b, wrapper + "(" + text[a:b] + ")"))
        missing = []
        if "extra_cflags" not in seen:
            missing.append("extra_cflags=_dv_host_flags(['-O3'])")
        if "extra_cuda_cflags" not in seen:
            missing.append("extra_cuda_cflags=_dv_nvcc_flags(['-O3'])")
        if missing:
            end = offsets[node.end_lineno - 1] + node.end_col_offset - 1
            # Existing multiline calls may already have a trailing comma.
            comma = "" if text[:end].rstrip().endswith(",") else ","
            edits.append((end, end, comma + " " + ", ".join(missing)))
    for a, b, replacement in sorted(edits, reverse=True):
        text = text[:a] + replacement + text[b:]
    return text


if __name__ == "__main__":
    base = source("f50a51d7fcca2ba09e5c35de412de5b38538cdc2", "kb/v1/decode_v1.py")
    moe = source("8736a4b1d325ef65e6cd53a3e4605f68d332b7fd", "candidates/moe_vms32.py")
    old = source("bf9357dd9a11ee6bbd08ad7336108949a2736b8b", "kb/exp/decode_vms.py")
    engine = definitions(moe, {"VMS_SRC", "VMS_CPP", "VMSGATE_ARGS", "_vmsgate_signature", "_vmsgate_static_check",
                               "_vmsgate_ext", "_vmsgate_repack", "_vmsgate_dequant"})
    helpers = definitions(old, {"_vd_inputs", "_vd_state", "_vd_cache", "_vd_reset_cache", "_vd_snapshot", "_vd_restore"})
    # The router is aligned to the public fixed stream basis, NOT a new random
    # mean per shape. Preserve that distribution in our fresh diagnostic draws.
    old_basis = '''mean = torch.randn(H, generator=gen, device=dev)
    shared = torch.randn((4, H), generator=gen, device=dev)'''
    new_basis = '''basis_gen = torch.Generator().manual_seed(5354)
    mean = torch.randn(H, generator=basis_gen).to(dev)
    shared = torch.randn((4, H), generator=basis_gen).to(dev)'''
    assert helpers.count(old_basis) == 1
    helpers = helpers.replace(old_basis, new_basis)
    header = '''# okc: problem=muse_decode_layer
# Decode incumbent + VMS32; Apache-2.0, inherited keell0renz/alygg77/chenwainuo kernels.
# Only d32S/d32F can opt in, each after an independent null-controlled three-pass gate.
def _dv_host_flags(flags=()):
    return [f for f in flags if not f.startswith(('-std=', '--std='))] + ['-std=c++20']

def _dv_nvcc_flags(flags=()):
    return [f for f in flags if not f.startswith(('-std=', '--std=', '-gencode', '--generate-code', '-arch=', '--gpu-architecture='))] + ['-std=c++20', '-gencode=arch=compute_90a,code=sm_90a']

'''
    # Retain the original top-level copyright/attribution notices verbatim.
    notices = "\n".join(line for line in base.splitlines()[:16] if line.startswith("#") and not line.startswith("# okc:")) + "\n"
    result = header + notices + flags(compact(base)) + "\n" + flags(engine) + "\n" + helpers + "\n"
    result += Path(__file__).with_name("decode_vms32_arm.py").read_text()
    ast.parse(result)
    target = Path(__file__).with_name("decode_vms32.py")
    target.write_text(result)
    assert target.stat().st_size < 1_000_000, target.stat().st_size
    print(target, target.stat().st_size)
