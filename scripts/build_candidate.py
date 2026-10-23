"""Assemble standalone kernels without importing untrusted submission modules."""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def functions(filename, renames):
    tree = ast.parse((ROOT / "candidates" / filename).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in renames and renames[node.name] is None:
                continue
            node.name = renames.get(node.name, node.name)
            nodes.append(ast.unparse(node))
    return "\n\n".join(nodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--small-limit", type=int, default=32)
    p.add_argument("--large", action="store_true")
    args = p.parse_args()
    chunks = ["# okc: problem=muse_bf16_proj\nimport torch\nimport triton\nimport triton.language as tl"]
    chunks.append(functions("small_batch.py", {"prepare": "_small_prepare", "kernel_fn": "_small_kernel"}))
    chunks.append(functions("large_batch.py", {"prepare": None, "kernel_fn": None}))
    chunks.append(functions("router_only.py", {"prepare": "_router_prepare", "kernel_fn": None}))
    chunks.append('''def prepare(weights):
    return {"small": _small_prepare(weights), "router": _router_prepare(weights)}
''')
    large_branch = 'return large_bf16_gemm(x, w, n)' if args.large else 'return torch.mm(x, w[:n].T)'
    chunks.append(f'''def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    n, layer = cfg["n"], cfg["layer"]
    if layer == "router":
        return router_gemm(x, state["router"], n)
    if x.shape[0] <= {args.small_limit}:
        return _small_kernel(data, state["small"])
    w = state["small"]["weights"][layer]
    {large_branch}
''')
    text = "\n\n".join(chunks) + "\n"
    ast.parse(text)
    output = ROOT / args.output
    output.resolve().relative_to(ROOT)
    output.write_text(text, encoding="utf-8", newline="\n")
    print(output.relative_to(ROOT), hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
