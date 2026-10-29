"""List / dump top-level string constants of a submission (offline, no import)."""
import ast, sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
tree = ast.parse(src)
want = sys.argv[2] if len(sys.argv) > 2 else None
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        for t in node.targets:
            if isinstance(t, ast.Name):
                if want is None:
                    print(f"{t.id:40s} line {node.lineno:6d} len {len(node.value.value)}")
                elif t.id == want:
                    out = Path(sys.argv[3])
                    out.write_text(node.value.value, encoding="utf-8")
                    print("wrote", out, len(node.value.value))
    elif want is None and isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        print(f"{'def ' + node.name:40s} line {node.lineno:6d}")
