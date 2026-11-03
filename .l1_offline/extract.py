from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'experiments/boards/moe/moe_b13.py'
t=p.read_bytes().decode()
n=next(n for n in ast.parse(t).body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='CUDA_SRC' for v in n.targets))
(ROOT/'.l1_offline/base.cu').write_text(ast.literal_eval(n.value))
