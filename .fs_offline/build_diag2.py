"""Create fsdiag2 ONLY; never rewrite the frozen candidate or fsdiag."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'experiments/boards/diag/fsdiag.py'
frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (source,ROOT/'experiments/boards/moe/moe_fs1.py')}
text=source.read_bytes().decode()+'\n\n# FSdiag2: separate instrumented and ablation builds.\n'
for file in ('fsdiag2_source.py','fsdiag2_body.py'):text+='\n'+(ROOT/'.fs_offline'/file).read_text()
ast.parse(text)
out=ROOT/'experiments/boards/diag/fsdiag2.py';out.write_bytes(text.encode())
(ROOT/'.fs_offline/frozen_diag2.json').write_text(json.dumps(frozen,indent=2))
print(out,len(text.encode()),hashlib.sha256(text.encode()).hexdigest())
