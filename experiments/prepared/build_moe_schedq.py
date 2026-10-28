"""Replace one scheduler division estimate, retaining exact integer correction."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
BASE_SHA = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
OLD = '  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);'
NEW = '''  // Local change: approximate only the starting CTA; the original integer
  // correction loops below still choose the exact cost-interval owner.
  int c = (int)__fdividef((float)(xe - 1) * (float)G, (float)tot);'''


def build():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    text = raw.decode('utf8')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    cuda = ast.literal_eval(node.value)
    assert cuda.count(OLD) == 1
    changed = cuda.replace(OLD, NEW, 1)
    assert changed.replace(NEW, OLD, 1) == cuda
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ['CUDA_SRC = ' + repr(changed) + '\n']
    result = ''.join(lines).encode('utf8')
    compile(result, '<moe-scheduler-quotient>', 'exec')
    (ROOT / '.tools/moe-schedq-variant.py').write_bytes(result)
    (ROOT / '.tools/moe-schedq-inspect.cu').write_bytes(changed.encode('utf8'))
    record = {'base_sha256': BASE_SHA, 'source_sha256': hashlib.sha256(result).hexdigest(),
              'packaged_bytes': len(result), 'cuda_replacements': 1,
              'original_integer_correction_loops_unchanged': True,
              'precision_and_weights_unchanged': True, 'gpu_status': 'unmeasured'}
    (ROOT / '.tools/moe-schedq-source-audit.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    build()
