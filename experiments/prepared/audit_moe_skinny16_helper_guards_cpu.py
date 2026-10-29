"""Detect every helper-only field use outside its lexical OKC_WS guard."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
from moe_skinny16_evidence import PREFLIGHT, verify_repair

raw = (PREFLIGHT / 'submission.py').read_bytes()
assert verify_repair(raw)
old_raw = (ROOT / 'results/moe-skinny16-selection-20261028/submission.py').read_bytes()
def alternate(payload):
    return next(ast.literal_eval(n.value) for n in ast.parse(payload).body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Subscript) and ast.unparse(x) == "ALT_SRCS['sk16']" for x in n.targets))
def unguarded(source):
    stack, violations, references = [], [], 0
    for number, line in enumerate(source.splitlines(), 1):
        directive = re.match(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)', line)
        if directive:
            kind, expression = directive.group(1), directive.group(2).strip()
            if kind in ('if', 'ifdef', 'ifndef'): stack.append([kind, expression, 0])
            elif kind == 'elif': stack[-1][2] = 2
            elif kind == 'else': stack[-1][2] = 1
            else: stack.pop()
        elif re.search(r'\bS\.(wse|wsrf|wsre)\b|\bOKC_WSB\b', line) and not line.lstrip().startswith('//'):
            references += 1
            if not any(kind == 'if' and expression == 'OKC_WS' and branch == 0 for kind, expression, branch in stack):
                violations.append(number)
    assert not stack
    return references, violations
old_refs, original_violations = unguarded(alternate(old_raw))
new_refs, violations = unguarded(alternate(raw))
assert old_refs == new_refs and len(original_violations) == 4
assert not violations
report = dict(source_sha256=hashlib.sha256(raw).hexdigest(), all_helper_references_guarded=True,
              references_checked=new_refs, unguarded_original_lines=original_violations,
              original_compile_failure_mutant_detected=True,
              complete_trial_recovered_by_inverse=True,
              limitation='Lexical conditional/source audit; CUDA compilation remains unverified')
(PREFLIGHT / 'helper-guard-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
