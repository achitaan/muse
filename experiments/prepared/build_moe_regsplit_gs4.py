"""Test the specific GS4 register requirement on the admitted GS2 split."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
PARENT_SHA = '17adb3e692d36fbb92af4a38b1570bfefd028696426a29ec9293517e581f6824'


def build():
    source = (ROOT/'results/resumed-moe-phase10-20261028'/PARENT_SHA/'submission.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == PARENT_SHA
    parent = ast.parse(source)
    tree = copy.deepcopy(parent)
    edits = []
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'CUDA_SRC' for t in n.targets):
            cuda = ast.literal_eval(n.value)
            assert cuda.count('#define OKC_WGSGS 2') == 1
            changed = cuda.replace('#define OKC_WGSGS 2','#define OKC_WGSGS 4',1)
            assert changed.replace('#define OKC_WGSGS 4','#define OKC_WGSGS 2',1) == cuda
            edits.append((n,'CUDA_SRC = '+repr(changed)))
            n.value = ast.Constant(changed)
        if isinstance(n,ast.FunctionDef) and n.name == '_ext':
            for child in ast.walk(n):
                if isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id == 'load_inline':
                    kws = {k.arg:k for k in child.keywords}
                    assert ast.literal_eval(kws['verbose'].value) is False
                    kws['verbose'].value = ast.parse("'-DOKC_WGS=1' in defs",mode='eval').body
                    kws['extra_cuda_cflags'].value = ast.BinOp(
                        left=kws['extra_cuda_cflags'].value,op=ast.Add(),
                        right=ast.parse("(['-Xptxas=-v'] if '-DOKC_WGS=1' in defs else [])",mode='eval').body)
            edits.append((n,ast.unparse(n)))
    lines = source.decode('utf8').splitlines(keepends=True)
    for n,text in sorted(edits,key=lambda x:x[0].lineno,reverse=True):
        lines[n.lineno-1:n.end_lineno] = [text+'\n']
    result = ''.join(lines).encode('utf8')
    compile(result,'<moe-register-split-gs4>','exec')
    # Reverse both the single numerical-source macro and build diagnostics.
    check = ast.parse(result)
    for n in check.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'CUDA_SRC' for t in n.targets):
            n.value = ast.Constant(cuda)
        if isinstance(n,ast.FunctionDef) and n.name == '_ext':
            for child in ast.walk(n):
                if isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id == 'load_inline':
                    kws = {k.arg:k for k in child.keywords}
                    assert ast.unparse(kws['verbose'].value) == "'-DOKC_WGS=1' in defs"
                    kws['verbose'].value = ast.Constant(False)
                    flag_expression = kws['extra_cuda_cflags'].value
                    assert isinstance(flag_expression,ast.BinOp) and isinstance(flag_expression.op,ast.Add)
                    assert '-Xptxas=-v' in ast.unparse(flag_expression.right)
                    kws['extra_cuda_cflags'].value = flag_expression.left
    assert ast.dump(check,include_attributes=False) == ast.dump(parent,include_attributes=False)
    # Independent schedule oracle: both settings issue the same sixteen K16
    # steps in order. Their only change is commit/fence/wait grouping and the
    # number of A-fragment registers retained between those instructions.
    schedules = {}
    for gs in (2,4):
        steps, groups, flush = [], [], []
        for lg in range(4):
            for h in range(4//gs):
                ci = lg*(4//gs)+h
                group = []
                for jj in range(gs):
                    js = h*gs+jj
                    k16 = lg*4+js
                    steps.append(k16)
                    group.append(k16)
                groups.append(group)
                if ci == 3:
                    flush.append(ci)
        assert steps == list(range(16)) and flush == [3]
        assert all(len(x) == gs for x in groups)
        # Alternating fragment bank i%2 is reused only after the group i-2
        # has retired at the prior wait_group1. At most two groups are live.
        active = []
        for i in range(len(groups)):
            assert i%2 not in [g%2 for g in active]
            active.append(i)
            active = active[-1:]  # post-commit wait_group1
        schedules[str(gs)] = {'k16_steps':steps,'commit_groups':groups,
                               'deferred_flush_commit':flush[0],'a_fragment_registers':2*gs*4}
    assert schedules['4']['a_fragment_registers']-schedules['2']['a_fragment_registers'] == 16
    assert 112-96 == 16
    assert '#define OKC_WGSFCI 3' in changed
    assert 'wg_wait<1>();' in changed and 'wg_wait<0>();' in changed
    assert 'constexpr int GS = OKC_WGSGS;' in changed
    assert changed.count('#define OKC_WGSGS 4') == 1
    (ROOT/'.tools/moe-regsplit-gs4-variant.py').write_bytes(result)
    (ROOT/'.tools/moe-regsplit-gs4-inspect.cu').write_bytes(changed.encode('utf8'))
    record = {'source_sha256':hashlib.sha256(result).hexdigest(),'parent_sha256':PARENT_SHA,
              'retained_baseline_sha256':'331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50',
              'packaged_bytes':len(result),'parent_module_ast_recovered':True,
              'cuda_only_changed_macro':'OKC_WGSGS:2to4','diagnostics':'main-only verbose build and ptxas-v',
              'fixed_register_budgets':[96,32,112],'schedules':schedules,
              'gpu_status':'unmeasured; static schedule equivalence does not establish GPU correctness or a speedup'}
    (ROOT/'.tools/moe-regsplit-gs4-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    build()
