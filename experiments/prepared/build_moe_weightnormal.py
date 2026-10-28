"""One WGS-weight cache priority; exact retained math, layout and scheduler."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
BASE_SHA = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
POLICY = '''// Local change: split jobs can reread a hot expert within this launch.
// Only WGS weight TMA loads use normal priority; other hints stay original.
#if OKC_L2H
#define WGS_WEIGHT_L2H_ARG , "l"(0x1000000000000000ull)
#else
#define WGS_WEIGHT_L2H_ARG
#endif
'''
PREFLIGHT = '''
weight_normal_enabled = bool(state['sfx'] and '-DOKC_WGS=1' in tuple(_main_defines(state['sfx'])))
print('MOE_WEIGHT_NORMAL_ENABLED', weight_normal_enabled, flush=True)
'''


def build():
    raw = (ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    parent = ast.parse(raw)
    tree = copy.deepcopy(parent)
    edits = []
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'CUDA_SRC' for t in n.targets):
            cuda = ast.literal_eval(n.value)
            begin = cuda.index('__device__ void wgs_wproducer(')
            end = cuda.index('// x producer:',begin)
            body = cuda[begin:end]
            assert body.count('L2H_ARG') == 1
            changed_body = body.replace(' L2H_ARG : "memory"',' WGS_WEIGHT_L2H_ARG : "memory"',1)
            assert changed_body != body
            changed = cuda[:begin] + POLICY + changed_body + cuda[end:]
            assert changed.replace(POLICY,'',1).replace(' WGS_WEIGHT_L2H_ARG : "memory"',' L2H_ARG : "memory"',1) == cuda
            assert changed.count('0x12F0000000000000ull') == cuda.count('0x12F0000000000000ull')
            assert changed.count('0x14F0000000000000ull') == cuda.count('0x14F0000000000000ull')
            assert '#define OKC_WGSGS 2' in changed and 'setmaxnreg' not in changed
            edits.append((n,'CUDA_SRC = '+repr(changed)))
        if isinstance(n,ast.FunctionDef) and n.name == 'prepare':
            n.body[-1:-1] = ast.parse(PREFLIGHT).body
            edits.append((n,ast.unparse(n)))
        if isinstance(n,ast.FunctionDef) and n.name == '_ext':
            for child in ast.walk(n):
                if isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id == 'load_inline':
                    kws = {k.arg:k for k in child.keywords}
                    assert ast.literal_eval(kws['verbose'].value) is False
                    kws['verbose'].value = ast.parse("'-DOKC_WGS=1' in defs",mode='eval').body
                    kws['extra_cuda_cflags'].value = ast.BinOp(left=kws['extra_cuda_cflags'].value,op=ast.Add(),
                        right=ast.parse("(['-Xptxas=-v'] if '-DOKC_WGS=1' in defs else [])",mode='eval').body)
            edits.append((n,ast.unparse(n)))
    lines = raw.decode('utf8').splitlines(keepends=True)
    for n,text in sorted(edits,key=lambda x:x[0].lineno,reverse=True):
        lines[n.lineno-1:n.end_lineno] = [text+'\n']
    result = ''.join(lines).encode('utf8')
    compile(result,'<moe-weight-normal>','exec')
    recovery = ast.parse(result)
    for n in recovery.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'CUDA_SRC' for t in n.targets):
            n.value = ast.Constant(cuda)
        if isinstance(n,ast.FunctionDef) and n.name == 'prepare':
            added = ast.parse(PREFLIGHT).body
            assert [ast.dump(x) for x in n.body[-len(added)-1:-1]] == [ast.dump(x) for x in added]
            n.body[-len(added)-1:-1] = []
        if isinstance(n,ast.FunctionDef) and n.name == '_ext':
            for child in ast.walk(n):
                if isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id == 'load_inline':
                    kws = {k.arg:k for k in child.keywords}
                    kws['verbose'].value = ast.Constant(False)
                    assert isinstance(kws['extra_cuda_cflags'].value,ast.BinOp)
                    kws['extra_cuda_cflags'].value = kws['extra_cuda_cflags'].value.left
    assert ast.dump(parent,include_attributes=False) == ast.dump(recovery,include_attributes=False)
    # The source address has no job-dependent term except S.jE[j]. Every job
    # on the same expert reads exactly the same phase/unit/stage address set.
    address_line = 'base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES'
    assert address_line in body and address_line in changed_body
    address_cases = []
    for count in (0,1,8,32,64,65,128,129,256,257,384,3072):
        jobs = [(q,min(64,count-q)) for q in range(0,count,64)]
        assert sum(c for q,c in jobs) == count and all(0 < c <= 64 for q,c in jobs)
        unique = {(pb,u,st,((5 << (4 if pb else 5))+u)*16+st)
                  for pb in (0,1) for u in range(16 if pb else 32) for st in range(16)} if jobs else set()
        assert len(unique) == (768 if count else 0)
        address_cases.append({'route_count':count,'jobs':len(jobs),'unique_weight_stages':len(unique),
                              'issued_weight_stages':len(jobs)*len(unique)})
    (ROOT/'.tools/moe-weightnormal-variant.py').write_bytes(result)
    (ROOT/'.tools/moe-weightnormal-inspect.cu').write_bytes(changed.encode('utf8'))
    record = {'source_sha256':hashlib.sha256(result).hexdigest(),'base_sha256':BASE_SHA,'packaged_bytes':len(result),
              'original_module_ast_recovered':True,'cuda_patch_reversible':True,
              'changed_policy':'WGS weight TMA only: evict_first to evict_normal',
              'all_other_cache_hints_unchanged':True,'all_addresses_barriers_math_layouts_unchanged':True,
              'registers_threads_pipeline':'original576threads/GS2, no dynamic redistribution',
              'address_cases':address_cases,'gpu_status':'unmeasured; logical reuse is not a measured cache hit or speedup'}
    (ROOT/'.tools/moe-weightnormal-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    build()
