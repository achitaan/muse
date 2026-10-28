"""Add a controlled prepare-only incumbent comparison to the exact redux core."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from build_decoder_router_redux import ROOT, BASE_SHA, build as build_core


def build():
    build_core()
    core = (ROOT/'.tools/decoder-router-redux-variant.py').read_bytes()
    parent = ast.parse(core)
    original = ast.parse((ROOT/'best/decoder_verified.py').read_bytes())
    def literal(tree,name):
        return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id == name for t in n.targets))
    old_cuda = literal(original,'_RF_CUDA')
    begin = old_cuda.index('template<int TM>')
    host = old_cuda.index('\nvoid run(')
    incumbent = old_cuda[begin:].replace('route_fp32','route_fp32_incumbent').replace('void run(', 'void run_incumbent(',1)
    cuda = literal(parent,'_RF_CUDA')+'\n// Prepare-only incumbent, with original exact arithmetic.\n'+incumbent
    cpp = literal(parent,'_RF_CPP')
    cpp += '\n'+cpp[cpp.index('void run('):].strip().replace('void run(', 'void run_incumbent(',1)+'\n'
    edits = []
    tree = copy.deepcopy(parent)
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id in ('_RF_CUDA','_RF_CPP'):
                value = cuda if n.targets[0].id == '_RF_CUDA' else cpp
                edits.append((n,n.targets[0].id+' = '+repr(value)))
        if isinstance(n,ast.FunctionDef) and n.name == '_rf_get_ext':
            for child in ast.walk(n):
                if isinstance(child,ast.keyword) and child.arg == 'functions':
                    assert ast.literal_eval(child.value)==['run']
                    child.value.elts.append(ast.Constant('run_incumbent'))
            edits.append((n,ast.unparse(n)))
    lines=core.decode('utf8').splitlines(keepends=True)
    for n,text in sorted(edits,key=lambda x:x[0].lineno,reverse=True):
        lines[n.lineno-1:n.end_lineno]=[text+'\n']
    addon=(ROOT/'experiments/prepared/decoder_router_redux_probe.py').read_text(encoding='utf8')
    text=''.join(lines)+'\n# Controlled diagnostics use synthetic inputs only, during prepare.\n'+addon
    raw=text.encode('utf8')
    compile(raw,'<decoder-router-redux-paired>','exec')
    assert len(raw)<1_000_000
    recovered=ast.parse(raw)
    addon_nodes=ast.parse(addon).body
    assert [ast.dump(n) for n in recovered.body[-len(addon_nodes):]]==[ast.dump(n) for n in addon_nodes]
    recovered.body[-len(addon_nodes):]=[]
    for n in recovered.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id in ('_RF_CUDA','_RF_CPP'):
                n.value=ast.Constant(literal(parent,n.targets[0].id))
        if isinstance(n,ast.FunctionDef) and n.name=='_rf_get_ext':
            for child in ast.walk(n):
                if isinstance(child,ast.keyword) and child.arg=='functions':
                    assert ast.literal_eval(child.value)==['run','run_incumbent']
                    child.value.elts.pop()
    assert ast.dump(recovered,include_attributes=False)==ast.dump(parent,include_attributes=False)
    assert incumbent.replace('route_fp32_incumbent','route_fp32').replace('void run_incumbent(','void run(',1)==old_cuda[begin:]
    (ROOT/'.tools/decoder-router-redux-paired-variant.py').write_bytes(raw)
    record={'source_sha256':hashlib.sha256(raw).hexdigest(),'core_sha256':hashlib.sha256(core).hexdigest(),
            'base_sha256':BASE_SHA,'packaged_bytes':len(raw),'core_ast_recovered':True,
            'incumbent_cuda_bytes_recovered':True,'prepare_probes':'M1..8 normal plus analytic zero-input/zero-bias ties',
            'controlled_timings':'M1/8,normal/impulse,8alternating cold CUDA-graph pairs each',
            'always_timed_router':'redux; no timing selector or actual-input work during prepare',
            'new_module_or_weight_copies':0,'gpu_status':'unmeasured'}
    (ROOT/'.tools/decoder-router-redux-paired-source-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__=='__main__':
    build()
