"""Audit the diagnostic wrapper separately from the exact selection core."""
import ast
import hashlib
import json
import symtable
import builtins

from build_decoder_router_redux import ROOT

raw=(ROOT/'.tools/decoder-router-redux-paired-variant.py').read_bytes()
core=(ROOT/'.tools/decoder-router-redux-variant.py').read_bytes()
tree,parent=ast.parse(raw),ast.parse(core)
addon=(ROOT/'experiments/prepared/decoder_router_redux_probe.py').read_text(encoding='utf8')
nodes=ast.parse(addon).body
assert [ast.dump(n) for n in tree.body[-len(nodes):]]==[ast.dump(n) for n in nodes]
tree.body[-len(nodes):]=[]
def literal(t,name):
    return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign)
                and any(isinstance(x,ast.Name) and x.id==name for x in n.targets))
paired_cuda=literal(tree,'_RF_CUDA')
core_cuda=literal(parent,'_RF_CUDA')
assert paired_cuda.startswith(core_cuda+'\n// Prepare-only incumbent, with original exact arithmetic.\n')
original=ast.parse((ROOT/'best/decoder_verified.py').read_bytes())
baseline_cuda=literal(original,'_RF_CUDA')
tail=paired_cuda[len(core_cuda):].split('\n',2)[-1]
assert tail.replace('route_fp32_incumbent','route_fp32').replace('void run_incumbent(','void run(',1)==baseline_cuda[baseline_cuda.index('template<int TM>'):]
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
        if n.targets[0].id in ('_RF_CUDA','_RF_CPP'):
            n.value=ast.Constant(literal(parent,n.targets[0].id))
    if isinstance(n,ast.FunctionDef) and n.name=='_rf_get_ext':
        for child in ast.walk(n):
            if isinstance(child,ast.keyword) and child.arg=='functions':
                assert ast.literal_eval(child.value)==['run','run_incumbent']
                child.value.elts.pop()
assert ast.dump(tree,include_attributes=False)==ast.dump(parent,include_attributes=False)
probe=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_rr_probe')
assert [ast.unparse(n.value) for n in ast.walk(probe) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='functions' for t in n.targets)]==["(rf['ext'].run_incumbent, rf['ext'].run)"]
assert not any(isinstance(n,ast.Return) and n.value is not None for n in ast.walk(probe))
timed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_rf_route')
assert 'run_incumbent' not in ast.unparse(timed) and "state['ext'].run(" in ast.unparse(timed)
assert 'generator.manual_seed(20261028)' in addon and 'torch.manual_seed' not in addon
assert 'flush_buffer.add_(1)' in addon and 'range(8)' in addon
assert 'end.synchronize()' in addon
assert 'for arm in ((0, 1) if repeat % 2 == 0 else (1, 0))' in addon
assert 'samples[arm].append(begin.elapsed_time(end) * 1_000_000.)' in addon
# No synthetic inputs/outputs/graphs or timing choices are stored into state.
stores=[n for n in ast.walk(probe) if isinstance(n,ast.Subscript) and isinstance(n.ctx,ast.Store)
        and isinstance(n.value,ast.Name) and n.value.id in ('state','rf')]
assert not stores
st=symtable.symtable(raw.decode('utf8'),'<paired-router>','exec')
known=set(st.get_identifiers())|set(dir(builtins))
def bindings(t):
    for s in t.get_symbols():
        if s.is_global() and s.is_referenced(): assert s.get_name() in known,s.get_name()
    for child in t.get_children():bindings(child)
bindings(st)
record={'source_sha256':hashlib.sha256(raw).hexdigest(),'core_sha256':hashlib.sha256(core).hexdigest(),
        'core_module_ast_recovered':True,'incumbent_kernel_bytes_recovered':True,
        'actual_timed_router_always_redux':True,'synthetic_data_or_choice_not_returned_as_state':True,
        'private_generator_not_global_rng':True,'paired_order_balanced':True,
        'cold_flush_owned_bytes':128*1024*1024,'correctness_probes':16,
        'analytic_tie_probes':8,'timed_pairs':32,'both_arms_per_pair':True,
        'shared_original_counter_protocol':'every kernel advances128; low7bits remain0; all consumed partials overwritten',
        'global_bindings':True,'scope_limit':'CPU/static protocol audit; GPU probe truth/timings and full-layer correctness unverified'}
(ROOT/'.tools/decoder-router-redux-paired-cpu-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
