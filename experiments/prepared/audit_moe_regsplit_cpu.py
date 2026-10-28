"""Reversible source ledger, barrier membership and register-pool liveness."""
import ast
import builtins
import contextlib
import hashlib
import io
import json
import re
import symtable
import types

from build_moe_regsplit import (BASE_SHA, ROOT, NTL_OLD, NTL_NEW, ROLE_OLD, ROLE_NEW,
                                RESOURCES, PREPARE_PREFLIGHT)


def literals(tree):
    return {n.targets[0].id: n.value.value for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}


def audit():
    base = (ROOT / 'best/moe_verified.py').read_bytes()
    raw = (ROOT / '.tools/moe-regsplit-variant.py').read_bytes()
    assert hashlib.sha256(base).hexdigest() == BASE_SHA
    bt, ct = ast.parse(base), ast.parse(raw)
    bl, cl = literals(bt), literals(ct)
    cuda = cl['CUDA_SRC']
    assert cuda.endswith(RESOURCES)
    recovered = cuda[:-len(RESOURCES)]
    assert recovered.count(ROLE_NEW) == recovered.count(NTL_NEW) == 1
    recovered = recovered.replace(ROLE_NEW, ROLE_OLD, 1).replace(NTL_NEW, NTL_OLD, 1)
    assert recovered == bl['CUDA_SRC']
    # Reverse only the declared host metadata/export/prepare changes.
    for n in ct.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ('CUDA_SRC', 'CPP_SRC'):
                n.value = ast.Constant(bl[n.targets[0].id])
        if isinstance(n, ast.FunctionDef) and n.name == '_ext':
            for child in ast.walk(n):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    child.value = child.value.replace('v216x_pair1dq8regsplit_', 'v216x_pair1dq8_')
                if isinstance(child, ast.keyword) and child.arg == 'functions':
                    assert ast.literal_eval(child.value) == ['okc_run', 'okc_set_pdl', 'okc_regsplit_resources']
                    child.value.elts.pop()
        if isinstance(n, ast.FunctionDef) and n.name == 'prepare':
            added = ast.parse(PREPARE_PREFLIGHT).body
            assert [ast.dump(x) for x in n.body[-len(added)-1:-1]] == [ast.dump(x) for x in added]
            n.body[-len(added)-1:-1] = []
    assert ast.dump(bt, include_attributes=False) == ast.dump(ct, include_attributes=False)
    # Device math, weight representations, all layouts, routing and work ranges
    # are included in the exact CUDA inversion above, not approximated models.
    assert 'const __grid_constant__' not in cuda
    assert 'setmaxnreg' not in bl['CUDA_SRC'] and cuda.count('setmaxnreg.') == 2
    assert 'constexpr int GS = OKC_WGSGS;' in cuda and '#define OKC_WGSGS 2' in cuda
    assert '"<<<"' not in RESOURCES and 'okc_run(' not in RESOURCES
    assert re.search(r'bar.sync 7, %0;[^\n]*"n"\(NT \+ 32\)', cuda)
    assert '#define WS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL)' in cuda
    assert '#define CTA_SYNC() asm volatile("bar.sync 1, %0;\\n" ::"n"(NT)' in cuda
    assert cuda.count('dim3(okc::NTL)') == 1 and cuda.count('okc::NTL, sm, st>>>') == 2

    producer_count = int(re.search(r'OKC_WGS \? (\d+)', NTL_NEW)[1])
    weight_limit = int(re.search(r'tid < NT \+ (\d+)', ROLE_NEW)[1])
    padding_limit = int(re.search(r'tid >= NT \+ (\d+)', ROLE_NEW)[1])
    nt, ntl = 512, 512 + producer_count
    consumers = set(range(nt))
    weights = set(range(nt, nt+weight_limit))
    xcopy = set(range(nt+weight_limit, nt+padding_limit))
    padding = set(range(nt+padding_limit, ntl))
    all_threads = consumers | weights | xcopy | padding
    assert len(consumers) == 512 and len(weights) == len(xcopy) == 32 and len(padding) == 64
    assert len(all_threads) == ntl == 640
    assert sum(map(len, (consumers, weights, xcopy, padding))) == len(all_threads)
    warp_action = []
    for first in range(0, ntl, 128):
        group = set(range(first, first+128))
        assert group <= consumers or group <= weights | xcopy | padding
        warp_action.append('inc112' if group <= consumers else 'dec32')
    assert warp_action == ['inc112']*4 + ['dec32']
    # Barrier 2 has every launched thread exactly once. Barrier 7 has routing
    # consumers plus just the x-copy warp. Other work barriers retain only the
    # original compute threads; padding exits before any of those operations.
    barrier2 = consumers | weights | xcopy | padding
    barrier7 = consumers | xcopy
    assert len(barrier2) == 640 and len(barrier7) == 544
    assert len(consumers) == 512 and len(consumers)//32 == 16
    assert not (padding & barrier7)
    assert ROLE_NEW.index('setmaxnreg.dec') < ROLE_NEW.index('WS_JOIN()')
    assert ROLE_NEW.index('setmaxnreg.inc') > ROLE_NEW.index('return;\n  }')
    assert 'WS_JOIN(); return;' in ROLE_NEW

    # An independent resource conservation graph allows each of four producer
    # warps to release2048 registers, and each of sixteen compute warps to
    # acquire512. It models partial warpgroup admissions and arbitrary order.
    # If any consumer is blocked, an unreleased producer has no dependency on
    # it before dec; if all producers released, every remaining request fits.
    initial = 96*640
    assert initial == 112*512 + 32*128 == 61440 <= 65536
    states, pending = set(), [(0, 0)]
    transitions = 0
    while pending:
        released, granted = pending.pop()
        if (released, granted) in states:
            continue
        states.add((released, granted))
        pool = released*2048 - granted*512
        assert pool >= 0
        next_states = []
        if released < 4:
            next_states.append((released+1, granted))
        if granted < 16 and pool >= 512:
            next_states.append((released, granted+1))
        assert next_states or (released, granted) == (4, 16)
        transitions += len(next_states)
        pending.extend(next_states)
    assert (4, 16) in states

    # Exercise the actual inserted prepare code with normal Python objects.
    # No kernels or extra allocations occur in this block.
    preflight_cases = []
    good = (640,96,96,0,0,231168,1,1,65536,132)
    for label, resources, prmt, cap, grid, expected in (
        ('admitted',good,True,(9,0),132,True),
        ('no_prmt',good,False,(9,0),132,False),
        ('small_initial',(640,88,*good[2:]),True,(9,0),132,'fail'),
        ('large_initial',(640,104,*good[2:]),True,(9,0),132,'fail'),
        ('big_kernel_mismatch',(*good[:2],88,*good[3:]),True,(9,0),132,'fail'),
        ('no_residency',(*good[:6],0,*good[7:]),True,(9,0),132,'fail'),
        ('register_file_too_small',(*good[:8],60000,132),True,(9,0),132,'fail'),
        ('grid_too_large',good,True,(9,0),133,'fail'),
        ('wrong_arch',good,True,(8,0),132,'fail'),
    ):
        queries = []
        def ext(defs):
            queries.append(defs)
            return types.SimpleNamespace(okc_regsplit_resources=lambda: resources)
        ns = {'state':{'sfx':('-DOKC_PRMTSC=1',) if prmt else (), 'grid':grid},
              '_main_defines':lambda sfx: ('-DOKC_NW=16','-DOKC_WGS=1'), '_ext':ext,
              'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda dev:cap)),
              'dev':'cuda:0'}
        failed = False
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(PREPARE_PREFLIGHT,'<actual-preflight>','exec'),ns)
            except AssertionError:
                failed = True
        if expected == 'fail':
            assert failed
        else:
            assert not failed and ns['regsplit_enabled'] == expected
        assert len(queries) == int(prmt)
        preflight_cases.append({'case':label,'outcome':expected})
    # Every compile/dispatch choice is retained; non-WGS configurations use
    # exactly the old total thread count (256 or288 under existing flags).
    declarations = {}
    for node in bt.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id.endswith('_DEFINES'):
                try: declarations[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError,TypeError): pass
    configurations = []
    for name in ('MAIN_DEFINES','TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'):
        flags = {x[2:].split('=')[0]:int(x.split('=')[1]) for x in declarations[name] if x.startswith('-D')}
        nw, wgs, ws = flags.get('OKC_NW',8),flags.get('OKC_WGS',0),flags.get('OKC_WS',0)
        old_threads = nw*32 + (64 if wgs else 32 if ws else 0)
        new_threads = nw*32 + (128 if wgs else 32 if ws else 0)
        assert (old_threads,new_threads) == ((576,640) if wgs else (old_threads,old_threads))
        configurations.append({'name':name,'old_threads':old_threads,'new_threads':new_threads})
    tree = symtable.symtable(raw.decode('utf8'),'<candidate>','exec')
    known = set(tree.get_identifiers()) | set(dir(builtins))
    def bindings(table):
        for symbol in table.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                assert symbol.get_name() in known, (table.get_name(),symbol.get_name())
        for child in table.get_children(): bindings(child)
    bindings(tree)
    record = {'source_sha256':hashlib.sha256(raw).hexdigest(),'base_sha256':BASE_SHA,
              'original_module_ast_recovered':True,'cuda_edits_reversible':True,
              'all_numerical_cuda_bytes_unchanged':True,'warpgroup_actions':warp_action,
              'barrier_membership_counts':{'full_join':640,'routing_join':544,'compute':512},
              'register_pool_states':len(states),'register_pool_transitions':transitions,
              'no_pool_dead_end':True,'total_registers':initial,'preflight_cases':preflight_cases,
              'compile_configurations':configurations,'global_bindings':True,
              'scope_limit':'CPU source/control/budget proof; actual compiler attributes, occupancy, GPU numerics and speed unverified'}
    (ROOT/'.tools/moe-regsplit-cpu-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    audit()
