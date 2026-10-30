"""Offline SM90 resource/SASS audit of the fixed O16/O32 PDL twins."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter

from pdl16_sources import derive, OUT, HERE, M16_PREFIX, M16_INST, O32_INST
from check_pdl_diag_cpu import Capture, MAIN
from check_r2_cpu import device_part
import nvrtc_screen_alternatives as nv


def instructions(sass):
    result = []
    for line in sass.splitlines():
        m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line)
        if m:
            result.append((int(m[1], 16), m[2]))
    return result


def ordering(name, sass):
    ins = instructions(sass)
    acq = [addr for addr, text in ins if 'ACQBULK' in text]
    assert len(acq) == 1
    calls = [(addr, int(re.search(r'0x([0-9a-f]+)', text)[1], 16))
             for addr, text in ins if 'CALL.REL.NOINC' in text]
    waits = [(addr, target) for addr, target in calls if target <= acq[0] < target + 64]
    assert len(waits) == 1
    call, target = waits[0]
    assert any(addr < call and 'PREEXIT' in text for addr, text in ins)
    if name == 'o16':
        loads = [(a, t) for a, t in ins if a < call and 'LDG.' in t]
        assert len(loads) == 4 and all('CONSTANT' in t for a, t in loads)
        dependent = [(a, t) for a, t in ins if 'LDG.' in t and 'STRONG.GPU' in t]
        assert dependent and all(a > call for a, t in dependent)
    else:
        loads = [(a, t) for a, t in ins if a < call and 'UBLKCP.S.G' in t]
        assert len(loads) == 8
        dependent = [(a, t) for a, t in ins if 'UTMALDG.3D' in t]
        assert dependent and all(a > call for a, t in dependent)
    return dict(wait_call_address=call, helper_target=target, acquire_address=acq[0],
                independent_loads_before_call=loads, first_dependent_load=dependent[0],
                scope='Instruction-order screen plus source control-flow review; not GPU validation.')


def ring_model():
    st, ab, xb = 8, 64 * 128 * 2, 32 * 128 * 2
    sb = ab + xb
    seen = [0] * (st * sb)
    for stage in range(st):
        for off in range(stage * sb, stage * sb + ab):
            seen[off] += 1
        for off in range(stage * sb + ab, (stage + 1) * sb):
            seen[off] += 1
        assert ab + xb == sb  # unchanged expect_tx per stage
    assert set(seen) == {1}
    # ST == IT: no refill or slot reuse can occur before the full-K ring is
    # consumed. The selected direct-register epilogue does not reuse shared
    # memory; the unchanged final WGMMA wait precedes its output stores.
    assert st == 1024 // 128
    return dict(stages=st, weight_bytes_per_stage=ab, activation_bytes_per_stage=xb,
                transaction_bytes_per_stage=sb, disjoint_complete_ring=True,
                dynamic_shared_bytes=st * (sb + 16),
                full_K_covered=True, new_global_traffic_bytes=0)


def proxy(name, source):
    if name == 'o16':
        return device_part(source)
    source = source.split('using OLEncode=', 1)[0]
    # Driver's CUtensorMap ABI: 128 bytes, alignment 64. No host driver API
    # or pointer emulation participates in the generated device instructions.
    source = source.replace('#include <cuda.h>', 'struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };')
    return source.replace('#include <cuda_runtime.h>', '')


def integration_checks():
    combined = '\n'.join(proxy(n, (OUT / (n + '-candidate.cu')).read_text()) for n in ['o16', 'o32'])
    expr = ['&pdlo16_gemv<' + ','.join(map(str, M16_INST)) + '>',
            '&pdlo32_projection<32,128,8,1,1,1,1>']
    nvr = Capture()
    r = nvr.compile(combined.encode(), expr, [], tags=True, lineinfo=True)
    (OUT / 'combined-device-proxy.cu').write_bytes(combined.encode())
    (OUT / 'combined-compile.log').write_text(r['log'])
    assert r['rc'] == 0, r['log'][-2000:]
    r['kernels'] = nv.parse_ptxas(r['log'])
    assert sorted(k['registers'] for k in r['kernels']) == [42, 72]
    assert all(k['spill_stores'] == k['spill_loads'] == k['stack_frame'] == 0 for k in r['kernels'])
    r.update(proxy_sha256=hashlib.sha256(combined.encode()).hexdigest(), GPU_jobs=0)
    (OUT / 'combined-compile.json').write_text(json.dumps(r, indent=2) + '\n')
    # Run only the data constructor on CPU, never the candidate module with
    # its GPU compiler threads. This checks page domains, not attention math.
    import torch
    module = ast.parse((HERE / 'pdl16_runtime.py').read_text())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_p16_data')
    env = {'torch': torch, 'ROPE_THETA': 10000.}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'metadata-only-CPU', 'exec'), env)
    tested = []
    for m in [16, 32]:
        gen = torch.Generator(device='cpu').manual_seed(19)
        data = env['_p16_data'](m, True, 'cpu', gen, ctx=8000)
        seq, loc, table = data['seq_lens'], data['cache_loc'], data['page_table']
        assert data['config']['batch'] == m and data['config']['q_len'] == 1 and data['config']['tokens'] == m
        assert bool((seq > 0).all()) and seq.unique().numel() == m
        assert torch.equal(table[torch.arange(m), seq.to(torch.int64)].to(torch.int64), loc)
        assert loc.unique().numel() == m and int(loc.max()) < data['k_cache'].shape[0]
        tested.append(dict(batch=m, q_len=1, mixed_min_context=int(seq.min()),
                           max_context=int(seq.max()), cache_locations_unique_in_bounds=True))
        del data
    record = dict(CPU_only=True, constructor_AST_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                  cases=tested, scope='Shape/index consistency only; no kernel numerical validation.')
    (OUT / 'metadata-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Combined compile and CPU metadata checks passed.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration-only', action='store_true')
    args = parser.parse_args()
    if args.integration_only:
        integration_checks()
        return
    sources = derive()
    nvr = Capture()
    records = {}
    exe = MAIN / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    for name, (parent, candidate, proof) in sources.items():
        for arm, source in [('parent', parent), ('candidate', candidate)]:
            if name == 'o16':
                kernel = M16_PREFIX + '_gemv' if arm == 'parent' else 'pdlo16_gemv'
                inst = M16_INST
            else:
                kernel = 'r3_o32_o_direct_a_ol_projection' if arm == 'parent' else 'pdlo32_projection'
                inst = O32_INST
            device = proxy(name, source)
            expression = '&' + kernel + '<' + ','.join(map(str, inst)) + '>'
            r = nvr.compile(device.encode(), [expression], [], tags=True, lineinfo=True)
            key = name + '-' + arm
            r.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     device_proxy_sha256=hashlib.sha256(device.encode()).hexdigest(), CPU_only=True)
            (OUT / (key + '-device-proxy.cu')).write_bytes(device.encode())
            (OUT / (key + '-compile.log')).write_text(r['log'])
            r['kernels'] = nv.parse_ptxas(r['log']) if not r['rc'] else []
            print(key, 'rc', r['rc'], r['kernels'], flush=True)
            if not r['rc']:
                cubin = OUT / (key + '.cubin')
                cubin.write_bytes(nvr.cubin)
                sass = subprocess.check_output([str(exe), '-sass', str(cubin)], text=True)
                (OUT / (key + '.sass')).write_bytes(sass.encode())
                excerpt = [x.strip() for x in sass.splitlines() if any(v in x for v in
                           ['Function :', 'LDG.', 'ACQBULK', 'CALL.', 'UTMALDG', 'UBLKCP', 'PREEXIT'])]
                r['ordering_excerpt'] = excerpt
                if arm == 'candidate':
                    r['ordering'] = ordering(name, sass)
                print('\n'.join(excerpt[:35]), flush=True)
            else:
                print(r['log'][-3000:], flush=True)
            (OUT / (key + '-compile.json')).write_bytes((json.dumps(r, indent=2) + '\n').encode())
            records[key] = r
    (OUT / 'compile-summary.json').write_bytes((json.dumps(records, indent=2) + '\n').encode())
    assert all(v['rc'] == 0 for v in records.values())
    audit = {'GPU_jobs': 0, 'o32_ring_model': ring_model(), 'resources': {}, 'math_instruction_counts': {}}
    for name in sources:
        old, new = (records[name + '-' + arm]['kernels'][0] for arm in ('parent', 'candidate'))
        assert old['registers'] == new['registers']
        for item in (old, new):
            assert not any(item[k] for k in ['stack_frame', 'spill_stores', 'spill_loads'])
        audit['resources'][name] = dict(registers=new['registers'], stack=0, spill_stores=0, spill_loads=0)
        counts = []
        for arm in ('parent', 'candidate'):
            ins = instructions((OUT / (name + '-' + arm + '.sass')).read_text())
            counts.append(Counter(re.sub(r'^@!?P\d+\s+', '', t).split()[0] for a, t in ins
                                  if re.search(r'\b(?:HMMA|HGMMA|FADD|FFMA|FMUL|MUFU|F2FP)\b', t)))
        audit['math_instruction_counts'][name] = dict(parent=dict(counts[0]), candidate=dict(counts[1]), equal=counts[0] == counts[1])
        assert counts[0] == counts[1]
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Resource, instruction-order, math-count and transaction-model audits passed.', flush=True)
    integration_checks()


if __name__ == '__main__':
    main()
