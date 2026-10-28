"""Audit route coverage, real dispatch, scratch bounds and WGS32 source edits."""
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import symtable
import types

import torch

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
raw = (root / '.tools/moe-wgs32-variant.py').read_bytes()
draft = raw.decode('utf-8')
tree = ast.parse(draft)
base = ast.parse((root / 'best/moe_verified.py').read_bytes())
record = json.loads((root / '.tools/moe-wgs32-source-audit.json').read_text(encoding='utf-8'))
assert hashlib.sha256(raw).hexdigest() == record['source_sha256']
literal = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                try: literal[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
restored = copy.deepcopy(tree.body[:len(base.body)])
for old, node in zip(base.body, restored):
    if ast.dump(old) == ast.dump(node): continue
    if isinstance(node, ast.Assign):
        assert node.targets[0].id in ('CUDA_SRC', 'CPP_SRC') and node.value.value.startswith(old.value.value)
        node.value = copy.deepcopy(old.value)
    elif node.name == '_ext':
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                child.value = child.value.replace('v216x_pair1dq8wgs32_', 'v216x_pair1dq8_')
            if isinstance(child, ast.keyword) and child.arg == 'functions':
                assert ast.literal_eval(child.value) == ['okc_run', 'okc_set_pdl', 'okc_resources']
                child.value.elts.pop()
    elif node.name == 'kernel_fn':
        node.body = [n for n in node.body if not (isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in ('scratch', 'wgs32') for t in n.targets))]
        index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'wgs32 is not None and 17 <= M <= 32')
        assert len(node.body[index].orelse) == 1
        node.body[index:index + 1] = node.body[index].orelse
        call = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'okc_run')
        assert ast.unparse(call.args[1]) == 'scratch'
        call.args[1] = ast.parse("state['scratch']", mode='eval').body
    else:
        raise AssertionError(getattr(node, 'name', ast.dump(node)[:80]))
    assert ast.dump(node) == ast.dump(old)
assert ast.dump(ast.Module(body=restored, type_ignores=[])) == ast.dump(base)

# Independently reconstruct the exact alternate edits, then reverse them by
# recorded byte offsets. No other CUDA mathematical statement may change.
source = literal['CUDA_SRC']
expected = source
edits = []
def edit(start, stop, replacement):
    global expected
    edits.append((start, expected[start:stop], replacement))
    expected = expected[:start] + replacement + expected[stop:]
i = expected.index('#define OKC_FT_T 64')
edit(i, i + len('#define OKC_FT_T 64'), '#define OKC_FT_T 32')
for begin, end in [('template <> struct WG<5> {', '// one 36 KB k-stage of a fat unit'),
                   ('template <> struct WGSMMA<5> {', '// two producer warps:')]:
    start = expected.index(begin)
    edit(start, expected.index(end, start), '')
for name in ('fat_compute_wg', 'fat_compute'):
    start = expected.index('          case 5: ' + name + '<5>')
    edit(start, expected.index('\n        }', start), '          default: __trap(); break;')
start = expected.index('      case 5: wgs_seg<PB, 5>')
edit(start, expected.index('\n    }', start), '      default: __trap(); break;')
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_wgs32_cuda_source')
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), '<pure-alternate>', 'exec'), namespace)
alternate = namespace['_wgs32_cuda_source'](source)
assert alternate == expected and hashlib.sha256(alternate.encode()).hexdigest() == record['alternate_cuda_sha256']
for start, old, new in reversed(edits):
    assert expected[start:start + len(new)] == new
    expected = expected[:start] + old + expected[start + len(new):]
assert expected == source
assert all(f'{family}<{n}>' not in alternate for family in ('WG', 'WGSMMA', 'fat_compute', 'fat_compute_wg') for n in range(5, 9))
assert all(f'wgs_seg<PB, {n}>' not in alternate for n in range(5, 9))
assert 'min(FT_T, tot - q * FT_T)' in source and 'min(FT_T, tot - q * FT_T)' in alternate

# Analytical oracle: split each expert's complete list of physical routes into
# consecutive chunks; the union must be exactly the original local-route set.
def jobs_for(routes, fat_t):
    jobs = []
    for e in range(32):
        entries = [i for i, expert in enumerate(routes) if expert == e]
        step = fat_t if len(entries) > 8 else 8
        jobs.extend(entries[start:start + step] for start in range(0, len(entries), step))
    return jobs
for fat_t in (32, 64):
    for count in range(3073):
        step = fat_t if count > 8 else 8
        lengths = [min(step, count - start) for start in range(0, count, step)]
        assert sum(lengths) == count and all(0 < n <= fat_t and (n + 7) // 8 <= fat_t // 8 for n in lengths)
generator = torch.Generator().manual_seed(20261028)
route_cases = 0
for m in (*range(1, 65), 127, 128, 256, 257, 383, 384):
    diffuse = torch.randint(0, 64, (m * 8,), generator=generator).tolist()
    patterns = [[40] * (m * 8), [0] * (m * 8), [i % 32 for i in range(m * 8)], diffuse]
    for routes in patterns:
        expected_routes = sorted(i for i, e in enumerate(routes) if 0 <= e < 32)
        for fat_t in (32, 64):
            jobs = jobs_for(routes, fat_t)
            actual = sorted(i for job in jobs for i in job)
            assert actual == expected_routes and len(actual) == len(set(actual))
            assert len(jobs) <= 32 + (m * 8) // fat_t + 8
            assert sum((len(job) + 7) // 8 * 8 for job in jobs) <= 3072 + 8 * (32 + 3072 // fat_t + 8)
            route_cases += 1

# Actual launcher layouts share the maximum allocation but have differing
# fixup counter offsets. Separate allocations are mandatory, even sequentially.
emax, hidden, slots = 3072, 4096, 160
allocated = [8 + 1024 + emax, emax + emax * hidden + 2 * slots * 4 * 4096 + 2 * slots * 4 + 136 * 16,
             (emax + 2 * (emax + 8 * (32 + emax // 32 + 8))) * hidden]
layouts = []
for fat_t in (32, 64):
    jmax, nt = 32 + emax // fat_t + 8, fat_t // 8
    fixw = nt * 512
    fixf = 2 * slots * 4 * fixw + 2 * slots * 4 + 136 * 16
    xrows = emax + 8 * jmax
    required = [8 + 1024 + emax, emax + emax * hidden + fixf, (emax + 2 * xrows) * hidden]
    assert all(a >= r for a, r in zip(allocated, required))
    assert 8 + jmax <= 8 + 512 and 8 + 512 + jmax <= 8 + 1024
    assert jmax * 16 <= 136 * 16
    layouts.append({'fat_tokens': fat_t, 'jmax': jmax, 'token_tiles': nt,
                    'fixup_counter_offset_from_ybuf': emax * hidden + 2 * slots * 4 * fixw,
                    'xrows': xrows, 'required_elements': required})
assert layouts[0]['fixup_counter_offset_from_ybuf'] != layouts[1]['fixup_counter_offset_from_ybuf']

# Exercise actual Python dispatch without claiming numerical results from mocks.
class Tensor:
    next_id = 0
    def __init__(self, shape, dtype='bf16', aligned=True):
        Tensor.next_id += 1
        self.identity, self.shape, self.dtype, self.device = Tensor.next_id, shape, dtype, 'cuda:0'
        self.aligned = aligned
    def contiguous(self): return self
    def to(self, dtype): return Tensor(self.shape, dtype, self.aligned)
    def data_ptr(self): return self.identity * 256 + (0 if self.aligned else 1)
    def clone(self): return Tensor(self.shape, self.dtype)
    def __getitem__(self, sl):
        start, stop, step = sl.indices(self.shape[0])
        return Tensor((len(range(start, stop, step)), *self.shape[1:]), self.dtype, self.aligned)
class Module:
    def __init__(self, key, calls): self.key, self.calls = key, calls
    def okc_run(self, weights, scratch, x, ids, tw, out, *args):
        self.calls.append((self.key, scratch, x.shape[0], x.dtype, ids.dtype, tw.dtype, out.dtype))
    def okc_resources(self): return [0] * 7  # resource protocol only; no hardware claim
fake_torch = types.SimpleNamespace(bfloat16='bf16', int32='i32', float32='f32',
    empty=lambda shape, dtype, device: Tensor(shape, dtype))
dispatch = dict(literal, torch=fake_torch, CHUNK_TOKENS=384, HIDDEN=4096)
for name in ('MAIN_DEFINES', 'MAIN_FALLBACK_DEFINES', 'TINY_DEFINES', 'TINY2_DEFINES', 'MID_DEFINES'):
    dispatch[name] = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
for node in tree.body[:len(base.body)]:
    if isinstance(node, ast.FunctionDef) and node.name in ('kernel_fn', '_main_defines'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-dispatch>', 'exec'), dispatch)
dispatch_cases = 0
for enabled in (False, True):
    for sfx in ((), ('-DOKC_PRMTSC=1',)):
        for m in (*range(1, 66), 128, 256, 257, 383, 384, 385, 768, 769):
            calls = []
            dispatch['_ext'] = lambda d: Module(tuple(d), calls)
            normal_bank, special_bank = [object() for _ in range(3)], [object() for _ in range(3)]
            special = Module('wgs32', calls)
            state = {'lo': 0, 'cnt': 32, 'w': [], 'scratch': normal_bank, 'grid': 132, 'sfx': sfx,
                     'wgs32': {'ext': special, 'scratch': special_bank} if enabled and sfx else None}
            data = {'config': {'local_start': 0, 'local_count': 32}, 'hidden_states': Tensor((m, 4096)),
                    'topk_ids': Tensor((m, 8), 'i32', aligned=False), 'topk_weights': Tensor((m, 8), 'f32')}
            out = dispatch['kernel_fn'](data, state)
            use_special = enabled and bool(sfx) and 17 <= m <= 32
            assert len(calls) == (m + 383) // 384 and sum(c[2] for c in calls) == m
            assert all((c[0] == 'wgs32') is use_special and c[1] is (special_bank if use_special else normal_bank) for c in calls)
            assert all(c[3:] == ('bf16', 'i32', 'f32', 'bf16') for c in calls)
            assert out.shape == (m, 4096) and out.dtype == 'bf16'
            assert dispatch['kernel_fn'](data, state) is not out
            dispatch_cases += 1

# The actual prepare wrapper must add just one unique module and a new bank,
# or keep its original state when the budget/domain/hardware gate fails.
wrapper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare'][-1]
prepare_cases = []
for prmt, capability, elapsed in ((True, (9, 0), 100), (False, (9, 0), 100),
                                  (True, (8, 0), 100), (True, (9, 0), 236)):
    calls, modules = [], {}
    original_bank = [object(), object(), object()]
    def extension(defs):
        defs = tuple(defs)
        return modules.setdefault(defs, Module(defs, calls))
    times = iter((0, elapsed))
    ns = {'_wgs32_time': types.SimpleNamespace(monotonic=lambda: next(times)),
          '_wgs32_original_prepare': lambda w: {'sfx': ('-DOKC_PRMTSC=1',) if prmt else (), 'scratch': original_bank},
          '_main_defines': dispatch['_main_defines'], '_ext': extension,
          '_make_scratch': lambda dev: [object(), object(), object()],
          'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda dev: capability))}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<actual-prepare-wrapper>', 'exec'), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        state = ns['prepare']({'w13': types.SimpleNamespace(device='cuda:0')})
    expected = prmt and capability == (9, 0) and elapsed < 235
    assert (state['wgs32'] is not None) is expected
    if expected:
        assert len(modules) == 2 and sum('@wgs32' in key for key in modules) == 1
        assert all(a is not b for a, b in zip(state['wgs32']['scratch'], original_bank))
    else: assert not modules and state['scratch'] is original_bank
    prepare_cases.append({'prmt': prmt, 'capability': capability, 'elapsed': elapsed, 'enabled': expected})

# All references, including source factory aliases, must resolve normally.
symbols = symtable.symtable(draft, 'candidate', 'exec')
known = set(symbols.get_identifiers()) | set(dir(builtins))
def bindings(table):
    for name in table.get_identifiers():
        s = table.lookup(name)
        if s.is_global() and s.is_referenced(): assert name in known, (table.get_name(), name)
    for child in table.get_children(): bindings(child)
bindings(symbols)

# Real factory: one additional cache key/source and no extra weight repacking.
factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ext')
builds = []
def loader(**kwargs):
    assert kwargs['cpp_sources'] == literal['CPP_SRC']
    assert kwargs['functions'] == ['okc_run', 'okc_set_pdl', 'okc_resources']
    special = 'srcwgs32' in kwargs['name']
    assert kwargs['cuda_sources'] == (alternate if special else source)
    assert not any(flag.startswith('@') for flag in kwargs['extra_cuda_cflags'])
    builds.append(kwargs)
    return types.SimpleNamespace(okc_set_pdl=lambda n: None, okc_resources=lambda: [0] * 7)
factory_ns = dict(literal, ALT_SRCS={'wgs32': alternate}, _EXTS={}, load_inline=loader,
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))))
exec(compile(ast.Module(body=[factory], type_ignores=[]), '<real-module-factory>', 'exec'), factory_ns)
sfx = ('-DOKC_PRMTSC=1',)
variants = [tuple(dispatch[name]) + sfx for name in ('MAIN_DEFINES', 'TINY_DEFINES', 'TINY2_DEFINES', 'MID_DEFINES')]
variants.append(variants[0] + ('@wgs32',))
for definitions in variants:
    ext = factory_ns['_ext'](definitions)
    assert factory_ns['_ext'](definitions) is ext
assert len(builds) == len({kw['name'] for kw in builds}) == 5
assert sum('srcwgs32' in kw['name'] for kw in builds) == 1
result = {'source_sha256': record['source_sha256'], 'seed': 20261028,
          'original_python_ast_recovered': True, 'original_cuda_kernel_bytes_unchanged': True,
          'alternate_edits_reversible': True, 'wide_mma_variants_absent': True,
          'single_expert_counts_checked': 2 * 3073, 'route_distribution_cases': route_cases,
          'duplicate_routes_preserved': True, 'scratch_layouts': layouts,
          'scratch_allocated_elements': allocated, 'dispatch_cases': dispatch_cases,
          'prepare_wrapper_cases': prepare_cases, 'global_bindings': True,
          'real_factory_cache_and_source_isolation': True, 'unique_modules_with_specialization': 5,
          'resource_attributes': 'not measured; CPU protocol uses mocks', 'gpu_status': 'unmeasured'}
(root / '.tools/moe-wgs32-cpu-audit.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps(result))
