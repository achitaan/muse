"""CPU-only dispatch, chunk, resource and evidence admission tests."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
addon = ROOT / 'experiments/prepared/moe_factor_y_addon.py'
tree = ast.parse(addon.read_bytes())
for n in tree.body:
    if isinstance(n, ast.Assign):
        assert isinstance(n.value, (ast.Name, ast.Constant, ast.Tuple))
    elif isinstance(n, ast.FunctionDef):
        assert not n.decorator_list and all(isinstance(v, ast.Constant) for v in n.args.defaults)
    else:
        assert isinstance(n, (ast.Expr, ast.Import, ast.ImportFrom))
ns = runpy.run_path(str(addon), init_globals=dict(prepare=None, kernel_fn=lambda d, s: 'original', torch=torch,
                                                HIDDEN=4096, EMAX=3072, CHUNK_TOKENS=384, MID_MAX_M=16))
resources = [96, 96, 0, 0, 0, 0, 231168, 1, 1, 576]
assert ns['_factor_resources_ok'](resources, 132)
resource_rejections = []
for name, index, value in [('register_budget', 0, 113), ('local_storage', 2, 8),
                           ('shared_size', 6, 231296), ('residency_false', 7, 0),
                           ('residency_true', 8, 0), ('thread_geometry', 9, 320)]:
    bad = resources.copy(); bad[index] = value
    assert not ns['_factor_resources_ok'](bad, 132)
    resource_rejections.append(name)
assert not ns['_factor_resources_ok'](resources, 161)

cases = []
for m, mode in [(m, 'normal') for m in ns['_FACTOR_SHAPES']] + [(128, 'skewed')]:
    old = [100000.123456789 + i * 64 for i in range(10)]
    new = [v * .95 for v in old]
    cases.append(dict(m=m, mode=mode, synthetic=True,
                      seed=20261030 + m + (10000 if mode == 'skewed' else 0),
                      old_ns=old, new_ns=new, arm_orders=[[0, 1]] * 5 + [[1, 0]] * 5,
                      median_ratio=.95, relative=0., worst=0.,
                      changed_input_graph_replays=[[0., 0.], [0., 0.]] if m == 32 else []))
analytic = [dict(m=m, mode='remote', passed=True) for m in (17, 33, 384, 385)]
analytic.append(dict(m=33, mode='duplicate', passed=True, relative=0., worst=0.))
assert ns['_factor_decision'](cases, resources, 132, analytic)
assert ns['_factor_decision'](json.loads(json.dumps(cases)), resources, 132, analytic)
mutations = []
bad = copy.deepcopy(cases); bad.pop(); mutations.append(('missing_skewed', bad))
bad = copy.deepcopy(cases); bad[3] = copy.deepcopy(bad[2]); mutations.append(('duplicate_shape', bad))
bad = copy.deepcopy(cases); bad[0]['arm_orders'][0] = [1, 0]; mutations.append(('unbalanced_orders', bad))
bad = copy.deepcopy(cases); bad[2]['new_ns'][0] = float('nan'); mutations.append(('nan_timing', bad))
bad = copy.deepcopy(cases); bad[2]['new_ns'][0] = 0; mutations.append(('zero_timing', bad))
bad = copy.deepcopy(cases); bad[0]['median_ratio'] = .9; mutations.append(('forged_ratio', bad))
bad = copy.deepcopy(cases); bad[2]['relative'] = .006; mutations.append(('numerical_error', bad))
bad = copy.deepcopy(cases); bad[2]['changed_input_graph_replays'] = []; mutations.append(('missing_fresh_replay', bad))
bad = copy.deepcopy(cases); bad[2]['changed_input_graph_replays'][1][1] = .03; mutations.append(('fresh_replay_error', bad))
bad = copy.deepcopy(cases); bad[0]['new_ns'] = [v * 1.021 for v in bad[0]['old_ns']]; bad[0]['median_ratio'] = 1.021
mutations.append(('nearby_regression', bad))
bad = copy.deepcopy(cases); bad[2]['new_ns'] = [v * .97 for v in bad[2]['old_ns']]; bad[2]['median_ratio'] = .97
mutations.append(('material_gain_missing', bad))
for name, bad in mutations:
    assert not ns['_factor_decision'](bad, resources, 132, analytic), name


class Module:
    def __init__(self):
        self.calls = []

    def okc_run(self, w, scratch, x, ids, weights, out, lo, count, grid):
        assert x.is_contiguous() and ids.is_contiguous() and ids.data_ptr() % 16 == 0
        assert ids.dtype == torch.int32 and weights.dtype == torch.float32
        assert out.shape == x.shape and out.dtype == torch.bfloat16
        assert (lo, count, grid) == (0, 32, 132)
        self.calls.append((x.shape[0], id(scratch)))
        out.fill_(1)


dispatch, chunks = [], []
for m in (0, 1, 4, 8, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 383, 384, 385, 768, 769):
    module = Module()
    ids_storage = torch.zeros(m * 8 + 1, dtype=torch.int32)
    data = dict(hidden_states=torch.zeros((m, 4096), dtype=torch.bfloat16),
                topk_ids=ids_storage[1:].reshape(m, 8),
                topk_weights=torch.ones((m, 8), dtype=torch.float64),
                config=dict(local_start=0, local_count=32))
    scratch = []
    state = dict(w=[], lo=0, cnt=32, grid=132, sfx=('-DOKC_PRMTSC=1',),
                 _factor_module=module, _factor_scratch=scratch)
    out = ns['_factor_run'](data, state, module, scratch)
    expected = [min(384, m - first) for first in range(0, m, 384)]
    assert [c[0] for c in module.calls] == expected
    assert all(c[1] == id(scratch) for c in module.calls)
    assert out.shape == (m, 4096) and bool((out == 1).all())
    chunks.append(dict(m=m, launches=expected))
    for enabled in (False, True):
        for eligible in (False, True):
            state['_factor_enabled'] = enabled
            state['sfx'] = ('-DOKC_PRMTSC=1',) if eligible else ()
            before = len(module.calls)
            got = ns['kernel_fn'](data, state)
            selected = enabled and eligible and m > 16
            assert (not isinstance(got, str)) == selected
            assert len(module.calls) - before == (len(expected) if selected else 0)
            dispatch.append((m, enabled, eligible, selected))
    state['_factor_enabled'] = True; state['sfx'] = ('-DOKC_PRMTSC=1',)
    a, b = ns['kernel_fn'](data, state), ns['kernel_fn'](data, state)
    if m > 16:
        assert a is not b and (m == 0 or a.data_ptr() != b.data_ptr())

metadata = []
for mode in ('normal', 'skewed', 'duplicate', 'remote'):
    data = ns['_factor_input'](33, torch.device('cpu'), mode)
    assert data['hidden_states'].dtype == torch.bfloat16 and data['hidden_states'].shape == (33, 4096)
    assert data['topk_ids'].dtype == torch.int32 and data['topk_ids'].shape == (33, 8)
    assert torch.isfinite(data['topk_weights']).all()
    assert torch.allclose(data['topk_weights'].sum(1), torch.ones(33), atol=1e-6)
    if mode == 'remote':
        assert not bool((data['topk_ids'] < 32).any())
    elif mode == 'duplicate':
        assert not bool(torch.count_nonzero(data['topk_ids']))
    metadata.append(mode)

evidence = dict(cuda_sha256=json.loads((ROOT / '.tools/moe-factor-y-source-audit.json').read_bytes())['factor_cuda_sha256'],
                synthetic_only=True, independent_jobs=1, repeats_are_not_independent=True,
                selected_main=True, resources=resources, grid=132, cases=cases, analytic=analytic,
                extra_weight_layouts=0, separate_scratch=True, original_small_and_ineligible_scale_paths=True,
                prepare_s=240.123456789, software=dict(torch='2.14.1+cu130', cuda='13.0', device='NVIDIA H200'))
encoded_size = len(json.dumps(evidence, separators=(',', ':')))
assert encoded_size <= 13000
report = dict(source_sha256=hashlib.sha256((ROOT / '.tools/moe-factor-y.py').read_bytes()).hexdigest(),
              cuda_sha256=evidence['cuda_sha256'], cpu_only=True, no_cuda_calls=True,
              dispatch_cases=len(dispatch), chunk_cases=chunks, fresh_outputs=True,
              id_alignment_and_dtype_preserved=True, separate_scratch_binding=True,
              synthetic_input_modes=metadata, resource_rejections=resource_rejections,
              selection_rejections=[name for name, _ in mutations], json_roundtrip_admission=True,
              evidence_fixture_bytes=encoded_size, evidence_limit=13000,
              original_prepare_and_original_kernel_not_executed=True, gpu_verified=False)
(ROOT / '.tools/moe-factor-y-host-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
print(json.dumps({k: v for k, v in report.items() if k != 'chunk_cases'}, indent=2))
