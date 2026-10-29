"""CPU control-flow test of the O128 wrapper (mocks only; no CUDA kernel is built, loaded or launched)."""
import ast
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import build_candidates as bc  # noqa: E402


def run(wrap_src, kind):
    calls = []
    ns = {'torch': torch, 'hashlib': __import__('hashlib'), 'load_inline': None}

    def base_prepare(weights):
        calls.append('base_prepare')
        return {'dense': {k: v for k, v in weights.items()}, 'achita_dense': {}}

    def base_kernel(data, state):
        calls.append(('base_kernel', data['x'].shape[0], data['config']['layer']))
        return 'BASE'

    def base_project(x, layer, state, rows=None):
        calls.append(('base_project', x.shape[0], layer, rows))
        return 'BASE'

    ns.update(prepare=base_prepare, kernel_fn=base_kernel, _achita_dense_project=base_project)
    exec(compile((HERE / 'o128_common.py').read_text(encoding='utf-8') + wrap_src, 'o128', 'exec'), ns)
    weights = {'qkv_gate': torch.zeros(2304, 64, dtype=torch.bfloat16), 'pooler': torch.zeros(4096, 64, dtype=torch.bfloat16),
               'unpooler': torch.zeros(8192, 32, dtype=torch.bfloat16)}
    out = {}
    state = ns['prepare'](weights)
    out['cpu_prepare_keeps_incumbent'] = state.get('o128') == {} and 'base_prepare' in calls
    x = torch.zeros(128, 64, dtype=torch.bfloat16)
    if kind == 'dense':
        r = ns['kernel_fn']({'x': x, 'config': {'m': 128, 'n': 4096, 'k': 64, 'layer': 'pooler'}}, state)
    else:
        r = ns['_achita_dense_project'](x, 'pooler', state)
    out['empty_choices_use_incumbent'] = r == 'BASE'
    # fake choice: dispatch only for the exact (M, layer, K) key and matching n
    launched = []
    ns['_o128_launch'] = lambda ext, i, pk, xx: launched.append((i, xx.shape[0])) or 'NEW'
    state['o128'] = {(128, 'pooler', 64): ('ext', 1, {'n': 4096})}
    probes = {
        'exact_key': (torch.zeros(128, 64, dtype=torch.bfloat16), 'pooler', None),
        'nearby_M129': (torch.zeros(129, 64, dtype=torch.bfloat16), 'pooler', None),
        'other_layer': (torch.zeros(128, 64, dtype=torch.bfloat16), 'qkv_gate', None),
        'non_contiguous': (torch.zeros(64, 128, dtype=torch.bfloat16).t(), 'pooler', None),
        'fp32_input': (torch.zeros(128, 64, dtype=torch.float32), 'pooler', None),
    }
    if kind == 'decoder':
        probes['sliced_rows'] = (torch.zeros(128, 64, dtype=torch.bfloat16), 'pooler', 1024)
    res = {}
    for name, (xx, layer, rows) in probes.items():
        if kind == 'dense':
            n = 4096 if layer == 'pooler' else 2304
            res[name] = ns['kernel_fn']({'x': xx, 'config': {'m': xx.shape[0], 'n': n, 'k': xx.shape[1], 'layer': layer}}, state)
        else:
            res[name] = ns['_achita_dense_project'](xx, layer, state, rows)
    out['dispatch'] = res
    out['new_path_only_on_exact_key'] = res['exact_key'] == 'NEW' and all(v == 'BASE' for k, v in res.items() if k != 'exact_key')
    return out


def main():
    r = {'dense': run(bc.DENSE_WRAP, 'dense'), 'decoder': run(bc.DECODER_WRAP, 'decoder'),
         'cuda_available_locally': torch.cuda.is_available()}
    p = ROOT / 'results/opus-fresh-investigation-20261029/o128_control_flow_test.json'
    p.write_text(json.dumps(r, indent=2), encoding='utf-8')
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
