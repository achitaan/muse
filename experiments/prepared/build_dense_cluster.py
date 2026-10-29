"""Extract two pinned plain-BF16 cluster GEMMs; no GPU evaluation is launched."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = 'edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b'
UPSTREAM = 'c1dd241c0c08b42fddfa7bc9656d0f596b59bb13005c50c0db4418d9f2f1dc7e'


def build():
    base = (ROOT / '.scored-measurements' / PARENT / 'submission.py').read_bytes()
    assert hashlib.sha256(base).hexdigest() == PARENT
    raw = (ROOT / 'results/dense-pinned-codec-screen-20261028/upstream_dense.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == UPSTREAM
    tree = ast.parse(raw)
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for x in n.targets:
                if isinstance(x, ast.Name):
                    try:
                        constants[x.id] = ast.literal_eval(n.value)
                    except (ValueError, TypeError):
                        pass
    assert constants['GQ_CONFIGS'][('pooler', 128)] == (1, 2, 8)
    assert constants['GQX_CONFIGS'][('qkv_gate', 128)] == (1, 3, 8)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'wgmma_ss_defs')
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pinned-wgmma-defs>', 'exec'), scope)
    gq_dispatch = 'if (bn == 128 && nwg == 1 && cl == 2 && nstage == 8) { launch_gq<128, 1, 8, 2>(W, x, out, M, N, K); return; }'
    gqx_dispatch = 'if (bn == 128 && nwg == 1 && cl == 3 && nstage == 8) { launch_gqx<128, 1, 8, 3>(W, x, out, M, N, K); return; }'
    assert constants['GQ_SRC'].count('@@GQ_DISPATCH@@') == 1
    assert constants['GQX_SRC'].count('@@GQX_DISPATCH@@') == 1
    header = '''#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <type_traits>
'''
    pieces = dict(header=header, mbar=constants['MBAR_HELPERS'], warpgroup=constants['WG_HELPERS'],
                  maps=constants['MAP_SRC'], mma=scope['wgmma_ss_defs']([128]),
                  gq=constants['GQ_SRC'].replace('@@GQ_DISPATCH@@', gq_dispatch),
                  gqx=constants['GQX_SRC'].replace('@@GQX_DISPATCH@@', gqx_dispatch))
    cuda = '\n'.join(pieces.values())
    assert '@@' not in cuda
    cpp = '''#include <torch/extension.h>
void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K,
        int64_t bn, int64_t nwg, int64_t cl, int64_t nstage);
void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K,
         int64_t bn, int64_t nwg, int64_t cl, int64_t nstage);
'''
    setup = "\n_LG_CPP = " + repr(cpp) + "\n_LG_CUDA = " + repr(cuda) + '\n'
    license_text = (ROOT / 'licenses/vjeux-Apache-2.0.txt').read_text(encoding='utf8')
    notice = ('\n# Plain BF16 cluster GEMMs adapted from Vjeux okc-2026, Apache-2.0,\n'
              '# f35a41ee1c13e316dabbf772830d47b23b453393/muse_bf16_proj.py.\n'
              '# Local changes isolate two curated configurations and add bounded selection.\n' +
              ''.join('# ' + l + '\n' for l in license_text.splitlines()))
    addon = (ROOT / 'experiments/prepared/dense_cluster_addon.py').read_text(encoding='utf8')
    output = base + (notice + setup + addon).encode('utf8')
    assert output[:len(base)] == base and len(output) < 1_000_000
    base_tree = ast.parse(base)
    aliases = {x.id: ast.unparse(n.value) for n in base_tree.body if isinstance(n, ast.Assign)
               for x in n.targets if isinstance(x, ast.Name)}
    assert aliases['_achita_wg_base_prepare'] == 'prepare' and aliases['_achita_wg_base_kernel'] == 'kernel_fn'
    compile(output, '<dense-cluster>', 'exec')
    (ROOT / '.tools/dense-cluster.py').write_bytes(output)
    (ROOT / '.tools/dense-cluster.cu').write_bytes(cuda.encode())
    report = dict(parent_sha256=PARENT, source_sha256=hashlib.sha256(output).hexdigest(),
                  upstream_sha256=UPSTREAM, bytes=len(output), original_source_is_exact_prefix=True,
                  cuda_piece_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in pieces.items()},
                  configs={'qkv_gate': {'kind': 'gqx', 'bn': 128, 'nwg': 1, 'stages': 8, 'cluster': 3, 'm': [33, 128]},
                           'pooler': {'kind': 'gq', 'bn': 128, 'nwg': 1, 'stages': 8, 'cluster': 2, 'm': [65, 128]}},
                  weights='original BF16, no compression/requantization/new weight copies',
                  router_and_all_small_paths='original incumbents',
                  removed_preparation='only the unselected final WG sweep, via existing base aliases',
                  gpu_status='unmeasured; require positive official check quota before launch')
    (ROOT / '.tools/dense-cluster-source-audit.json').write_bytes(json.dumps(report, indent=2).encode())
    print(json.dumps(report))


if __name__ == '__main__':
    build()
