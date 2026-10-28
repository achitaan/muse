"""Prepare one fused pooler/router trial from the pinned public component."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
upstream = (root / '.tools/vjeux-decoder-upstream.py').read_text()
tree = ast.parse(upstream)
values = {}
functions = {}
for n in tree.body:
    if isinstance(n, ast.FunctionDef): functions[n.name] = n
    if isinstance(n, ast.Assign):
        for a in n.targets:
            if isinstance(a, ast.Name):
                try: values[a.id] = ast.literal_eval(n.value)
                except (ValueError, TypeError): pass

cfg = next(x for x in values['bfp_SK2_CONFIGS'] if x[0] == 'pooler')
assert cfg == ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1)
scope = {'bfp_SK2_K': {'pooler': 8192}}
exec(compile(ast.Module(body=[functions['bfp__sk2_geom']], type_ignores=[]), '<audited-geometry>', 'exec'), scope)
dispatch = []
geometries = {}
for tokr in [16, 32]:
    hl, stages = scope['bfp__sk2_geom'](cfg, tokr)
    assert not hl
    mt = 2 if tokr == 32 else 1
    nacc = 1 if mt == 2 and stages >= 4 else 2
    geometries[tokr] = stages
    dispatch.append(f'''if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, 4, 1, 8, 1, {stages}, true, false,
        {nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr,
        reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(),
        rcnt.data_ptr<int>()); return; }}''')
cuda = (values['bfp_CUDA_SRC'].replace('@@MBAR@@', values['bfp_MBAR_HELPERS']).replace('@@BODY@@', '')
        + values['bfp_MAP_SRC']
        + values['bfp_SK2_SRC'].replace('@@SK2_DISPATCH@@', '')
          .replace('@@SK2F_DISPATCH@@', '\n'.join(dispatch)))
cuda = '#include <type_traits>\n' + cuda
assert '@@' not in cuda

keep = ['bfp_sk2_tileunits', 'bfp_sk2_compress', 'bfp_sk2_stream', 'bfp_sk2_table',
        'bfp__sk2_tokr', 'bfp_fused_pooler_router']
names = {name: '_fpr_' + name for name in keep}
names.update({'bfp__get_ext': '_fpr_get_ext', 'bfp_SK2_PDL': '_fpr_PDL'})
class Rename(ast.NodeTransformer):
    def visit_Name(self, n):
        n.id = names.get(n.id, n.id)
        return n
    def visit_FunctionDef(self, n):
        n.name = names.get(n.name, n.name)
        return self.generic_visit(n)
    def visit_Global(self, n):
        n.names = [names.get(x, x) for x in n.names]
        return n
helper_tree = Rename().visit(ast.Module(body=[functions[name] for name in keep], type_ignores=[]))
ast.fix_missing_locations(helper_tree)
helpers = ast.unparse(helper_tree)

setup = r'''
from torch.utils.cpp_extension import load_inline as _fpr_load_inline
_fpr_ext = None
_fpr_PDL = 0

def _fpr_get_ext():
    global _fpr_ext
    if _fpr_ext is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f'{major}{minor}a'
        _fpr_ext = _fpr_load_inline(
            name='okc_fpr_HASH_' + arch,
            cpp_sources=_fpr_CPP,
            cuda_sources=_fpr_CUDA, functions=['sk2f', 'sk2_set_pdl'],
            extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}'],
            no_implicit_headers=True, verbose=False)
        _fpr_ext.sk2_set_pdl(0)
    return _fpr_ext

def _fpr_prepare(weights):
    if torch.cuda.get_device_capability()[0] != 9:
        return None
    w = weights['pooler'].contiguous()
    bits = w.view(torch.int16)
    if bool((bits == -32768).any()) or not bool(torch.isfinite(w).all()):
        return None  # strict lossless representation, including the sign of zero
    comp = _fpr_bfp_sk2_compress(w)
    if comp is None:
        return None
    tab, pmax = _fpr_bfp_sk2_table(comp, 4096, 32)
    # Match launch_sk2's exact shared-memory formula, including the actual
    # escape count. Unusual weight distributions fall back before compilation.
    for tokr, nstage in _fpr_GEOMETRIES.items():
        toks = 32 if tokr == 32 else 16
        sst = ((8 * tokr * 128 + 8 * 4 * 768) + 1023) // 1024 * 1024
        nesc = min(nstage, 4)
        smem = (1024 + nstage * sst + 16 * nstage + 32 +
                nesc * ((toks * 33 + 3) // 4 * 4) * 4 + pmax * 8 + 128)
        if smem > 232448:  # Hopper SM90 opt-in limit; matches upstream geometry
            return None
    ps = _fpr_bfp_sk2_stream(comp['packed'], 4096, 32, 4, 1, 8, 1)
    router = weights['router']
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    T, U = 32, 128
    tub = torch.cat([
        _fpr_bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
        _fpr_bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
    rs = _fpr_bfp_sk2_stream(tub, 256, 8, 1, 1, 8, 1)
    state = {'sm90': True, 'sk2': {
        'pooler': (0, ps, tab, pmax, comp['base4'], 4096),
        'router': (0, rs, None, 0, 0, 256)},
        'fused_part': torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=w.device),
        'fused_cnt': torch.zeros(32, dtype=torch.int32, device=w.device)}
    _fpr_get_ext()
    return state

_fpr_original_prepare = prepare
def prepare(weights):
    state = _fpr_original_prepare(weights)
    state['fpr'] = _fpr_prepare(weights)
    return state
'''
setup = setup.replace('okc_fpr_HASH_', 'okc_fpr_' + hashlib.sha256(cuda.encode()).hexdigest()[:16] + '_')

decoder = (root / 'best/decoder_verified.py').read_text()
start = '    side_stream.wait_stream(main_stream)\n'
end = '    # This GPU\'s experts start at local_start:'
assert decoder.count(start) == 1 and decoder.count(end) == 1
i, j = decoder.index(start), decoder.index(end)
old = decoder[i:j]
branch = '''    if 8 < tokens <= 32 and state['fpr'] is not None:
        expert_input, logits = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    else:
'''
decoder = decoder[:i] + branch + ''.join('    ' + line if line.strip() else line for line in old.splitlines(keepends=True)) + decoder[j:]
first, rest = decoder.split('\n', 1)
notice = ('# Additional fused pooler/router from Vjeux, Apache-2.0, revision\n'
          '# f35a41ee1c13e316dabbf772830d47b23b453393. Local changes extract only\n'
          '# the 9..32-token fused path, validate lossless fallback/scratch bounds,\n'
          '# allocate control buffers eagerly and preserve the FP32 small-T router.\n')
combined = (first + '\n' + notice + rest + '\n\n_fpr_CUDA = ' + repr(cuda) +
            '\n_fpr_CPP = ' + repr('#include <torch/extension.h>\n' + values['bfp_CPP_SRC']) +
            '\n_fpr_GEOMETRIES = ' + repr(geometries) + '\n' + helpers + setup)
compile(combined, '<decoder-fused-pr>', 'exec')
assert 'F.linear(router_input, state["router"])' in combined
out = root / '.tools/decoder-fused-pr-variant.py'
out.write_bytes(combined.encode())
(root / '.tools/decoder-fused-pr-cuda.cu').write_text(cuda)
audit = {'source_sha256': hashlib.sha256(combined.encode()).hexdigest(),
         'base_sha256': hashlib.sha256((root / 'best/decoder_verified.py').read_bytes()).hexdigest(),
         'cuda_sha256': hashlib.sha256(cuda.encode()).hexdigest(),
         'geometries': geometries, 'enabled_tokens': '9..32, public shape dispatch',
         'small_token_fp32_router': 'original full path preserved',
         'workspace': 'eager allocation; sequential replay', 'gpu_status': 'unmeasured'}
(root / '.tools/decoder-fused-pr-source-audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit))
