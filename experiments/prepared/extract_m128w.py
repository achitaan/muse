import ast, zlib, base64, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
src = (ROOT / 'best/verified.py').read_text(encoding='utf-8')
tree = ast.parse(src)
ns = {'_zsrc': lambda s: zlib.decompress(base64.b64decode(s)).decode()}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        k = node.targets[0].id
        if k in ('dd_decodediet_tt_mw_CUDA_PREFIX', 'dd_decodediet_tt_mw_CUDA_CORE'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<ast>', 'exec'), ns)
    elif isinstance(node, ast.FunctionDef) and node.name == 'dd_decodediet_tt_mw_mma':
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<ast>', 'exec'), ns)

mma = ns['dd_decodediet_tt_mw_mma']
core = ns['dd_decodediet_tt_mw_CUDA_CORE'].replace(
    'if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));',
    'if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));'
)
cu = ns['dd_decodediet_tt_mw_CUDA_PREFIX'] + '\n'.join(mma(n) for n in (16, 32, 64, 128)) + core
out_dir = ROOT / 'results/m128w-nvrtc-screen-20261029'
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'test_m128w.cu').write_text(cu, encoding='utf-8')
print('Wrote', out_dir / 'test_m128w.cu', len(cu))
