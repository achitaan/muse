"""Exercise actual payload dispatch/resource guards without GPU execution."""
import ast
import contextlib
import hashlib
import io
import json
import sys
from types import SimpleNamespace

from build_wgs_final_reuse_payload import ROOT, OUT, HERE, sources


def main():
    parent, _, cuda, cpp, vals = sources()
    raw = (HERE / 'moe_wgs_final_reuse.py').read_bytes()
    binding = json.loads((OUT / 'payload-binding.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == binding['source_sha256']
    assert raw.startswith(parent.rstrip(b'\n')) and len(raw) < 1_000_000
    base, tree = ast.parse(parent), ast.parse(raw)
    assert [ast.dump(n) for n in tree.body[:len(base.body)]] == [ast.dump(n) for n in base.body]
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') in ('WR_CUDA', 'WR_CPP')}
    assert literals == dict(WR_CUDA=cuda, WR_CPP=cpp)
    def last(name):
        return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    assert '_wr_build' not in ast.unparse(last('kernel_fn'))
    assert '_make_scratch' not in ast.unparse(last('kernel_fn'))
    env = dict(CHUNK_TOKENS=384, _wr_call=lambda d, s, m: ('candidate', m),
               _wr_parent_kernel=lambda d, s: 'parent')
    exec(compile(ast.Module(body=[last('kernel_fn')], type_ignores=[]), '<dispatch>', 'exec'), env)
    cases = 0
    for enabled in (False, True):
        for m in [*range(386), 1000]:
            data = dict(hidden_states=SimpleNamespace(shape=(m, 4096)))
            got = env['kernel_fn'](data, dict(wr_accept=enabled, wr_module='owned'))
            expected = ('candidate', 'owned') if enabled and 17 <= m <= 384 else 'parent'
            assert got == expected
            cases += 1
    adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_WRAdapter')
    env = dict(MAIN_DEFINES=vals['MAIN_DEFINES'], DQ7=vals['DQ7'], MAIN_NS=3,
               WR_CPP=cpp, WR_CUDA=cuda, PDL=False)
    exec(compile(ast.Module(body=[adapter, last('_wr_build')], type_ignores=[]), '<guard>', 'exec'), env)
    key = 'torch.utils.cpp_extension'
    previous = sys.modules.get(key)
    resource_cases = 0
    try:
        good = [96, 16, 0, 1, 200000, 576] * 2
        for index, value, valid in [(0, 96, True), (0, 0, False), (0, 97, False),
                                    (1, 17, False), (2, 1, False), (3, 0, False),
                                    (4, 0, False), (5, 512, False), (6, 97, False), (7, 17, False)]:
            metadata = list(good)
            metadata[index] = value
            calls = []
            module = SimpleNamespace(wr_resources=lambda: metadata, wr_run='native', wr_set_pdl=calls.append)
            sys.modules[key] = SimpleNamespace(load_inline=lambda **kwargs: module)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    wrapped = env['_wr_build'](dict(sfx=('-DOKC_PRMTSC=1',)))
                passed = wrapped.okc_run == 'native' and calls == [0]
            except AssertionError:
                passed = False
                assert not calls
            assert passed == valid
            resource_cases += 1
    finally:
        if previous is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = previous
    report = dict(source_sha256=binding['source_sha256'], parent_prefix_exact=True,
        native_literals_exact=True, dispatch_cases=cases, resource_guard_cases=resource_cases,
        host_only=True, GPU_jobs=0, admitted_for_GPU=False,
        limitation='Mocks establish Python dispatch and rejection ordering only; native host build, attributes, bytes and latency require H200.')
    (OUT / 'payload-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
