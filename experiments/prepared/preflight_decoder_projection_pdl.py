"""Compile-only screen of three selected O preambles; never loads a GPU kernel."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
BASE = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
CONDITIONAL = '''namespace std {
template<bool B, class T, class F> struct conditional { using type = T; };
template<class T, class F> struct conditional<false, T, F> { using type = F; };
}
'''
WAIT = '\n    asm volatile("griddepcontrol.wait;" ::: "memory");'


def literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(literal(n) for n in node.elts)
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name) and node.func.id == '_zsrc'
        assert len(node.args) == 1 and not node.keywords
        return zlib.decompress(base64.b64decode(literal(node.args[0]))).decode()
    raise ValueError(ast.dump(node)[:200])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ordered-weight-loads', action='store_true')
    args = parser.parse_args()
    raw = (ROOT / 'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    values = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values.setdefault(target.id, []).append(node.value)

    def get(name):
        assert len(values[name]) == 1
        return literal(values[name][0])

    out = ROOT / ('.tools/decoder-pdl-preflight-v2-ordered' if args.ordered_weight_loads
                  else '.tools/decoder-pdl-preflight-v1')
    out.mkdir(exist_ok=True)
    specs = [
        ('o1', get('tt_v10_sm2_SMALLM2_CUDA_TEMPLATE'), 'smallm2', get('V12_SM2_O1')['o', 1][0]),
        ('o8', get('V18S_CUDA_TEMPLATE'), 'v18s', get('V18_SMALL')['o', 8][0]),
        ('o16', get('mt_M16TRACE_SOURCES')['o'][1], 'm16trace_59fcb5b72994', get('mt_M16TRACE_ROUTES')['o']['inst']),
    ]
    manifest = dict(baseline_sha256=BASE, GPU_jobs=0, candidate_installed=False,
                    proxy_adaptations=['Drop unused algorithm header and host launcher',
                                       'Supply minimal std::conditional instead of host type_traits',
                                       'NVRTC virtual integral_constant/stdint headers from existing local helper'],
                    variants=[])
    for label, cuda, symbol, inst in specs:
        cutoff = f'static int {symbol}_smem_optin()'
        assert cuda.count(cutoff) == 1
        prefix = cuda.split(cutoff)[0]
        assert prefix.count('#include <algorithm>') == 1
        assert prefix.count('#include <type_traits>') == 1
        proxy = prefix.replace('#include <algorithm>', '').replace('#include <type_traits>', CONDITIONAL)
        anchor = 'for (int s = 0; s < ST; ++s) load(s, s);'
        assert proxy.count(anchor) == 1
        # Only the three selected direct-activation paths are compiled here.
        if label == 'o16':
            assert inst[14] == 7 and inst[15] == 1
        else:
            assert inst[14] & 2
        coherent = 'ld.global.v4.u32'
        assert proxy.count('ld.global.nc.v4.u32') == 1
        changed = proxy.replace(anchor, anchor + WAIT).replace('ld.global.nc.v4.u32', coherent)
        # Exact inverse: no arithmetic, store, launch geometry or weight loader changes.
        assert changed.replace(WAIT, '').replace(coherent, 'ld.global.nc.v4.u32') == proxy
        if args.ordered_weight_loads:
            # v1 emitted the wait before all weight loads. Give the compiler
            # explicit memory effects for the independent inline-PTX loads.
            start = changed.index(f'int4 {symbol}_ldw(')
            end = changed.index('return v;', start)
            original = changed[start:end]
            assert original.count(': "l"(p));') == 1
            ordered = original.replace(': "l"(p));', ': "l"(p) : "memory");')
            changed = changed[:start] + ordered + changed[end:]
            assert changed[:start] + original + changed[start + len(ordered):] == proxy.replace(anchor, anchor + WAIT).replace('ld.global.nc.v4.u32', coherent)
        for arm, source in [('baseline', proxy), ('wait_coherent', changed)]:
            path = out / f'{label}-{arm}.cu'
            data = source.encode()
            assert not path.exists() or path.read_bytes() == data, 'Frozen preflight source collision'
            path.write_bytes(data)
            expressions = []
            for fmt in (inst[0], 0):
                concrete = (fmt,) + inst[1:]
                expressions.append(symbol + '_gemv<' + ','.join(map(str, concrete)) + '>')
            result = out / f'{label}-{arm}.json'
            if not result.exists():
                command = [sys.executable, str(ROOT / 'experiments/prepared/local_nvrtc_compile.py'),
                           '--source', str(path), '--minimal-std-tags', '--out', str(path.with_suffix(''))]
                for expression in expressions:
                    command += ['--name', expression]
                subprocess.run(command, cwd=ROOT, check=True)
            report = json.loads(result.read_text())
            assert report['return_code'] == 0
            assert report['original_source_sha256'] == hashlib.sha256(data).hexdigest()
            manifest['variants'].append(dict(label=label, arm=arm,
                                             exact_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                                             source_sha256=hashlib.sha256(data).hexdigest(),
                                             instances=expressions,
                                             log=path.with_suffix('.log').read_text(),
                                             cubin_bytes=report['cubin_bytes']))
    destination = out / 'manifest.json'
    if args.ordered_weight_loads:
        manifest['candidate_weight_load_memory_clobber'] = True
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    assert not destination.exists() or destination.read_bytes() == payload
    destination.write_bytes(payload)
    print(json.dumps(dict(variants=len(manifest['variants']), GPU_jobs=0, candidate_installed=False)))


if __name__ == '__main__':
    main()
