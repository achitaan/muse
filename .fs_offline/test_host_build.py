"""CPU-only host build contract regression; no extension build or GPU execution."""
import ast, base64, contextlib, hashlib, importlib.util, io, json, re, subprocess, sys, types, zlib
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.fs_offline/host_check'
OUT.mkdir(exist_ok=True)
def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
fs = load(ROOT/'experiments/boards/moe/moe_fs1.py', 'fs_host')
diag = load(ROOT/'experiments/boards/diag/fsdiag.py', 'diag_host')
# Both log paths must expose the known failure rather than the ninja tail.
known = '/torch/include/ATen/ATen.h:5:2: error: #error C++20 or later compatible compiler is required to use ATen'
exc = RuntimeError("Error building extension 'test':\n[1/2] nvcc -std=c++17\n"+known+'\n'+
                   'binding.cpp -o binding.o\nninja: build stopped: subcommand failed.\n'*200)
for fmt in (fs._fs_error_lines, diag._fs_error_lines):
    assert fmt(exc) == known
    assert len(fmt(RuntimeError('file.cpp: error: '+'x'*2000))) == 600
    assert fmt(RuntimeError('ordinary exception')) == 'no compiler error lines captured'
    assert fmt(RuntimeError('x.cpp: ERROR: first\ny.cpp: error: second')) == 'x.cpp: ERROR: first | y.cpp: error: second'
buf = io.StringIO()
with contextlib.redirect_stdout(buf):diag.error('build', exc)
assert known in buf.getvalue() and 'ninja:' not in buf.getvalue()
with patch.object(fs, '_fs_parent_prepare', lambda w: {}), patch.object(fs, '_fs_state', side_effect=exc):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):state = fs.prepare({})
assert state['fs'] is None and known in buf.getvalue() and 'binding.cpp -o' not in buf.getvalue()

captured = {}
def inline(**kw):
    captured.update(kw)
    return types.SimpleNamespace(okc_set_pdl=lambda x: None)
with patch.object(fs, 'load_inline', inline):fs._fs_ext()
assert not any(x.startswith(('-std=', '--std=')) for x in captured['extra_cuda_cflags'])
assert captured['cpp_sources'] == fs.FS_CPP
assert 'PYBIND11_MODULE' not in fs.FS_CPP
assert '#include <torch/extension.h>' in fs.FS_CPP
for bad in ('cudaStream_t', 'CUtensorMap', '__nv_bfloat16', '__global__'):
    assert bad not in fs.FS_CPP
for f in captured['functions']:
    # Exactly one declaration in host source and one definition in CUDA source.
    pat = r'(?:void|std::vector<int64_t>)\s+'+f+r'\('
    assert len(re.findall(pat, fs.FS_CPP)) == len(re.findall(pat, fs.FS_CUDA)) == 1
main = fs.FS_CPP+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){\n'+''.join(
    f'm.def("{f}", torch::wrap_pybind_function({f}));\n' for f in captured['functions'])+'}\n'
(OUT/'main.cpp').write_text(main)
def getdir(name, verbose=False):return str(OUT)
def bundle_loader(**kw):
    assert not any(x.startswith(('-std=', '--std=')) for x in kw['extra_cuda_cflags'])
    assert len(kw['sources']) == 4
    return kw
with patch('torch.utils.cpp_extension._get_build_directory', getdir), patch('torch.utils.cpp_extension.load', bundle_loader):
    diag.build_bundle(vars(fs))
cpp = (OUT/'binding.cpp').read_text()
assert cpp.count('PYBIND11_MODULE') == 1
assert zlib.decompress(base64.b85decode(diag.CANDIDATE_B85)) == (ROOT/'experiments/boards/moe/moe_fs1.py').read_bytes()
# Mirror host declarations/signatures and generated binding syntax with a minimal
# header. This is NOT a Torch ABI/header compatibility test or a linked build.
header = OUT/'stub/torch/extension.h'; header.parent.mkdir(parents=True, exist_ok=True)
header.write_text('''#pragma once
#include <cstdint>
#include <vector>
namespace torch {
struct Tensor {};
template<class R,class... A> auto wrap_pybind_function(R(*f)(A...)) {return f;}
}
struct Module {template<class R,class... A> void def(const char*,R(*)(A...)) {}};
#define PYBIND11_MODULE(name,m) void test_module(Module &m)
''')
compiler = Path('C:/msys64/ucrt64/bin/g++.exe')
if compiler.exists():
    for file in ('main.cpp', 'binding.cpp'):
        cmd = [str(compiler), '-std=c++20', '-fsyntax-only', '-I'+str(OUT/'stub'), str(OUT/file)]
        p = subprocess.run(cmd, text=True, capture_output=True)
        (OUT/(file+'.log')).write_text(p.stdout+p.stderr)
        assert p.returncode == 0, p.stdout+p.stderr
    print('PASS g++ C++20 -fsyntax-only main.cpp/binding.cpp with stub headers (not real Torch headers)')
else:
    print('SKIP host syntax compile: modern g++ absent; declarations reviewed/tested structurally')
print('PASS inherited-standard flags, host exports, bounded actual error logging, fallback, and embedding')
