"""Source isolation, unchanged device prefix, and fail-closed admission audit."""

import ast
import base64
import hashlib
import json
import re
import zlib

from pdl16_sources import HERE, ROOT, literal
from build_norm_pair_diag import DENSE_SHA, extract
from build_o32_resource_diag import OUT, QUERY


def main():
    raw = (HERE / 'dense_o32_resource_diag.py').read_bytes()
    dense = (ROOT / 'best/verified.py').read_bytes()
    binding = json.loads((OUT / 'source-binding.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest() == binding['source_sha256']
    assert hashlib.sha256(dense).hexdigest() == DENSE_SHA and raw.startswith(dense)
    app = raw[len(dense):].decode()
    nodes = ast.parse(app).body
    assert [n.name for n in nodes if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn'] == []
    old = (OUT / 'legacy-device-and-host-prefix.cu').read_bytes()
    cuda = (OUT / 'diagnostic.cu').read_bytes()
    assert cuda.startswith(old) and cuda.count(b'void pdlo32_projection(') == 1
    private = (OUT / 'private-host.cu').read_text(encoding='utf-8')
    assert 'int o32r_launch_t(' in private and 'int o32r_launch(' in private
    assert not re.search(r'\bol_\w+', private)
    assert 'auto fn=pdlo32_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;' in private
    assert private.index('cudaFuncSetAttribute') < private.index('if(query)')
    assert 'cudaOccupancyMaxActiveBlocksPerMultiprocessor' not in private
    assert 'return ol_launch(0,0,0,0,32,0,1);' in QUERY
    old_diag = (HERE / 'dense_handoff16_diag.py').read_text(encoding='utf-8')
    for name in ('_d16_reference', '_d16_spin', '_d16_times', '_d16_projection_probe', '_pdli_equal', '_pdli_probe'):
        assert ast.dump(ast.parse(extract(old_diag, [name]))) == ast.dump(ast.parse(extract(app, [name])))
    assert '_d16_projection_probe(diag, 32, \'o\'' in app
    assert '_P16_EXT.o32r_launch(0, x.data_ptr()' in app
    assert 'fixed[3] == 196736' in app and 'fixed[6] > 0' in app and 'fixed[1] == 0' in app
    # Check acceptance using the actual AST predicate, including negative resources.
    prep = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    ready = next(n.value for n in ast.walk(prep) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'ready' for t in n.targets))
    predicate = compile(ast.Expression(ready), '<resource-predicate>', 'eval')
    cfg = (32, 128, 8, 1, 1, 1)
    original = dict(report={'private_query': 420000}, fixed=[42,0,0,196736,128,-1,1,233472,232448],
                    ol={'enabled': True, 'configs': [cfg], 'packed': object()}, cfg=cfg)
    assert eval(predicate, {}, original)
    rejected = 0
    for field, value in ((0,0),(1,4),(3,49152),(4,64),(6,0),(6,-1),(8,190000)):
        env = dict(original, fixed=list(original['fixed']))
        env['fixed'][field] = value
        assert not eval(predicate, {}, env)
        rejected += 1
    assert not eval(predicate, {}, dict(original, report={'private_query': 420001}))
    assert not eval(predicate, {}, dict(original, ol=dict(original['ol'], configs=[])))
    report = dict(source_sha256=binding['source_sha256'], checks_passed=True,
        unchanged_parent_prefix=True, unchanged_device_source_prefix=True,
        private_host_names=True, shared_probe_AST_exact=True,
        resource_positive_and_negative_cases=1+rejected+2,
        CPU_ABI_reproducer='Not executed: WSL g++ unavailable; no installation.',
        H200_host_build_and_resource_proof='Unverified until bounded diagnostic; no GPU jobs in this audit.')
    (OUT / 'cpu-audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
