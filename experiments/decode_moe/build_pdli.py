"""Build the gated T1/T8 production integration from the exact scored R2f parent."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

from harness import append_harness
import pdli_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / 'results/decode-moe-plan-20261030'


def parent_namespace(raw):
    tree = ast.parse(raw)
    nodes = {}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            nodes.setdefault(n.targets[0].id,[]).append(n.value)
    ns = {}
    for name in ('tt_v10_sm2_SMALLM2_CUDA_TEMPLATE','V12_SM2_O1','V18S_CUDA_TEMPLATE','V18_SMALL',
                 'tt_v10_sm2_SMALLM2_ROUTES','tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE','tt_v12_smtrace1_SMTRACE_ROUTES',
                 'tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE','tt_v13_smtrace8_SMTRACE_ROUTES',
                 'v18d_SM_CUDA_TEMPLATE','BUNDLE_QKV_CUDA','_R2_DEV','_R2_HOST'):
        n = nodes[name][-1]
        if isinstance(n,ast.DictComp):
            assert name=='tt_v10_sm2_SMALLM2_ROUTES'
            assert ast.unparse(n)=="{key: value for (key, value) in tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}"
            value = {('pooler',8):ast.literal_eval(nodes[name][0])['pooler',8]}
        elif isinstance(n,ast.Call) and getattr(n.func,'id','')=='_zsrc':
            value = zlib.decompress(base64.b64decode(ast.literal_eval(n.args[0]))).decode()
        else:
            value = ast.literal_eval(n)
        ns[name] = value
    f = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_r2_source')
    exec(compile(ast.Module(body=[f],type_ignores=[]),'parent-r2-derivation','exec'),ns)
    return ns


def derive():
    raw = (HERE/'decoder_r2f.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='75fa2a73b4de60410c10dc3609141f3677c9ff79acff0512e45fdd6b5084f747'
    pdli_sources.__dict__.update(parent_namespace(raw))
    sources,specs,records = pdli_sources.pdli_make_sources()
    out = RESULTS/'pdli-cpu'
    out.mkdir(exist_ok=True)
    for name,source in sources.items():
        (out/(name+'.cu')).write_bytes(source.encode())
    packed,cpp,functions=pdli_sources.pdli_pack(sources,specs)
    (out/'combined.cu').write_bytes(packed.encode())
    (out/'combined.cpp').write_bytes(cpp.encode())
    record = {'parent_sha256':hashlib.sha256(raw).hexdigest(),'specs':specs,'derivations':records}
    record['combined_CUDA_sha256']=hashlib.sha256(packed.encode()).hexdigest()
    record['combined_functions']=functions
    (out/'source-binding.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    return raw.decode(),sources,specs,record


def main():
    import sys
    raw,sources,specs,record = derive()
    print(json.dumps({'sources':{k:len(v) for k,v in sources.items()},'parent_bytes':len(raw.encode())},indent=2))
    if '--sources-only' in sys.argv:
        return
    helpers = (HERE/'pdl_source.py').read_text()
    more = (HERE/'pdli_sources.py').read_text()
    line = 'from pdl_source import edit_once, twin\n'
    assert more.count(line)==1
    more = more.replace(line,'')
    app = '\n# PDL integration, derived from compiler-verified exact parent sources.\n' + helpers + '\n' + more
    app += '\n_PDLI_EXPECTED = ' + repr({k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}) + '\n'
    app += '_PDLI_COMBINED_HASH = '+repr(record['combined_CUDA_sha256'])+'\n'
    tree=ast.parse(raw);lines=raw.splitlines(True)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
    layer=''.join(lines[node.lineno-1:node.end_lineno])
    edits=[
        ('def kernel_fn(data, state):','def _pdli_layer(data, state, mask):'),
        (" qkv = _achita_dense_project(data['attn_input'], 'qkv_gate', state)",
         " qkv = _pdli_project(data['attn_input'], 'qkv_gate', state) if mask & 1 else _achita_dense_project(data['attn_input'], 'qkv_gate', state)"),
        (' o = _native_attn_kernel_fn(attention_data).reshape(tokens, hq * d)',
         ' o = _pdli_attention(attention_data, bool(mask & 1)).reshape(tokens, hq * d)'),
        (" attn_output = _achita_dense_project(o, 'o', state)",
         " attn_output = _pdli_project(o, 'o', state) if mask & 2 else _achita_dense_project(o, 'o', state)"),
        ('(r2 := _r2_pool_route(ffn_input, router_input, state))',
         '(r2 := (_pdli_route(ffn_input, router_input, state) if mask & 4 else _r2_pool_route(ffn_input, router_input, state)))'),
        ("  unpooled = _achita_dense_project(expert_output, 'unpooler', state, UNPOOL_SLICE)",
         "  unpooled = _pdli_project(expert_output, 'unpooler', state) if mask & 8 else _achita_dense_project(expert_output, 'unpooler', state, UNPOOL_SLICE)"),
    ]
    for old,new in edits:
        assert layer.count(old)==1,old
        layer=layer.replace(old,new)
    inverse=layer
    for old,new in reversed(edits):
        assert inverse.count(new)==1
        inverse=inverse.replace(new,old)
    assert inverse==''.join(lines[node.lineno-1:node.end_lineno])
    app += '\n'+layer+'\n'
    app += (HERE/'pdli_runtime.py').read_text()
    out = append_harness(raw) + app
    ast.parse(out)
    b = out.encode()
    assert len(b)<1_000_000
    assert b.startswith(raw.rstrip('\n').encode())
    (HERE/'decoder_pdli.py').write_bytes(b)
    record.update(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),headroom=1_000_000-len(b),GPU_check_launched=False)
    (RESULTS/'pdli-binding.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in record.items() if k not in ('specs','derivations')},indent=2))


if __name__=='__main__':
    main()
