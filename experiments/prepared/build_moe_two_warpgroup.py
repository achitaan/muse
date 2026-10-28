"""Package one fixed WGS architecture, preserving all skinny/fallback code."""
import ast
import hashlib
import json
from pathlib import Path
from moe_two_warpgroup_transform import transform

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
PARENT='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'


def build():
    raw=(ROOT/'best/moe_verified.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT
    base=raw.decode('utf8');tree=ast.parse(base)
    cn=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    cuda=ast.literal_eval(cn.value);new_cuda,ledger=transform(cuda)
    edits=[]
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        if node.name=='_ext':
            for item in ast.walk(node):
                if isinstance(item,ast.keyword) and item.arg=='cpp_sources':
                    item.value=ast.parse("CPP_SRC + ('\\nstd::vector<int64_t> okc_w2g_resources();\\n' if '@w2g' in defines else '')",mode='eval').body
                if isinstance(item,ast.keyword) and item.arg=='functions':
                    item.value=ast.parse("['okc_run', 'okc_set_pdl', 'okc_w2g_resources'] if '@w2g' in defines else ['okc_run', 'okc_set_pdl']",mode='eval').body
            edits.append((node,ast.unparse(node)))
        if node.name=='kernel_fn':
            index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.For))
            injected=ast.parse('''
weights_bank, scratch_bank, grid = state['w'], state['scratch'], state['grid']
if M > MID_MAX_M and state.get('w2g') is not None:
    weights_bank = state['w2g']['w']
    scratch_bank = state['w2g']['scratch']
    grid = state['w2g']['grid']
''').body
            node.body[index:index]=injected
            call=next(n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='okc_run')
            assert ast.unparse(call.args[0])=="state['w']" and ast.unparse(call.args[-1])=="state['grid']"
            call.args[0]=ast.Name(id='weights_bank',ctx=ast.Load());call.args[1]=ast.Name(id='scratch_bank',ctx=ast.Load());call.args[-1]=ast.Name(id='grid',ctx=ast.Load())
            edits.append((node,ast.unparse(ast.fix_missing_locations(node))))
    lines=base.splitlines(keepends=True)
    for node,replacement in sorted(edits,key=lambda x:x[0].lineno,reverse=True):
        lines[node.lineno-1:node.end_lineno]=[replacement+'\n']
    addon='''
# Local architectural variant: two WGS consumer groups, two X-ring slots,
# 32-token jobs, exact wide scheduler arithmetic and a capacity-checked grid.
ALT_SRCS = dict(ALT_SRCS)
ALT_SRCS['w2g'] = W2G_CUDA
MAIN_DEFINES = tuple(d for d in MAIN_DEFINES if not d.startswith('-DOKC_NW=')) + ('-DOKC_NW=8', '@w2g')
_w2g_original_prepare = prepare

def _w2g_repack(bank, experts, units):
    return bank.view(experts, units, 16, 2, 2, 9216).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1)

def prepare(weights):
    state = _w2g_original_prepare(weights)
    state['w2g'] = None
    if state['sfx'] and '-DOKC_WGS=1' in _main_defines(state['sfx']):
        ext = _ext(tuple(_main_defines(state['sfx'])) + state['sfx'])
        resource = tuple(ext.okc_w2g_resources())
        print('MOE_W2G_RESOURCES', resource, flush=True)
        count = state['cnt']
        bank = list(state['w'][:12]) + [_w2g_repack(state['w'][12], count, 32), _w2g_repack(state['w'][13], count, 16)]
        grid = min(2 * state['grid'], resource[-1], 271)
        if grid <= 0:
            raise RuntimeError('WGS2 kernel cannot be resident')
        state['w2g'] = dict(w=bank, scratch=_make_scratch(weights['w13'].device), grid=grid, resources=resource)
        torch.cuda.synchronize()
        # Own analytic all-remote inputs exercise zero-job ranges, both BIG
        # variants, repeated scratch reuse and full4096-column output stores.
        # No actual call input or expected output is prepared/cached.
        remote = (state['lo'] + count) % 256
        for tokens in (17, 33, 384):
            x = torch.zeros((tokens, 4096), dtype=torch.bfloat16, device=weights['w13'].device)
            ids = torch.full((tokens, 8), remote, dtype=torch.int32, device=x.device)
            route_weights = torch.full((tokens, 8), 0.125, dtype=torch.float32, device=x.device)
            out = torch.empty_like(x)
            ext.okc_run(bank, state['w2g']['scratch'], x, ids, route_weights, out, state['lo'], count, grid)
            if torch.count_nonzero(out).item() != 0:
                raise RuntimeError('WGS2 analytic all-remote probe failed')
        print('MOE_W2G_ZERO True', flush=True)
        del x, ids, route_weights, out
    print('MOE_W2G_ACTIVE', state['w2g'] is not None, flush=True)
    # New bank/scratch initialization must finish before CUDA graph warm-up.
    torch.cuda.synchronize()
    return state
'''
    result=''.join(lines)+'\nW2G_CUDA = '+repr(new_cuda)+'\n'+addon
    compile(result,'<moe-two-warpgroup>','exec')
    source=result.encode('utf8');path=ROOT/'.tools/moe-two-warpgroup-variant.py';path.write_bytes(source)
    (ROOT/'.tools/moe-two-warpgroup-CUDA.cu').write_bytes(new_cuda.encode('utf8'))
    (ROOT/'.tools/moe-two-warpgroup-edit-ledger.json').write_bytes((json.dumps(ledger,indent=2)+'\n').encode())
    summary=dict(parent_sha256=PARENT,source_sha256=hashlib.sha256(source).hexdigest(),source_bytes=len(source),
        alternate_cuda_sha256=hashlib.sha256(new_cuda.encode()).hexdigest(),inverse_exact=True,
        original_cuda_unchanged=True,modified_python_functions=['_ext','kernel_fn','prepare wrapper'],
        main_fast_geometry='2consumer groups; FT_T32; NX2; NW8; NTL320; launch_bounds min2',
        main_fast_grid='min(2*SMs, actual capacity,271); reject promotion if occupancy<2',
        original_tiny_mid_and_ineligible_fallback=True,independent_main_weight_and_scratch_bank=True,
        compiled_main_modules='new main replaces old fast main; no extra fast-main compilation',gpu_verified=False)
    (ROOT/'.tools/moe-two-warpgroup-source-audit.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':build()
