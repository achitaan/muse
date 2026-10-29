"""One exact identity-rescale short circuit for L1 flat Attention; launch nothing."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'


def transform(cuda, diagnostic=True):
    edits = []
    def replace(old, new):
        nonlocal cuda
        assert cuda.count(old) == 1, (old[:100], cuda.count(old))
        at = cuda.index(old)
        cuda = cuda[:at] + new + cuda[at + len(old):]
        edits.append(dict(offset=at, old=old, new=new))
    replace('    const float m_new = fmaxf(m_r[0], mx);\n',
            '    const float m_new = fmaxf(m_r[0], mx);\n'
            '    // Warp-uniform test, before any per-row finite-state branch.\n'
            '    const bool rescale = !FLAT || __any_sync(0xffffffffu, m_new != m_r[0]);\n')
    replace('      const float alpha = ex2(m_r[0] - m_new);\n',
            '      float alpha = 1.f;\n'
            '      if (rescale) alpha = ex2(m_r[0] - m_new);\n')
    old = '''        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
'''
    new = '''        if (rescale) {
          l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
          for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
        } else {
          // ex2(+/-0) is exactly1; FMA(l,1,ls) equals RN(l+ls).
          l_r[0] += ls;
        }
'''
    replace(old, new)
    if diagnostic:
        replace('#include <type_traits>', '#include <type_traits>\n#include <cstdio>')
        name_count = cuda.count('attn_kernel_fp8pv')
        for _ in range(name_count):
            at = cuda.index('attn_kernel_fp8pv')
            cuda = cuda[:at] + 'attn_kernel_skipscale' + cuda[at + len('attn_kernel_fp8pv'):]
            edits.append(dict(offset=at, old='attn_kernel_fp8pv', new='attn_kernel_skipscale'))
        marker = '    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);'
        replace(marker, marker + '''
    if constexpr (L == 1) {
      cudaFuncAttributes attr{};
      TORCH_CHECK(cudaFuncGetAttributes(&attr, attn_kernel_skipscale<L, true>) == cudaSuccess,
                  "Identity-rescale resource query failed");
      printf("ATTN_SKIP_RESOURCES %d %zu %d %d %zu %d %d %d %d\\n",
             attr.numRegs, attr.localSizeBytes, occ_f, sm_f, sizeof(Smem<L, true>),
             Cfg<L, true>::MINB, Cfg<L, true>::BASEREGS, Cfg<L, true>::CREGS, nsm);
      TORCH_CHECK(attr.numRegs > 0 && attr.numRegs <= 128 && attr.localSizeBytes == 0 && occ_f >= 2,
                  "Identity-rescale fixed resource gate failed");
    }''')
    return cuda, edits


def build():
    raw = (ROOT / 'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    text = raw.decode()
    tree = ast.parse(text)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    old = ast.literal_eval(assignment.value)
    new, edits = transform(old)
    restored = new
    for edit in reversed(edits):
        a = edit['offset']
        assert restored[a:a + len(edit['new'])] == edit['new']
        restored = restored[:a] + edit['old'] + restored[a + len(edit['new']):]
    assert restored == old
    lines = text.splitlines(keepends=True)
    node = assignment.value
    start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
    result = text[:start] + repr(new) + text[end:]
    old_tag, new_tag = 'okc_attn_v298{tag}_90a', 'okc_attn_v298_skipscale{tag}_90a'
    assert result.count(old_tag) == 1
    result = result.replace(old_tag, new_tag)
    # Restore the CUDA literal and one module-name constant in the candidate AST.
    check = ast.parse(result)
    target = next(n for n in check.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    target.value = assignment.value
    class UndoTag(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str):
                n.value = n.value.replace('okc_attn_v298_skipscale', 'okc_attn_v298')
            return n
    check = UndoTag().visit(check)
    assert ast.dump(check, include_attributes=False) == ast.dump(tree, include_attributes=False)
    payload = result.encode()
    compile(payload, '<attention-skip-identity-scale>', 'exec')
    # The identical arithmetic edit anchors also exist in retained Decode.
    decoder = (ROOT / 'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(decoder).hexdigest() == 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
    dnode = next(n for n in ast.parse(decoder).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_native_attn_CUDA_SRC' for t in n.targets))
    _, transfer_edits = transform(ast.literal_eval(dnode.value), diagnostic=False)
    report = dict(parent_sha256=PARENT, source_sha256=hashlib.sha256(payload).hexdigest(),
                  cuda_sha256=hashlib.sha256(new.encode()).hexdigest(), bytes=len(payload),
                  original_cuda_inverse_exact=True, all_other_Python_AST_exact_except_module_tag=True,
                  changed_math_scope='L1 FLAT only: skip identity rescale when every lane of the warp keeps its maximum',
                  unchanged_probability_scale=256, unchanged_L8=True, unchanged_L1_nonflat=True,
                  same_decode_arithmetic_anchors=len(transfer_edits),
                  no_scratch_geometry_routing_or_precision_changes=True, gpu_verified=False,
                  edits=edits, module_tag_edit=dict(old=old_tag, new=new_tag))
    (ROOT / '.tools/attention-skipscale.py').write_bytes(payload)
    (ROOT / '.tools/attention-skipscale.cu').write_bytes(new.encode())
    (ROOT / '.tools/attention-skipscale-source-audit.json').write_bytes((json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k != 'edits'}, indent=2))


if __name__ == '__main__':
    build()
