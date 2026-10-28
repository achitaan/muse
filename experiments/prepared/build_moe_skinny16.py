"""Repair the dormant second skinny tile in one alternate, never the incumbents."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'


def build():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    tree = ast.parse(raw)
    cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == 'CUDA_SRC' for x in n.targets))
    original = cuda
    changes = []

    def edit(old, new, count=1):
        nonlocal cuda
        assert cuda.count(old) == count, (old[:80], cuda.count(old), count)
        cuda = cuda.replace(old, new)
        changes.append(dict(old=old, new=new, count=count))

    edit('#define OKC_SKT 8\n', '#define OKC_SKT 16\n')
    start = cuda.index('#else  // static skinny_ring: fixed range, no run-time pool\n')
    end = cuda.index('\n#endif\n\n// ---------------------------------------------------------------- fat unit:', start)
    static_original = cuda[start:end]
    static = static_original

    def static_edit(old, new):
        nonlocal static
        assert static.count(old) == 1, (old[:80], static.count(old))
        static = static.replace(old, new)

    # Scope the change to the static skinny_ring. SKDYN is false at SKT16.
    anchor = '    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n'
    static_edit(anchor, anchor + '    float second[4][4] = {};  // four independent K accumulators for tokens 8..15\n')
    static_edit('''          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);''',
         '''          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4,
                   a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(second[st], a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);''')
    old = '''    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);'''
    new = '''#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    if (two) {
#pragma unroll
      for (int k = 0; k < 4; ++k)
        second[0][k] += second[1][k] + second[2][k] + second[3][k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    if (two) *reinterpret_cast<float4*>(rd + 4) =
        make_float4(second[0][0], second[0][1], second[0][2], second[0][3]);'''
    static_edit(old, new)
    edit(static_original, static)
    # Every selected launch has its own symbol for actual-use profile evidence.
    count = cuda.count('moe_fused')
    edit('moe_fused', 'moe_fused_sk16', count=count)
    restored = cuda
    for c in reversed(changes):
        assert restored.count(c['new']) == c['count']
        restored = restored.replace(c['new'], c['old'])
    assert restored == original
    defs = ('@sk16', '-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1',
            '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4',
            '-DOKC_UWARP=0', '-DOKC_WS=0', '-DOKC_FINPF=1')
    setup = '\nALT_SRCS["sk16"] = ' + repr(cuda) + '\n_SK16_DEFINES = ' + repr(defs) + '\n'
    # Keep this as a prepared alternate source. Selection is a separate review
    # packet; no prepare/kernel_fn replacement or GPU run is made by this builder.
    output = raw + setup.encode()
    compile(output, '<moe-skinny16-prepared>', 'exec')
    (ROOT / '.tools/moe-skinny16.py').write_bytes(output)
    (ROOT / '.tools/moe-skinny16.cu').write_bytes(cuda.encode())
    report = dict(parent_sha256=PARENT, source_sha256=hashlib.sha256(output).hexdigest(),
                  bytes=len(output), original_source_exact_prefix=True,
                  original_cuda_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  alternate_cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                  inverse_ledger_exact=True, changes=changes, defines=defs,
                  intended_tokens=[9, 16],
                  status='prepared alternate only; not dispatched or GPU evaluated')
    (ROOT / '.tools/moe-skinny16-source-audit.json').write_bytes(json.dumps(report, indent=2).encode())
    print(json.dumps({k: v for k, v in report.items() if k != 'changes'}))


if __name__ == '__main__':
    build()
