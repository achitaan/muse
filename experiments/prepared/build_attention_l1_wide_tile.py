"""Prepare one fixed 128-key/2-stage L1-flat Attention draft; no GPU execution."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
BASE = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
OUT = ROOT / 'results/attention-l1-wide-tile-preflight-20261029'


def transform(cuda):
    edits = []

    def replace(old, new):
        nonlocal cuda
        assert cuda.count(old) == 1, (old[:120], cuda.count(old))
        pos = cuda.index(old)
        cuda = cuda[:pos] + new + cuda[pos + len(old):]
        edits.append(dict(offset=pos, old=old, new=new))

    replace('template <int L, bool F>\nstruct Cfg {',
            '#ifndef ATTN_L1_WIDE_TILE\n#define ATTN_L1_WIDE_TILE 1\n#endif\n'
            'template <int L, bool F>\nstruct Cfg {\n'
            '  static constexpr int KEYS = (ATTN_L1_WIDE_TILE && L == 1 && F) ? 128 : 64;')
    replace('static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;',
            'static constexpr int STAGES = F ? ((L == 1) ? (ATTN_L1_WIDE_TILE ? 2 : 3) : 4) : 3;')
    replace('struct Smem {\n', 'struct Smem {\n'
            '  static constexpr int TILE = Cfg<L, F>::KEYS;\n'
            '  static constexpr int STAGE_BYTES = TILE * ROWB;\n')
    replace('alignas(128) int pidx[4][80];',
            'alignas(128) int pidx[4][(ATTN_L1_WIDE_TILE && L == 1 && F) ? 144 : 80];')
    # The two scopes are segment() and attn_kernel(). Other helpers retain
    # their original 64-key layouts; L8 and non-flat instantiations stay64.
    scope = '  constexpr int NT = Cfg<L, FLAT>::NT;\n  constexpr int STAGES = Cfg<L, FLAT>::STAGES;'
    assert cuda.count(scope) == 2
    for start in reversed([m.start() for m in re.finditer(re.escape(scope), cuda)]):
        new = '  constexpr int TILE = Cfg<L, FLAT>::KEYS;\n  constexpr int STAGE_BYTES = TILE * ROWB;\n' + scope
        cuda = cuda[:start] + new + cuda[start + len(scope):]
        edits.append(dict(offset=start, old=scope, new=new))
    replace('const long long w1 = min(pt_lim, (e0 + 64 + 3) & -4);',
            'const long long w1 = min(pt_lim, (e0 + TILE + 3) & -4);')
    begin = cuda.index('    float s[4][4];')
    end = cuda.index('  pdl_launch_dependents();', begin)
    old = cuda[begin:end]
    assert old.count('float s[4][4];') == 1 and old.count('uint32_t kw[4][4];') == 1
    assert old.count('nt < 4') == 5 and old.count('32 * wsub') == 3
    assert old.count('load_v_frags_f8(V, 32 * wsub') == 1
    pv_old = '''      uint32_t b0[8], b1[8];
      load_v_frags_f8(V, 32 * wsub, hk, g, tig, b0, b1);
      const uint32_t a0 = e4m3x4f(s[0][0], s[0][1], s[1][0], s[1][1]);
      const uint32_t a2 = e4m3x4f(s[2][0], s[2][1], s[3][0], s[3][1]);
#pragma unroll
      for (int t = 0; t < 8; ++t) mma_f8pv(o[0][t], a0, a2, b0[t], b1[t]);
'''
    assert old.count(pv_old) == 1
    pv_new = '''#pragma unroll
      for (int pv = 0; pv < TILE / 64; ++pv) {
        uint32_t b0[8], b1[8];
        load_v_frags_f8(V, (TILE / 2) * wsub + 32 * pv, hk, g, tig, b0, b1);
        const uint32_t a0 = e4m3x4f(s[4*pv][0], s[4*pv][1], s[4*pv+1][0], s[4*pv+1][1]);
        const uint32_t a2 = e4m3x4f(s[4*pv+2][0], s[4*pv+2][1], s[4*pv+3][0], s[4*pv+3][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f8pv(o[0][t], a0, a2, b0[t], b1[t]);
      }
'''
    new = old.replace(pv_old, pv_new)
    new = new.replace('float s[4][4];', 'float s[TILE / 16][4];')
    new = new.replace('uint32_t kw[4][4];', 'uint32_t kw[TILE / 16][4];')
    new = new.replace('nt < 4', 'nt < TILE / 16')
    new = new.replace('32 * wsub', '(TILE / 2) * wsub')
    replace(old, new)
    restored = cuda
    for edit in reversed(edits):
        pos = edit['offset']
        assert restored[pos:pos + len(edit['new'])] == edit['new']
        restored = restored[:pos] + edit['old'] + restored[pos + len(edit['new']):]
    return cuda, edits, restored


def main():
    raw = (ROOT / 'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    text = raw.decode()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    cuda = ast.literal_eval(node.value)
    changed, edits, restored = transform(cuda)
    assert restored == cuda
    lines = text.splitlines(keepends=True)
    n = node.value
    start = sum(map(len, lines[:n.lineno - 1])) + n.col_offset
    end = sum(map(len, lines[:n.end_lineno - 1])) + n.end_col_offset
    draft = text[:start] + repr(changed) + text[end:]
    old_tag = 'okc_attn_v298{tag}_90a'
    assert draft.count(old_tag) == 1
    draft = draft.replace(old_tag, 'okc_attn_v298_wide_l1{tag}_90a')
    baseline_tree = ast.parse(text)
    draft_tree = ast.parse(draft)
    draft_node = next(n for n in draft_tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    draft_node.value = node.value

    class UndoTag(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str):
                n.value = n.value.replace('okc_attn_v298_wide_l1', 'okc_attn_v298')
            return n

    assert ast.dump(UndoTag().visit(draft_tree), include_attributes=False) == ast.dump(baseline_tree, include_attributes=False)
    compile(draft, '<attention-l1-wide-draft>', 'exec')
    OUT.mkdir(exist_ok=True)
    for name, data in [('submission.py', draft.encode()), ('baseline.cu', cuda.encode()),
                       ('candidate.cu', changed.encode()),
                       ('edits.json', (json.dumps(edits, indent=2) + '\n').encode())]:
        path = OUT / name
        assert not path.exists() or path.read_bytes() == data, 'Frozen draft collision'
        path.write_bytes(data)
    binding = dict(baseline_sha256=BASE, draft_sha256=hashlib.sha256(draft.encode()).hexdigest(),
                   baseline_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                   candidate_CUDA_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                   inverse_edits=len(edits), affected='L1 flat only; complete context, every request and key',
                   fixed_configuration='128 keys, two stages, original128 consumer/128 producer threads',
                   unchanged='L8, non-flat kernels, dispatch, numerical materializations, cache representation, finalizers',
                   GPU_jobs=0, candidate_installed=False, numerical_and_runtime_correctness='unverified')
    payload = (json.dumps(binding, indent=2) + '\n').encode()
    path = OUT / 'binding.json'
    assert not path.exists() or path.read_bytes() == payload
    path.write_bytes(payload)
    print(json.dumps(binding))


if __name__ == '__main__':
    main()
