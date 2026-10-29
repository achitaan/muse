"""Add native prelaunch admission and distinct execution symbols to the fixed draft."""
import ast
import hashlib
import json

from build_attention_l1_wide_tile import OUT, ROOT

DRAFT = '7ef8ab72409ef250979fcd14802af38ec3b7e966f9c662d3ce535f650eced553'


def main():
    raw = (OUT / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DRAFT
    text = raw.decode()
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    original = ast.literal_eval(node.value)
    cuda = original.replace('attn_kernel_fp8pv', 'attn_kernel_wide_l1')
    symbol_replacements = original.count('attn_kernel_fp8pv')
    assert symbol_replacements > 10
    assert cuda.count('#include <type_traits>') == 1
    cuda = cuda.replace('#include <type_traits>', '#include <type_traits>\n#include <cstdio>')
    anchor = '    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);'
    assert cuda.count(anchor) == 1
    admission = '''
    if constexpr (L == 1) {
      cudaFuncAttributes attr{};
      TORCH_CHECK(cudaFuncGetAttributes(&attr, attn_kernel_wide_l1<1, true>) == cudaSuccess,
                  "Wide L1 resource query failed");
      printf("ATTN_WIDE_RESOURCES %d %zu %zu %d %d %zu %d %d %d %d %d %d %d\\n",
             attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, occ_f, sm_f,
             sizeof(Smem<1, true>), Cfg<1, true>::KEYS, Cfg<1, true>::STAGES,
             Cfg<1, true>::NT, Cfg<1, true>::NP, Cfg<1, true>::CREGS,
             Cfg<1, true>::BASEREGS, nsm);
      TORCH_CHECK((attr.numRegs > 0 && attr.numRegs <= 128 && attr.localSizeBytes == 0 &&
                  attr.sharedSizeBytes == 0 && occ_f >= 2 && sm_f == 84992 &&
                  sizeof(Smem<1, true>) == 83968 && Cfg<1, true>::KEYS == 128 &&
                  Cfg<1, true>::STAGES == 2 && Cfg<1, true>::NT == 128 &&
                  Cfg<1, true>::NP == 128 && Cfg<1, true>::CREGS == 200 &&
                  Cfg<1, true>::BASEREGS == 128 && nsm > 0),
                  "Wide L1 fixed resource admission failed");
    }'''
    cuda = cuda.replace(anchor, anchor + admission)
    restored = cuda.replace(admission, '').replace('#include <type_traits>\n#include <cstdio>', '#include <type_traits>')
    restored = restored.replace('attn_kernel_wide_l1', 'attn_kernel_fp8pv')
    assert restored == original
    lines = text.splitlines(keepends=True)
    n = node.value
    start = sum(map(len, lines[:n.lineno-1])) + n.col_offset
    end = sum(map(len, lines[:n.end_lineno-1])) + n.end_col_offset
    payload = (text[:start] + repr(cuda) + text[end:]).encode()
    compile(payload, '<attention-l1-wide-admitted>', 'exec')
    candidate_tree = ast.parse(payload)
    changed_node = next(n for n in candidate_tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    changed_node.value = node.value
    assert ast.dump(candidate_tree, include_attributes=False) == ast.dump(tree, include_attributes=False)
    report = dict(draft_sha256=DRAFT, source_sha256=hashlib.sha256(payload).hexdigest(),
                  CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                  original_draft_restored_exactly=True, Python_AST_unchanged=True,
                  device_symbol_renames=symbol_replacements,
                  resource_marker='ATTN_WIDE_RESOURCES', resource_fields=['registers','local_bytes','static_shared_bytes',
                      'active_blocks','dynamic_shared_bytes','struct_bytes','keys','stages','consumer_threads',
                      'producer_threads','consumer_register_limit','base_register_limit','SMs'],
                  marker_expectation=[128,0,0,2,84992,83968,128,2,128,128,200,128,132],
                  marker_notes='Register count may be1..128; active blocks at least2; SMs positive. H200 required by full result.',
                  host_macro_condition_parenthesized=True, GPU_jobs=0, installed=False)
    destination = OUT / 'admission-v2'
    destination.mkdir(exist_ok=True)
    for name, data in [('submission.py', payload), ('kernel.cu', cuda.encode()),
                       ('binding.json', (json.dumps(report,indent=2)+'\n').encode())]:
        path = destination / name
        assert not path.exists() or path.read_bytes() == data, 'Frozen admission collision'
        path.write_bytes(data)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
