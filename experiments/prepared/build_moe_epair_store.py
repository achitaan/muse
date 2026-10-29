"""Build one source-preserving phase-A store candidate, without GPU calls."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
HERE = Path(__file__).resolve().parent
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'


def main():
    parent = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT
    tree = ast.parse(parent)
    original = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    cuda, edits = original, []

    def replace(old, new, name):
        nonlocal cuda
        assert cuda.count(old) == 1, name
        at = cuda.index(old)
        cuda = cuda[:at] + new + cuda[at + len(old):]
        edits.append(dict(name=name, offset=at, old=old, new=new))

    helper = '''__device__ __forceinline__ void epair_st_keep_u32(uint32_t* a, uint32_t v) {
#if OKC_L2H
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "r"(v), "l"(0x14F0000000000000ull) : "memory");
#else
  *a = v;
#endif
}

'''
    anchor = 'constexpr int HD = 4096, ID = 4096;'
    replace(anchor, helper + anchor, '32bit store with original intermediate-cache policy')
    start = cuda.index('template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile')
    a = cuda.index('    // SwiGLU straight into xB', start)
    b = cuda.index('  } else {', a)
    phase_b = cuda[b:cuda.index('template <bool PB>\n__device__ __forceinline__ void wgs_flush', b)]
    phase_a = (HERE / 'moe_epair_store_phase_a.cuh').read_text(encoding='utf-8').replace('\r\n', '\n')
    replace(cuda[a:b], phase_a, 'WGS phase-A scalar stores become exact adjacent-column pairs')
    assert phase_b in cuda
    anchor = '__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n'
    replace(anchor, anchor + '  if (nj == 0) return 0;  // all-nonlocal inputs have no job-table entry\n',
            'Explicit empty-job return, preserving all nonempty ranges')
    replace('namespace okc {', 'namespace okc_epair {', 'Actual new-path namespace')
    while 'okc::' in cuda:
        at = cuda.index('okc::')
        cuda = cuda[:at] + 'okc_epair::' + cuda[at + 5:]
        edits.append(dict(name='Launcher namespace', offset=at, old='okc::', new='okc_epair::'))
    restored = cuda
    for e in reversed(edits):
        a = e['offset']
        assert restored[a:a + len(e['new'])] == e['new']
        restored = restored[:a] + e['old'] + restored[a + len(e['new']):]
    assert restored == original
    addon = (HERE / 'moe_epair_store_addon.py').read_bytes()
    combined = parent + b'\n# WGS adjacent-column BF16 epilogue-store candidate.\n_EPAIR_CUDA = ' + repr(cuda).encode() + b'\n' + addon
    combined_tree = ast.parse(combined)
    assert ast.dump(ast.Module(body=combined_tree.body[:len(tree.body)], type_ignores=[])) == ast.dump(tree)
    report = dict(parent_sha256=PARENT, original_cuda_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  epair_cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                  source_sha256=hashlib.sha256(combined).hexdigest(), source_bytes=len(combined),
                  original_cuda_inverse_exact=True, parent_prefix_exact=combined.startswith(parent),
                  phase_B_exact_ignoring_namespace=True, scratch_layout_unchanged=True,
                  mainloop_weights_routing_readiness_finalizer_unchanged_ignoring_namespace=True,
                  numerical_contract='Same Marlin BF16 boundaries, FP32 SiLU expression and BF16 store value; opaque lane exchange only',
                  new_synchronization='One XOR4 shuffle reached by all32 lanes before valid-store predicate; no new CTA/grid join',
                  runtime_wrapper_built=True, gpu_verified=False)
    out = ROOT / '.tools'
    (out / 'moe-epair-store.cu').write_bytes(cuda.encode())
    (out / 'moe-epair-store.py').write_bytes(combined)
    (out / 'moe-epair-store-edits.json').write_bytes((json.dumps(edits, indent=2) + '\n').encode())
    (out / 'moe-epair-store-source-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
