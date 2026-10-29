"""Build a lossless BF16-factor scratch/cache prototype; launch nothing."""
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
    cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    original, edits = cuda, []

    def replace(old, new, name):
        nonlocal cuda
        assert cuda.count(old) == 1, name
        at = cuda.index(old)
        cuda = cuda[:at] + new + cuda[at + len(old):]
        edits.append(dict(name=name, offset=at, old=old, new=new))

    anchor = '__device__ __forceinline__ float probe_q(float v) {'
    helper = (HERE / 'moe_factor_y_helpers.cuh').read_text(encoding='utf-8').replace('\r\n', '\n')
    replace(anchor, helper + '\n' + anchor, 'BF16-bit-product and FP32 reconstruction helpers')
    start = cuda.index('template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile')
    end = cuda.index('template <bool PB>\n__device__ __forceinline__ void wgs_flush', start)
    old_epi = cuda[start:end]
    new_epi = old_epi
    replacements = [
        ('    const float w0 = __bfloat162float(__ldg(p.normw + n0)), w1 = __bfloat162float(__ldg(p.normw + n0 + 8));\n', ''),
        ('        const float2 achita_pair_wgs = achita_marlin_out2(d4[h], d4[2 + h], gsc);',
         '        const __nv_bfloat162 product = factor_marlin_bits2(d4[h], d4[2 + h], gsc);'),
        ('        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), achita_pair_wgs.x * w0);',
         '        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0), __low2bfloat16(product));'),
        ('        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), achita_pair_wgs.y * w1);',
         '        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0 + 8), __high2bfloat16(product));'),
    ]
    for old, new in replacements:
        assert new_epi.count(old) == 1
        new_epi = new_epi.replace(old, new)
    replace(old_epi, new_epi, 'WGS phase B stores exact BF16 pre-gain product bits')
    start = cuda.index('#if OKC_WGS && OKC_WGSFINSS\n', cuda.index('__global__ void __launch_bounds__'))
    end = cuda.index('#elif OKC_FINCOL', start)
    finalizer = (HERE / 'moe_factor_y_finalizer.cuh').read_text(encoding='utf-8').replace('\r\n', '\n')
    replace(cuda[start:end], finalizer, 'WGS finalizer caches all eight BF16-factor rows in registers')
    range_anchor = '__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n'
    replace(range_anchor, range_anchor + '  if (nj == 0) return 0;  // no job-table read or divide for all-nonlocal inputs\n',
            'Explicit empty-job range, preserving every nonempty range')
    # A distinct device namespace makes actual execution visible in profiles.
    replace('namespace okc {', 'namespace okc_factor_y {', 'Distinct device namespace')
    renamed_refs = cuda.count('okc::')
    for _ in range(renamed_refs):
        at = cuda.index('okc::')
        cuda = cuda[:at] + 'okc_factor_y::' + cuda[at + 5:]
        edits.append(dict(name='Host references distinct device namespace', offset=at,
                          old='okc::', new='okc_factor_y::'))
    # Neither the scratch offsets nor synchronization/ownership protocol changes.
    restored = cuda
    for edit in reversed(edits):
        a = edit['offset']
        assert restored[a:a + len(edit['new'])] == edit['new']
        restored = restored[:a] + edit['old'] + restored[a + len(edit['new']):]
    assert restored == original
    digest = hashlib.sha256(cuda.encode()).hexdigest()
    report = dict(parent_sha256=PARENT, original_cuda_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  factor_cuda_sha256=digest, cuda_bytes=len(cuda.encode()), reversible_edits=len(edits),
                  factor_representation_edits=3, empty_job_guard_edits=1, namespace_reference_renames=renamed_refs,
                  original_cuda_inverse_exact=True, phase_A_math_exact_ignoring_namespace=True,
                  nonempty_routing_weights_readiness_fixups_and_grid_barrier_exact_ignoring_namespace=True,
                  scratch_capacity_and_fixup_offsets_exact=True,
                  numerical_contract='Same BF16 GEMM accumulator/global-scale/product; same BF16 effective gain; explicit FP32 gain product before identical norm/route reductions',
                  logical_y_traffic_bytes_per_route_before=12 * 4096,
                  logical_y_traffic_bytes_per_route_after=4 * 4096,
                  traffic_is_not_measured_HBM=True,
                  gpu_verified=False, runtime_wrapper_built=False)
    (ROOT / '.tools/moe-factor-y.cu').write_bytes(cuda.encode())
    (ROOT / '.tools/moe-factor-y-cuda-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    (ROOT / '.tools/moe-factor-y-cuda-edits.json').write_bytes((json.dumps(edits, indent=2) + '\n').encode())
    addon = (HERE / 'moe_factor_y_addon.py').read_bytes()
    addon_tree = ast.parse(addon)
    for node in addon_tree.body:
        if isinstance(node, ast.FunctionDef):
            assert node.name.startswith('_factor_') or node.name in ('prepare', 'kernel_fn')
            assert not node.decorator_list
            assert not node.args.defaults or node.name == '_factor_input'
        elif isinstance(node, ast.Assign):
            assert all(isinstance(n, ast.Name) and n.id.startswith(('_factor_', '_FACTOR_')) for n in node.targets)
        else:
            assert isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom))
    submission = parent + b'\n# Lossless BF16 output-factor scratch and register-cache candidate.\n_FACTOR_CUDA = ' + repr(cuda).encode() + b'\n' + addon
    combined = ast.parse(submission)
    assert ast.dump(ast.Module(body=combined.body[:len(tree.body)], type_ignores=[])) == ast.dump(tree)
    report.update(source_sha256=hashlib.sha256(submission).hexdigest(), source_bytes=len(submission),
                  original_parent_prefix_exact=submission.startswith(parent), runtime_wrapper_built=True)
    (ROOT / '.tools/moe-factor-y.py').write_bytes(submission)
    (ROOT / '.tools/moe-factor-y-source-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
