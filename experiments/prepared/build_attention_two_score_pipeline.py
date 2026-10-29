"""One fixed offline L8 score-pipeline prototype; no active source or GPU use."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PARENT = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
OUT = ROOT/'results/attention-two-score-pipeline-preflight-20261029/v1'


def freeze(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists() or path.read_bytes() == raw, str(path)
    path.write_bytes(raw)


def once(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spill-repair', action='store_true',
        help='One diagnosed repair: maximum feasible 232-consumer budget in a 168-register initial CTA')
    args = parser.parse_args()
    out = OUT if not args.spill_repair else OUT.parent/'v2-register-repair'
    consumer = 232 if args.spill_repair else 152
    raw = (ROOT/'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    tree = ast.parse(raw)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    original = ast.literal_eval(assignment.value)
    sa = original.index('template <int L, bool FLAT, int ROLE, int MODE = 0>')
    sb = original.index('template <int L, bool WS, int NT>\n', sa)
    segment = original[sa:sb]
    clone = once(segment, 'bool segment(', 'bool segment_two_score(')
    # The clone is instantiated only for L8/flat consumer role2. Preserve every
    # other original function, including the producer and all fallback kernels.
    loop_a = clone.index('  const int nit = ntiles;')
    loop_b = clone.index('  PROF_DUMP;', loop_a)
    old_loop = clone[loop_a:loop_b]
    prep_a = old_loop.index('      if constexpr (WS) {')
    prep_b = old_loop.index('    const int st = st_c;', prep_a)
    prep = old_loop[prep_a:prep_b]
    qk_a = old_loop.index('    wgmma_fence();', prep_b)
    qk_b = old_loop.index('    if constexpr (!WS)', qk_a)
    qk = old_loop[qk_a:qk_b]
    # Use a helper with explicit target bank and stage; the two score arrays
    # live outside the stepping lambda and its bank tags are compile-time.
    qk_helper = '  auto issue_score = [&](float* s, int st) {\n'+qk+'  };\n'
    vt_a = old_loop.index('    uint8_t* V16t =')
    vt_b = old_loop.index('    wgmma_wait<0>();', vt_a)
    transpose = old_loop[vt_a:vt_b].replace('if (jj == 0)', 'if (i == 0)')
    math_a = old_loop.index('    if constexpr (WS) {   // K (wgmma)', vt_b)
    math_b = old_loop.index('      float* od = &o[0][0][0];\n      \n           // done before the next tile', math_a)
    math = old_loop[math_a:math_b]
    # Numerical source is copied as one exact block, from raw-stage release
    # through masking, offset moves, FP8 packing, and the original PV commit.
    assert 'constexpr bool PCHK = WS;' in math
    assert 'wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);' in math
    stage_advance = '''
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
'''
    assert stage_advance in old_loop
    # prepare must take a future stage/phase explicitly; do not advance the
    # current raw-stage ownership or its transposed-V parity early.
    prepare = prep.replace('st_c', 'stage').replace('ph_c', 'phase')
    new_loop = '''  static_assert(L == 8 && FLAT && ROLE == 2, "fixed two-score consumer");
  const int nit = ntiles;
  float score0[32], score1[32];
  auto prepare_score = [&](int i, int stage, uint32_t phase) {
'''+prepare+'''  };
'''+qk_helper+'''
  prepare_score(0, st_c, ph_c);
  issue_score(score0, st_c);
  auto step_score = [&](int i, auto bank_tag, auto next_tag) {
    constexpr int BANK = decltype(bank_tag)::value;
    constexpr bool NEXT = decltype(next_tag)::value;
    float* s = BANK == 0 ? score0 : score1;
    float* sn = BANK == 0 ? score1 : score0;
    if (i == 0) wgmma_wait<0>();
    else wgmma_wait<1>(); // QK_i complete; PV_(i-1) may remain
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
    if constexpr (NEXT) {
      int ns = st_c + 1;
      uint32_t np = ph_c;
      if (ns == STAGES) { ns = 0; np ^= 1u; }
      prepare_score(i + 1, ns, np);
      issue_score(sn, ns);
    }
    const int st = st_c;
'''+transpose+'''
    if constexpr (NEXT) wgmma_wait<1>();
    else wgmma_wait<0>(); // previous PV is complete; next QK may remain
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    float* prior_o = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(prior_o[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
'''+math+'''
    PROF_TILE;
'''+stage_advance+'''  };
  int i = 0;
  for (; i + 2 < nit; i += 2) {
    step_score(i, std::integral_constant<int,0>(), std::true_type());
    step_score(i + 1, std::integral_constant<int,1>(), std::true_type());
  }
  if (i + 1 < nit) {
    step_score(i, std::integral_constant<int,0>(), std::true_type());
    step_score(i + 1, std::integral_constant<int,1>(), std::false_type());
  } else {
    step_score(i, std::integral_constant<int,0>(), std::false_type());
  }
'''
    clone = clone[:loop_a]+new_loop+clone[loop_b:]
    ka = original.index('template <int L, bool FLAT, int MODE = 0>\n__global__')
    kb = original.index('\n}  // namespace', ka)
    kernel = original[ka:kb]
    new_kernel = once(kernel, '__launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel_fp8pv(',
        '__launch_bounds__(Cfg<L, FLAT>::NTX, 1) attn_kernel_two_score(')
    old_call = '        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,\n                            0, 0,\n                            L == 1 && Cfg<L, FLAT>::WS && k == 0);'
    new_call = '''        if constexpr (R == 1) {
'''+old_call+'''
        } else {
'''+old_call.replace('segment<L, FLAT, R>', 'segment_two_score<L, FLAT, R>')+'''
        }'''
    new_kernel = once(new_kernel, old_call, new_call)
    new_kernel = once(new_kernel, '"n"(Cfg<L, FLAT>::CREGS)', '"n"('+str(consumer)+')')
    # No host dispatch in this first compile-only artifact. The full original
    # source is recoverable by deleting precisely these two additions.
    source = original[:sa]+clone+original[sa:kb]+ '\n'+new_kernel+original[kb:]
    recovered = once(once(source, clone, ''), '\n'+new_kernel, '')
    assert recovered == original
    anchor = 'template <int L>\nvoid launch('
    prefix = source.split(anchor)[0]
    for header in ('#include <torch/extension.h>\n', '#include <ATen/cuda/CUDAContext.h>\n', '#include <cuda.h>\n'):
        prefix = once(prefix, header, '')
    report = dict(parent_source_sha256=PARENT, parent_CUDA_sha256=hashlib.sha256(original.encode()).hexdigest(),
        prototype_CUDA_sha256=hashlib.sha256(source.encode()).hexdigest(),
        compiler_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        original_source_exact_after_deleting_two_additions=True,
        numerical_tile_block_exact=True, numerical_tile_block_sha256=hashlib.sha256(math.encode()).hexdigest(),
        producer_calls_original_segment=True, cooperative_finalizer_unchanged=True,
        score_banks=2, producer_register_target=32, consumer_register_target=consumer,
        minimum_blocks_launch_bound=1, intended_native_initial_registers=[168,168] if args.spill_repair else [112,152],
        redistributed_registers=128*32+256*consumer,
        repair_reason='v1 compiled but spilled 88B stack/126B stores/344B loads; test one maximum-feasible budget, not a sweep' if args.spill_repair else None,
        CUDA_prototype_written=True, host_dispatch_written=False, CPU_only=True, GPU_jobs=0,
        outstanding='Offline compilation and generated-code audit; native resource admission, grid/split host integration, numerical/replay/coverage proof and performance remain unverified')
    # v1 predates the explicit resource/repair metadata; preserve its binding.
    if not args.spill_repair:
        report.pop('redistributed_registers'); report.pop('repair_reason')
    freeze(out/'kernel.cu', source.encode())
    freeze(out/'compiler-prefix.cu', prefix.encode())
    freeze(out/'binding.json', (json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
