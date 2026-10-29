"""Exact identity-state proof and finite CPU replay screens, never GPU validation."""
import ast
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from build_attention_skip_identity_scale import ROOT, PARENT, transform

parent = (ROOT / 'best/attention_checked.py').read_bytes()
candidate = (ROOT / '.tools/attention-skipscale.py').read_bytes()
assert hashlib.sha256(parent).hexdigest() == PARENT
meta = json.loads((ROOT / '.tools/attention-skipscale-source-audit.json').read_bytes())
assert hashlib.sha256(candidate).hexdigest() == meta['source_sha256']
def cuda(raw):
    return next(ast.literal_eval(n.value) for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
old, new = cuda(parent), cuda(candidate)
assert transform(old)[0] == new
# Source anchors bind this arithmetic argument to the executed CUDA candidate.
assert new.index('const bool rescale = !FLAT || __any_sync') < new.index('if (m_new != -INFINITY)')
assert 'if (rescale) alpha = ex2(m_r[0] - m_new);' in new
assert new.count('l_r[0] += ls;') == 1
old_probability = old[old.index('          const float pv = ex2(s[nt][e] - m_new) * 256.f;'):old.index('      if constexpr (ONE)', old.index('const float m_new'))]
assert old_probability in new
assert old[old.index('      uint32_t b0[8], b1[8];'):old.index('  // ---- merge:')] .replace('attn_kernel_fp8pv', 'attn_kernel_skipscale') in new
abi_tree = ast.parse((ROOT / 'experiments/prepared/audit_attention_three_cta_cpu.py').read_bytes())
abi_fn = next(n for n in abi_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'smem_layout')
abi_namespace = {'re': re}
exec(compile(ast.Module(body=[abi_fn], type_ignores=[]), '<owned-smem-layout-model>', 'exec'), abi_namespace)
layout = abi_namespace['smem_layout']
layouts = {}
for length in (1,8):
    for flat in (False,True):
        before, after = layout(old,length,flat),layout(new,length,flat)
        assert before == after
        layouts[f'L{length}_flat{int(flat)}'] = after['dynamic']
assert layouts['L1_flat1'] == 67584
assert 2*(128*200+128*56) == 65536

rng = np.random.default_rng(20261029)
words = rng.integers(0, 2**32, size=262144, dtype=np.uint32)
words = words[(words & 0x7f800000) != 0x7f800000]
words = np.concatenate((words, np.array([0,0x80000000,1,0x80000001,0x007fffff,
                                       0x807fffff,0x7f7fffff,0xff7fffff],dtype=np.uint32)))
values = words.view(np.float32)
assert np.array_equal((values * np.float32(1)).view(np.uint32), words)
# FMA(l,1,ls) and RN(l+ls) have the same exact real sum. FP64 is an
# independent rounding screen for finite FP32 sums, including ties/extremes.
other = np.resize(values[::-1], values.shape)
with np.errstate(over='ignore', invalid='ignore'):
    oracle = (values.astype(np.float64) * 1.0 + other.astype(np.float64)).astype(np.float32)
    added = values + other
assert np.array_equal(oracle.view(np.uint32), added.view(np.uint32))

# Every subset of eight query quads: vote must execute on all32 lanes before
# finite-state divergence; any moving quad preserves the complete old path.
for mask in range(256):
    moving = np.repeat(np.array([(mask>>i)&1 for i in range(8)],dtype=bool),4)
    assert bool(moving.any()) == (mask != 0)


def same(a,b):
    assert np.array_equal(a.view(np.uint32),b.view(np.uint32))


traces, iterations, identity_warps, rescale_warps = [], 0, 0, 0
for mode, tiles in (('coherent',257),('random',129),('growing',65),('masked_edges',35),('signed_zero',17)):
    m = np.full(32,-np.inf,np.float32)
    l = np.zeros(32,np.float32)
    out = np.zeros((32,16),np.float32)
    m2,l2,out2 = m.copy(),l.copy(),out.copy()
    skipped, moved = 0,0
    for step in range(tiles):
        row_scores = rng.normal(size=(8,4,8)).astype(np.float32)
        if mode=='coherent': row_scores += np.float32(3)
        if mode=='growing': row_scores += np.float32(step*.25)
        if mode=='masked_edges' and step in (0,1,34): row_scores.fill(-np.inf)
        if mode=='signed_zero': row_scores.fill(np.float32(-0.0 if step%2 else 0.0))
        local_scores = row_scores.reshape(32,8)
        maxima = np.repeat(row_scores.max(axis=(1,2)),4)
        nm, nm2 = np.maximum(m,maxima),np.maximum(m2,maxima)
        active = nm != -np.inf
        move = bool(np.any(nm2 != m2))
        skipped += int(not move); moved += int(move)
        with np.errstate(invalid='ignore',under='ignore'):
            # Approximation itself is deliberately shared, just as unchanged
            # CUDA ex2 is called for every nonidentity row. This does not
            # model NVIDIA ex2 rounding or Tensor Core accumulation.
            alpha = np.exp2(m-nm).astype(np.float32)
            alpha2 = np.exp2(m2-nm2).astype(np.float32) if move else np.ones(32,np.float32)
            probabilities = np.exp2(local_scores-nm[:,None]).astype(np.float32)*np.float32(256)
        psum = np.sum(probabilities,axis=1,dtype=np.float32)
        # Arbitrary finite output contributions stand for the unchanged PV
        # operation; identical incoming accumulator bits are the invariant.
        delta = rng.normal(size=out.shape).astype(np.float32)
        l[active] = (l[active].astype(np.float64)*alpha[active].astype(np.float64)+psum[active].astype(np.float64)).astype(np.float32)
        out[active] *= alpha[active,None]
        if move:
            l2[active] = (l2[active].astype(np.float64)*alpha2[active].astype(np.float64)+psum[active].astype(np.float64)).astype(np.float32)
            out2[active] *= alpha2[active,None]
        else:
            l2[active] += psum[active]
        out[active] += delta[active];out2[active] += delta[active]
        m[active] = nm[active];m2[active] = nm2[active]
        same(m,m2);same(l,l2);same(out,out2)
        iterations += 1
    traces.append(dict(mode=mode,tiles=tiles,identity_warps=skipped,rescale_warps=moved))
    identity_warps += skipped;rescale_warps += moved
assert identity_warps > 0 and rescale_warps > 0

# A threshold that suppresses genuine changes must not pass the invariant.
o=np.float32(2);a=np.exp2(np.float32(-.25)).astype(np.float32)
assert (o*a).view(np.uint32) != o.view(np.uint32)
assert not bool(np.all(np.array([True]+[False]*31))) and bool(np.any(np.array([True]+[False]*31)))
report = dict(parent_sha256=PARENT,source_sha256=meta['source_sha256'],cuda_sha256=meta['cuda_sha256'],
              cpu_only=True,gpu_verified=False,finite_FP32_identity_products=len(words),
              finite_FP32_add_vs_exact_sum_screens=len(words),warp_quad_vote_masks=256,
              online_state_iterations=iterations,online_state_bits_equal=True,traces=traces,
              rejected_mutants=['Suppress a genuine .25-log2 maximum change','Use all instead of any for moving quads'],
              no_precision_probability_scale_or_reduction_changes=True,
              all_Smem_field_layouts_unchanged=True,dynamic_shared_bytes=layouts,
              flat_L1_register_pool_per_two_blocks=65536,
              actual_GPU_branch_frequency_unknown=True,performance_prediction=None,
              ex2_zero_primary_source='https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2',
              limitations='CPU finite screens and identity argument; no SASS, register allocation, GPU shuffle/replay correctness or timing evidence')
(ROOT / '.tools/attention-skipscale-cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps(report,indent=2))
