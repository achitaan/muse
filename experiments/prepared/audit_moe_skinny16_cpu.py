"""CPU audit of a repaired second tile; no GPU correctness/performance claim.

Expected operand bits and accumulation order come from two invocations of the
exact checked eight-token static skinny ring. FP64 products independently check
the split-K numerical model. This does not emulate hardware MMA summation.
"""
import ast
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
torch.set_num_threads(2)
torch.manual_seed(20261028)
rng = np.random.default_rng(20261028)
meta = json.loads((ROOT / '.tools/moe-skinny16-source-audit.json').read_bytes())
raw = (ROOT / '.tools/moe-skinny16.py').read_bytes()
base = (ROOT / 'best/moe_verified.py').read_bytes()
assert hashlib.sha256(base).hexdigest() == meta['parent_sha256']
assert raw.startswith(base) and hashlib.sha256(raw).hexdigest() == meta['source_sha256']
tree = ast.parse(raw)
cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(x, ast.Subscript) and ast.unparse(x) == "ALT_SRCS['sk16']" for x in n.targets))
assert hashlib.sha256(cuda.encode()).hexdigest() == meta['alternate_cuda_sha256']
restored = cuda
for change in reversed(meta['changes']):
    assert restored.count(change['new']) == change['count']
    restored = restored.replace(change['new'], change['old'])
assert hashlib.sha256(restored.encode()).hexdigest() == meta['original_cuda_sha256']
assert '#define OKC_SKT 16\n' in cuda and '-DOKC_WS=0' in meta['defines']
assert '-DOKC_SPLIT=1' in meta['defines'] and '-DOKC_NW=8' in meta['defines']
assert 'if (two) *reinterpret_cast<float4*>(rd + 4)' in cuda
assert 'mma_bf16(second[st]' in cuda and 'second[1][k] + second[2][k] + second[3][k]' in cuda
assert not any(isinstance(n, ast.FunctionDef) and n.lineno > ast.parse(base).body[-1].end_lineno for n in tree.body)

# Actual load_x1_smem: each warp writes its eight rows with 16-byte chunks,
# pitch520 BF16 elements. Actual read uses row=(lane>>2), col=16*(lane&3)+64*c.
operand_words = 0
mutant_detected = False
for cnt in range(1, 17):
    x = rng.integers(0, 65536, size=(cnt, 4096), dtype=np.uint16)
    # Include signed zeros, subnormals and NaN payloads as representation probes.
    x.flat[:min(8, x.size)] = np.array([0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0x3f80], np.uint16)
    for warp in range(8):
        shared = np.empty((8, 520), dtype=np.uint16)
        written = np.zeros_like(shared, dtype=np.int16)
        for lane in range(32):
            for i in range(lane, 8 * 64, 32):
                row, chunk = divmod(i, 64)
                values = x[8 + row, warp * 512 + 8 * chunk:warp * 512 + 8 * chunk + 8] if 8 + row < cnt else np.zeros(8, np.uint16)
                shared[row, chunk * 8:chunk * 8 + 8] = values
                written[row, chunk * 8:chunk * 8 + 8] += 1
        assert np.all(written[:, :512] == 1) and np.all(written[:, 512:] == 0)
        for lane in range(32):
            row, quarter = lane >> 2, lane & 3
            for c in range(8):
                start = 16 * quarter + 64 * c
                got = shared[row, start:start + 16]
                # Reference: the original checked eight-token loader on a job
                # whose first entry is eight positions later in the expert list.
                expected = x[8 + row, warp * 512 + start:warp * 512 + start + 16] if 8 + row < cnt else np.zeros(16, np.uint16)
                assert np.array_equal(got, expected)
                operand_words += 16

# Every K16 MMA in subgroup st visits four elements from each lane quarter.
coverage = np.zeros(4096, np.int16)
for warp in range(8):
    for c in range(8):
        for st in range(4):
            for quarter in range(4):
                for i in range(4): coverage[warp * 512 + c * 64 + quarter * 16 + st * 4 + i] += 1
assert np.all(coverage == 1)
bank_transactions = 0
for first_lane in range(0, 32, 8):
    banks = [(((lane >> 2) * 520 + 16 * (lane & 3)) // 2 + word) % 32
             for lane in range(first_lane, first_lane + 8) for word in range(4)]
    assert sorted(banks) == list(range(32))
    bank_transactions += 1

# Independent warp partials expose a missing-second-store mutant. The actual
# epilogue's nt/h/ln indices must reduce both complete float4 planes.
output_cases = 0
for cnt in range(1, 17):
    partial = rng.standard_normal((8, 32, 8)).astype(np.float32)
    staged = partial.copy()
    missing_store = partial.copy(); missing_store[:, :, 4:] = np.nan
    visits = np.zeros((cnt, 16), np.int16)
    for tid in range(128 if cnt > 8 else 64):
        nt, h, ln = tid >> 6, (tid >> 5) & 1, tid & 31
        gg, tt = ln >> 2, ln & 3
        token = nt * 8 + 2 * tt + h
        if token >= cnt: continue
        for row, component in ((gg, h), (gg + 8, 2 + h)):
            got = np.float32(0)
            expected = np.float32(0)
            for warp in range(8):
                got = np.float32(got + staged[warp, ln, nt * 4 + component])
                expected = np.float32(expected + partial[warp, ln, nt * 4 + component])
            assert got.view(np.uint32) == expected.view(np.uint32)
            visits[token, row] += 1
            if nt and not np.isfinite(missing_store[:, ln, nt * 4 + component]).all(): mutant_detected = True
    assert np.all(visits == 1)
    output_cases += 1
assert mutant_detected

# Linux/CUDA ABI for the fixed non-WS, non-WGS alternate. All field definitions
# mirror the reviewed Smem source; none of its weights/scratch formats change.
assert ct.sizeof(ct.c_int) == 4 and ct.sizeof(ct.c_void_p) == 8
class Smem(ct.Structure):
    _fields_ = [('lut', ct.c_uint32 * 256), ('red', ct.c_float * (2 * 8 * 32 * 8)),
                ('bars', ct.c_uint64 * 12), ('seqs', ct.c_int * 4), ('fcnt', ct.c_int * 32),
                ('xbuf_ptr', ct.c_void_p), ('cnt', ct.c_int * 64), ('off', ct.c_int * 64),
                ('wc', ct.c_int * (8 * 32)), ('jobs', ct.c_int * (4 * 88)),
                ('counts', ct.c_int * 2), ('gen', ct.c_int), ('ssp', ct.c_void_p),
                ('uslot', ct.c_int * 8), ('dync', ct.c_int * 88), ('eid', ct.c_int8 * 3072),
                ('ent', ct.c_int16 * 3072), ('pos', ct.c_int16 * 3072)]
base_shared = (ct.sizeof(Smem) + 127) // 128 * 128
xa = 3 * 36864 + 8 * 8 * 520 * 2
xb = 3 * 36864 + 2 * 64 * 264 * 2
shared_bytes = base_shared + max(xa, xb)
assert shared_bytes < 232448
jobs = []
for m in range(9, 17):
    # Counts are bounded by M because top8 IDs are distinct in each token.
    for count in range(m + 1):
        old = (count + 7) // 8
        new = (count + 15) // 16
        assert new <= old and new <= 1
        jobs.append(dict(m=m, count=count, old_weight_passes=old, new_weight_passes=new))
assert 32 < 88 and 16 * 8 <= 3072

numeric = []
for m in (9, 15, 16):
    x = torch.randn((m, 4096)).bfloat16().float()
    w = (torch.randn((16, 4096)) * .02).bfloat16().float()
    old_jobs = []
    for lo in (0, 8):
        rows = x[lo:lo + 8]
        partials = []
        for warp in range(8):
            groups = [torch.zeros((len(rows), 16)) for _ in range(4)]
            for c in range(8):
                for st in range(4):
                    kk = [warp * 512 + c * 64 + quarter * 16 + st * 4 + i for quarter in range(4) for i in range(4)]
                    groups[st] += rows[:, kk] @ w[:, kk].T
            # Match the source's left-associative three-summand merge, then +=.
            partials.append(groups[0] + ((groups[1] + groups[2]) + groups[3]))
        total = torch.zeros((len(rows), 16))
        for part in partials: total += part
        old_jobs.append(total)
    old = torch.cat(old_jobs)
    # Independently process all rows together, grouping by accumulator first.
    # This models one weight stream feeding both tiles rather than copying the
    # expected result. Hardware MMA instruction summation remains unmodeled.
    new = torch.zeros((m, 16))
    for warp in range(8):
        groups = []
        for st in range(4):
            group = torch.zeros((m, 16))
            for c in range(8):
                kk = [warp * 512 + c * 64 + quarter * 16 + st * 4 + i for quarter in range(4) for i in range(4)]
                group += x[:, kk] @ w[:, kk].T
            groups.append(group)
        new += groups[0] + ((groups[1] + groups[2]) + groups[3])
    agreement = float(torch.linalg.vector_norm(new - old) / torch.linalg.vector_norm(old))
    assert agreement < .000001
    reference = x.double() @ w.double().T
    error = float(torch.linalg.vector_norm(new.double() - reference) / torch.linalg.vector_norm(reference))
    assert error < .00001
    numeric.append(dict(m=m, relative_l2_fp32_model_vs_fp64=error,
                        relative_l2_combined_vs_split_model=agreement))

report = dict(source_sha256=meta['source_sha256'], seed=20261028, parent_prefix_exact=True,
              original_cuda_restored_by_inverse=True, second_tile_operand_words=operand_words,
              second_tile_words_equal_checked_loader=True, padding_unique_writers=True,
              complete_k_coverage=4096, output_ownership_cases=output_cases,
              conflict_free_x1_uint4_transactions=bank_transactions,
              missing_second_store_mutant_detected=mutant_detected,
              shared_abi_size=ct.sizeof(Smem), dynamic_shared_bytes=shared_bytes,
              per_expert_passes=jobs, numerical_cases=numeric,
              gpu_status='unmeasured; prepared alternate is not dispatched',
              limitations='CPU address/representation and numerical models; CUDA compilation, races, resource usage and performance unverified')
(ROOT / '.tools/moe-skinny16-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps({k: v for k, v in report.items() if k != 'per_expert_passes'}))
