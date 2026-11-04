"""CPU-only proofs of RSTREAM packing, addressing, arithmetic and scheduling.

Run: uv run --offline --no-project --with torch --with numpy python
     experiments/moe_restart/validate_cpu.py

Expected values below come from the NVFP4 E2M1/E4M3 definitions and from
independent row-major matrix/routing formulas, not from another GPU kernel.
This is NOT a GPU compilation or race-safety test.
"""
import ast
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name('moe_restart.py')
spec = importlib.util.spec_from_file_location('restart', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rng = np.random.default_rng(721539)
torch.manual_seed(721539)
torch.set_num_threads(4)
results = {}


def npad(c):
    return next(n for n in (8, 16, 32, 64, 96, 128, 192, 256) if n >= c)


def bf16(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def rotl(x, n):
    return (x << np.uint32(n)) | (x >> np.uint32(32-n))


def dq(q):
    return (q & np.uint32(0x81c081c0),
            rotl(q, 10) & np.uint32(0x81c081c0),
            rotl(q, 13) & np.uint32(0x81c081c0),
            ((q & np.uint32(0x001b001b)) * np.uint32(0x1084)) & np.uint32(0x81c081c0))


def validate_pack():
    rows = 512
    codes = torch.randint(0, 256, (rows, 2048), dtype=torch.uint8)
    sf = torch.randint(8, 127, (rows, 256), dtype=torch.uint8)
    # Nontrivial row permutation reproduces interleaved gate/up groups.
    rr = torch.stack((torch.arange(256).view(-1, 8),
                      torch.arange(256, 512).view(-1, 8)), 1).flatten()
    raw = codes.numpy()
    nib = np.stack((raw & 15, raw >> 4), -1).reshape(rows, 4096)
    e2m1 = np.array((0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6), np.float32)
    wgs = mod._wgs_copy(codes, sf, rr)
    dummy = [None] * 14
    dummy[12] = dummy[13] = wgs
    dummy[2] = dummy[5] = dummy[7] = torch.ones(1)
    converted = mod._restart_weights({}, {'w': dummy})
    for tile in (64, 128):
        packed = mod._restart_pack(codes, sf, rr, tile)
        # Independent direct packing and the fast WGS byte reorder agree.
        assert torch.equal(packed, converted[tile][0])
        assert torch.equal(packed, converted[tile][1])
        assert packed.numel() == rows * 4096 * 9 // 16
        wg = tile // 64
        p = packed.numpy().reshape(rows//tile, 64, wg, 2304)
        q = p[..., :2048].copy().view(np.uint32).reshape(rows//tile, 64, wg, 4, 32, 4)
        ss = p[..., 2048:].reshape(rows//tile, 64, wg, 4, 8, 4, 2)
        ub, kg, group, warp, lane, step = np.indices(q.shape)
        base_row = ub*tile + group*64 + warp*16 + lane//4
        base_col = kg*64 + step*16 + 2*(lane%4)
        for reg, (h, upper_k) in enumerate(((0,0),(1,0),(0,1),(1,1))):
            word = dq(q)[reg]
            for upper in (0, 1):
                got = bf16((word >> (16*upper)) & 0xffff) * np.float32(2.0**126)
                original_row = rr.numpy()[base_row + h*8]
                original_col = base_col + upper_k*8 + upper
                # E2M1 mapping independently fixes both fragment row and K.
                expected = e2m1[nib[original_row, original_col]]
                assert np.array_equal(got, expected), (tile, reg, upper)
        for ub0 in range(rows//tile):
            for kg0 in range(64):
                for wg0 in range(wg):
                    for w in range(4):
                        for g in range(8):
                            for j in range(4):
                                for h in range(2):
                                    row = rr[ub0*tile+wg0*64+w*16+g+h*8]
                                    assert ss[ub0,kg0,wg0,w,g,j,h] == sf[row,kg0*4+j]
    results['packing'] = '64/128-row full-K code and scale roundtrips; WGS conversion identical'


def validate_decoder():
    magnitudes = torch.tensor((0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6), dtype=torch.float64)
    scale_bytes = torch.arange(8, 127, dtype=torch.uint8)
    sf = scale_bytes.view(torch.float8_e4m3fn).float().double()
    for code in range(16):
        # dq4 places E2M1 bits at (6,7,8,15) in each BF16 half.
        bits = sum(((code >> i)&1) << bit for i, bit in enumerate((6,7,8,15)))
        raw = float(bf16(bits))
        for byte, scale in zip(scale_bytes.tolist(), sf.tolist()):
            rebased = float(bf16(byte*16 + 0x5c00))
            # The specified E2M1 * specified FP8 block scale, with 2^-62
            # encoding, is exact before entering the BF16 tensor core.
            expected = float(magnitudes[code]) * scale * 2.0**-62
            assert raw * rebased == expected, (code, byte)
            assert float(torch.tensor(raw*rebased).bfloat16().float()) == expected
    # Explicitly prove selector bytes for both adjacent scale pairs.
    for a in range(8, 127):
        for b in range(8, 127):
            word = np.uint32(a | (b<<16)) * np.uint32(16) + np.uint32(0x5c005c00)
            assert int(word & np.uint32(65535)) == a*16+0x5c00
            assert int(word >> np.uint32(16)) == b*16+0x5c00
    results['dequant'] = 'all 16 codes x 119 allowed FP8 bytes exact; paired scale arithmetic exact'


def validate_routing_and_x():
    draws = 0
    for m in (1,4,8,16,32,33,64,65,96,97,128,129,192,193,256):
        for mode in ('random', 'hot', 'all_local', 'remote'):
            ids = np.stack([rng.choice(256, 8, replace=False) for _ in range(m)])
            if mode == 'hot': ids[:,0] = 0; ids[:,1:] = np.arange(32,39)
            if mode == 'all_local': ids[:] = np.arange(8)
            if mode == 'remote': ids[:] = np.arange(32,40)
            counts = [(ids == e).sum() for e in range(32)]
            pads = [npad(c) if c else 0 for c in counts]
            offsets = np.concatenate(([0],np.cumsum(pads)))
            assert offsets[-1] <= 8192
            assert offsets[-1] <= 2*8*m+7*32
            ent = [np.flatnonzero(ids.flatten() == e) for e in range(32)]
            # CUDA ballot prefix and stable row order, independently emulated.
            for e in range(32):
                ballot_ent = []
                for base in range(0,8*m,32):
                    hits = [l for l in range(32) if base+l < 8*m and ids.flat[base+l] == e]
                    ballot_ent += [base+l for l in hits]
                assert np.array_equal(ballot_ent, ent[e])
            for e in range(32):
                if not pads[e]: continue
                # Every token/column maps bijectively to the expert's packed
                # range. This proves gather + B descriptors + A epilogue use
                # exactly the same layout, including padding and stage edges.
                token, col = np.indices((pads[e], 4096))
                ix = offsets[e]*4096 + (col//64)*pads[e]*64 + (token//8)*512 + ((col%64)//8)*64 + (token%8)*8 + col%8
                unique = np.unique(ix)
                assert unique.size == pads[e]*4096
                assert unique[0] == offsets[e]*4096
                assert unique[-1]+1 == offsets[e+1]*4096
                # WGMMA K16 descriptor adds 16 units of 16 bytes per step.
                for step in range(4):
                    assert (ix[0,step*16] - ix[0,0])*2 == step*256
                if pads[e] >= 16:
                    assert (ix[8,0] - ix[0,0])*2 == 1024
            draws += 1
    results['routing_activation'] = f'{draws} routings incl empty/hot/full local; descriptor bijection and capacity'


def validate_ownership_and_ring():
    plans = 0
    for grid in (1,2,8,66,132):
        for active in range(33):
            for rows in (64,128):
                for total_rows in (4096,8192):
                    total = active * (total_rows // rows)
                    jobs = [i for i in range(grid) if i < total]
                    queue = grid
                    while queue < total:
                        jobs.append(queue); queue += 1
                    assert sorted(jobs) == list(range(total))
                    if active:
                        owned = {(u%active,u//active) for u in jobs}
                        assert len(owned) == total
                    plans += 1
    for look in (3,6):
        window = list(range(look))
        loads = list(window)
        consumed = []
        for k in range(64):
            consumed.append(window[0])
            window[:-1] = window[1:]
            if k+look < 64:
                window[-1] = k+look
                loads.append(k+look)
        assert loads == list(range(64))
        assert consumed == list(range(64))
    # Ring parity and register ping-pong: input af[group] can be rewritten
    # only after its earlier WGMMA has retired. wait_group<1> enforces it.
    for ntile in (0,1,2,7,32):
        inflight = []
        register_live = set()
        arrivals = [0,0,0]
        produced = [0,0,0]
        for seq in range(ntile*64):
            slot = seq%3
            produced[slot] += 1
            for cg in (0,1):
                assert cg not in register_live
                register_live.add(cg); inflight.append((seq,cg,slot))
                while len(inflight) > 1:
                    _, retired, _ = inflight.pop(0); register_live.remove(retired)
                if cg == 0 and seq:
                    arrivals[(seq-1)%3] += 1
        if ntile:
            inflight.clear(); register_live.clear(); arrivals[(ntile*64-1)%3] += 1
        assert arrivals == produced
    results['ownership_ring'] = f'{plans} whole-output-tile plans; 3/6-group unique prefetches; 0/1/2/7/32-tile parity/register lifetimes'


def validate_arithmetic():
    # Analytical manufactured MLP: both matrices have only K0 nonzero. No
    # reduction-order ambiguity exists, so the exact reference follows from
    # scalar E2M1 values and the explicitly required BF16 boundaries.
    for m in (1,8,33):
        x = torch.zeros(m,64, dtype=torch.bfloat16)
        x[:,0] = torch.arange(1,m+1).bfloat16() / 8
        a = torch.zeros(128,64)
        a[:64,0] = .5
        a[64:,0] = 1.5
        b = torch.zeros(64,64); b[:,0] = 2
        gs_a = torch.tensor(.75).bfloat16()
        gs_b = torch.tensor(.5).bfloat16()
        h = ((x.float() @ a.T).bfloat16() * gs_a).bfloat16().float()
        inter = (torch.nn.functional.silu(h[:,:64]) * h[:,64:]).bfloat16()
        y = ((inter.float() @ b.T).bfloat16() * gs_b).bfloat16().float()
        gate = ((x[:,0].float() * .5).bfloat16() * gs_a).bfloat16().float()
        up = ((x[:,0].float() * 1.5).bfloat16() * gs_a).bfloat16().float()
        expected_inter = (torch.nn.functional.silu(gate)*up).bfloat16().float()
        expected_y = ((expected_inter*2).bfloat16()*gs_b).bfloat16().float()
        assert torch.equal(y, expected_y[:,None].expand(-1,64))
        norm = y * torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)
        expected_norm = expected_y * torch.rsqrt(expected_y.square()+1e-8)
        assert torch.equal(norm, expected_norm[:,None].expand(-1,64))
    results['arithmetic'] = 'analytical one-nonzero-column MLP, BF16 GEMM/GS/SwiGLU boundaries and RMSNorm'


def validate_static():
    tree = ast.parse(SOURCE.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    names = [n.module if isinstance(n,ast.ImportFrom) else n.names[0].name for n in imports]
    assert not any(n.startswith(('subprocess','reference','nvfp4','triton','requests')) for n in names)
    assert mod.RESTART_CUDA.count('wgmma.mma_async.sync.aligned.') == 16
    assert '__MMA_SOURCE__' not in mod.RESTART_CUDA
    assert 'cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes' in mod.RESTART_CUDA
    # No compressed/dequantized weight writes to shared memory in the new core.
    assert 'ring + slot * N * KGROUP * 2' in mod.RESTART_CUDA
    assert 'griddepcontrol' not in mod.RESTART_CUDA
    assert not torch.cuda.is_available()
    for n in (8,16,32,64,96,128,192,256):
        assert f'RSMMA<{n}>' in mod.RESTART_CUDA
        body = mod._restart_mma_source().split(f'template<> struct RSMMA<{n}> {{')[1].split('\n};')[0]
        first, rest = body.split('} else {')
        nr = n // 2
        # PTX RS BF16 shapes use N/2 FP32 outputs, four 32-bit A registers,
        # one 64-bit B descriptor; output-only first, read/write afterwards.
        assert list(map(int,re.findall(r'"=f"\(d\[(\d+)\]\)',first))) == list(range(nr))
        assert list(map(int,re.findall(r'"\+f"\(d\[(\d+)\]\)',rest))) == list(range(nr))
        for fragment in (first,rest):
            assert list(map(int,re.findall(r'%(\d+)',fragment))) == list(range(nr+5))
            assert list(map(int,re.findall(r'"r"\(a\[(\d+)\]\)',fragment))) == list(range(4))
            assert fragment.count('"l"(b)') == 1
    # Verify host declaration and definition in the same frontend TU too.
    assert 'cudaFuncAttributeMaxDynamicSharedMemorySize' in mod.RESTART_CUDA
    assert 'grid > 0 && grid <= sms' in mod.RESTART_CUDA
    (ROOT/'results/moe_restart_expanded.cu').write_text(mod.RESTART_CUDA)
    (ROOT/'results/moe_restart_instantiated.cu').write_text(mod.RESTART_CUDA +
        '\nstatic_assert(sizeof(rstream::Shared) == 21376);\n'
        'static_assert(sizeof(rstream::Shared) + 3 * 256 * 64 * 2 == 119680);\n'
        '\ntemplate __global__ void rstream::kernel<64,8>(rstream::Params);\n'
        'template __global__ void rstream::kernel<128,256>(rstream::Params);\n')
    (ROOT/'results/moe_restart_host_stub.cu').write_text(mod.RESTART_CPP + mod.RESTART_CUDA)
    results['python_static'] = 'AST/import, all 8 generated MMA shapes/operands/constraints, host/device signature TU, empty fallback guard; no GPU locally'


def validate_admission():
    # Expected policy follows the documented 3%-both-orders, no-regression,
    # 6/8-draw and correctness contract, not measurements from a GPU.
    assert mod._restart_admission([[.95]*8,[.95]*8], True)[0]
    assert not mod._restart_admission([[.95]*8,[.95]*8], False)[0]
    assert not mod._restart_admission([[.95]*8,[.98]*8], True)[0]
    assert not mod._restart_admission([[.90]*7+[1.02],[.95]*8], True)[0]
    assert not mod._restart_admission([[.90]*5+[1.0]*3,[.95]*8], True)[0]
    assert not mod._restart_admission([[.90]*7,[.95]*8], True)[0]
    assert not mod._restart_admission([[],[]], True)[0]
    # Exact zero and explicit nonfinite rejection, including near-zero rows.
    zero = torch.zeros(2,64,dtype=torch.bfloat16)
    assert mod._restart_error(zero,zero)[0]
    assert not mod._restart_error(zero+1,zero)[0]
    nan = zero.clone(); nan[0,0] = float('nan')
    assert not mod._restart_error(nan,zero)[0]
    results['admission'] = '7 policy cases; zero/near-zero/nonfinite numerical guard cases'


if __name__ == '__main__':
    validate_pack()
    validate_decoder()
    validate_routing_and_x()
    validate_ownership_and_ring()
    validate_arithmetic()
    validate_admission()
    validate_static()
    print(json.dumps(results, indent=2))
    (ROOT/'results/moe_restart_cpu.json').write_text(json.dumps(results,indent=2)+'\n')
