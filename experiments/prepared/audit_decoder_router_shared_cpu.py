"""Bit-preserving copy/index oracle for the exact FP32 router cache patch."""
import ast
import hashlib
import json
import re
import numpy as np

from build_decoder_router_shared import ROOT, BASE_SHA, ANCHOR, COPY, READ_OLD, READ_NEW


def literal(tree):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == '_RF_CUDA' for t in n.targets))


def audit():
    raw = (ROOT/'.tools/decoder-router-shared-variant.py').read_bytes()
    base = (ROOT/'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(base).hexdigest() == BASE_SHA
    bt, ct = ast.parse(base),ast.parse(raw)
    old, new = literal(bt),literal(ct)
    assert new.count(COPY) == new.count(READ_NEW) == 1
    restored = new.replace('route_fp32_xshared','route_fp32').replace(COPY,'',1).replace(READ_NEW,READ_OLD,1)
    assert restored == old
    for node in ct.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == '_RF_CUDA' for t in node.targets):
            node.value = ast.Constant(old)
    assert ast.dump(bt,include_attributes=False) == ast.dump(ct,include_attributes=False)
    assert new.count('__syncthreads();') == old.count('__syncthreads();')+1
    assert new.index(COPY) < new.index(ANCHOR) < new.index('if (!last || warp >= M) return;')
    assert new.count('__fmaf_rn') == old.count('__fmaf_rn') == 2
    assert 'route_fp32_xshared<1><<<128, 256, 0, stream>>>' in new
    assert 'route_fp32_xshared<8><<<128, 256, 0, stream>>>' in new
    assert '((old & 127u) == 127u)' in new and 'atomicAdd(counter, 1u)' in new
    # Bind this oracle to the actual C++ map, not an unrelated copy model.
    assert 'i < M * 1024; i += blockDim.x' in COPY
    assert 'xshared[i] = x[(size_t)(i >> 10) * 8192 + ks * 1024 + (i & 1023)]' in COPY
    assert READ_NEW == 'const float v = xshared[m * 1024 + j * 32 + lane];'
    rng = np.random.default_rng(20261028)
    specials = np.array([0,0x80000000,1,0x007fffff,0x00800000,0x3f800000,
                         0x7f7fffff,0x7f800000,0xff800000,0x7fc00001],dtype=np.uint32)
    cases, compared, written = [],0,0
    for m in range(1,9):
        tm = 1 if m == 1 else 8
        shared = np.full(tm*1024,0x7fc0dead,dtype=np.uint32)
        for replay in range(2):
            x = rng.integers(0,2**32,size=(m,8192),dtype=np.uint32)
            x.ravel()[::19] = np.resize(specials,x.ravel()[::19].size)
            for ks in range(8):
                # Different replay/partition poisons detect any missing write.
                shared.fill(np.uint32(0x7fc0dead+replay))
                writers = np.zeros(tm*1024,dtype=np.uint8)
                for tid in range(256):
                    i = np.arange(tid,m*1024,256,dtype=np.int64)
                    shared[i] = x.ravel()[(i>>10)*8192+ks*1024+(i&1023)]
                    writers[i] += 1
                assert np.all(writers[:m*1024] == 1) and not np.any(writers[m*1024:])
                # Oracle is the original x[:,k_split] slice. Comparing uint32
                # tests representation, including every sign/subnormal bit.
                expected = x[:,ks*1024:(ks+1)*1024].ravel()
                assert np.array_equal(shared[:m*1024],expected)
                for token in range(m):
                    for j in range(32):
                        lanes = np.arange(32)
                        original = x[token,ks*1024+j*32+lanes]
                        cached = shared[token*1024+j*32+lanes]
                        assert np.array_equal(cached,original)
                        assert len(set(((token*1024+j*32+lanes)%32).tolist())) == 32
                        compared += 8*32 # identical input used by all eight warps
                written += m*1024
        cases.append({'m':m,'template':tm,'shared_bytes':tm*4096,'replays':2,'k_splits':8})
    # Independent complete expert-row coverage of the unchanged 128-CTA grid.
    rows = [nb*16+warp*2+parity for nb in range(16) for warp in range(8) for parity in (0,1)]
    assert sorted(rows) == list(range(256))
    ks_columns = [ks*1024+j*32+lane for ks in range(8) for j in range(32) for lane in range(32)]
    assert sorted(ks_columns) == list(range(8192))
    record = {'source_sha256':hashlib.sha256(raw).hexdigest(),'base_sha256':BASE_SHA,
              'original_module_ast_recovered':True,'complete_cuda_edit_reversible':True,
              'seed':20261028,'numpy_version':np.__version__,'active_cases':cases,
              'bitwise_operand_comparisons':compared,'shared_words_written':written,
              'unique_writers_and_no_uninitialized_active_reads':True,
              'bank_conflict_free_lane_map':True,'fresh_replay_overwrite':True,
              'all_expert_rows_and_k_columns_covered':True,
              'fma_shuffle_routing_counter_protocol_unchanged':True,
              'scope_limit':'CPU representation/control proof; actual CUDA compilation, resource use, numerical results and timing unverified'}
    (ROOT/'.tools/decoder-router-shared-cpu-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    audit()
