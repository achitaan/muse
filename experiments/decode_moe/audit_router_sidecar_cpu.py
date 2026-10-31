"""Source/SASS dependency audit and sequential counter replay model; never calls CUDA."""
import hashlib
import json
import random
import re
from screen_router_sidecar_cpu import OUT, derive, ownership


def main():
    parent, source = derive()
    audit = json.loads((OUT / 'cpu-audit.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest() == audit['candidate_CUDA_sha256']
    assert (OUT / 'candidate.cu').read_bytes() == source.encode()
    assert ownership() == audit['ownership']
    sass = (OUT / 'candidate.sass').read_text()
    ins = {int(a, 16): b.strip() for a, b in re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);', sass)}
    # Fixed compiler output; all assertions are source-bound and fail if the CFG changes.
    assert ins[0x20] == 'PREEXIT'
    assert ins[0x50] == 'ISETP.NE.AND P0, PT, R0, 0x80, PT'
    assert ins[0x90] == '@!P0 BRA 0x4300'
    assert ins[0xa0] == 'ISETP.GT.U32.AND P0, PT, R0, 0x80, PT'
    assert ins[0xb0] == '@P0 BRA 0x1410'
    assert ins[0x1430] == ins[0x4320] == ins[0xd890] == 'ACQBULK'
    assert ins[0x280] == 'CALL.REL.NOINC 0xd880'
    before = [(a, b) for a, b in ins.items() if a < 0x280 and 'LDG.' in b]
    assert len(before) == 4 and all('CONSTANT' in b for a, b in before)
    assert 'LDG.E.STRONG.GPU' in ins[0x1550]
    assert all(a > 0x1430 for a, b in ins.items() if 0x1410 <= a < 0x4300 and 'LDG.' in b)
    assert ins[0x4240] == 'BAR.SYNC.DEFER_BLOCKING 0x1, 0x400'
    assert ins[0x41f0] == 'MEMBAR.SC.GPU'
    assert 'ATOMG.E.ADD.STRONG.GPU' in ins[0x42e0]
    assert ins[0x42c0] == 'IMAD.SHL.U32 R5, R0, 0x2, RZ'
    assert 'ATOMG.E.ADD.STRONG.GPU' in ins[0x43a0] and ins[0x43a0].endswith(', RZ')
    assert '0x7f' in ins[0x43c0] and '0x80' in ins[0x4440]
    rng = random.Random(2026103104)
    # Parts not used by a smaller M retain their previous values in both arms.
    old_parts, new_parts = {}, {}
    counter = 0
    attempts = 0
    for generation in range(200):
        m = (1, 8, 2, 4)[generation % 4]
        order = list(range(64))
        rng.shuffle(order)
        assert counter == 0
        for done, b in enumerate(order, 1):
            ks, nb = b >> 3, b & 7
            for w in range(32):
                row = nb * 32 + w
                for t in range(m):
                    key = (ks, t, row)
                    value = (generation, ks, t, row)
                    new_parts[key] = value
                    old_b, old_w = 2 * b + w // 16, w % 16
                    old_key = (old_b >> 4, t, (old_b & 15) * 16 + old_w)
                    old_parts[old_key] = value
            # This model assumes the fenced writers have completed before publication.
            counter += 2
            attempts += 1
            assert (counter >= 128) == (done == 64)
        assert old_parts == new_parts
        assert all(new_parts[k][0] == generation for k in new_parts if k[1] < m)
        counter = 0  # unchanged top-k resets only after every route unit arrived
    record = dict(candidate_CUDA_sha256=audit['candidate_CUDA_sha256'],
                  cubin_sha256=audit['resources']['candidate']['cubin_sha256'],
                  outgoing_trigger_pc='0x20',
                  dispatch={'topk_CTA_128':'0x4300','router_CTA_129_to_192':'0x1410','pooler_CTA_0_to_127':'fallthrough'},
                  wait_pc={'router':'0x1430','topk':'0x4320','pooler_helper':'0xd890'},
                  independent_pooler_load_count=4, first_route_coherent_input_load_pc='0x1550',
                  publication={'writer_fence_pc':'0x41f0','1024_thread_barrier_pc':'0x4240','atomic_pc':'0x42e0','source_increment':2},
                  replay_model={'invocations':200,'counter_observations':attempts,'alternating_M':[1,8,2,4],
                                'entire_partial_buffers_equal':True,'stale_unused_rows_preserved':True},
                  GPU_jobs=0, admitted_for_GPU=False,
                  limitation='Fixed source/control-flow correspondence and integer protocol model. Instruction locations alone do not prove dynamic execution, GPU memory visibility, PDL liveness or bitwise numerical equality.',
                  next_gate='Exact-byte graph replay and paired cold-L2 joint dense-unit timings;no full-layer claim or Decode check yet.')
    (OUT / 'dependency-audit.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
