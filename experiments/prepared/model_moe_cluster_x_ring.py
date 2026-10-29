"""CPU event model of the proposed X ring, not a CUDA memory-model proof."""
import ast
import hashlib
import json
from pathlib import Path
import random

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())


class ProtocolViolation(Exception):
    def __init__(self, reason, trace):
        self.reason, self.trace = reason, trace[-64:]
        super().__init__(reason)


def simulate(copies, seed, mode='random', mutant=None):
    slots, warps = 3, 16
    rng = random.Random(seed)
    init, live = [False, False], [True, True]
    full_phase = [[0] * slots for _ in range(2)]
    full_remaining = [[None] * slots for _ in range(2)]
    full_arm_generation = [[None] * slots for _ in range(2)]
    empty_phase = [[0] * slots for _ in range(2)]
    expected = warps if mutant == 'one_CTA_empty_count' else 2 * warps
    empty_pending = [[expected] * slots for _ in range(2)]
    payload = [[None] * slots for _ in range(2)]
    issued = [0] * (2 * warps)
    active = [set() for _ in range(2 * warps)]
    finished = [set() for _ in range(2 * warps)]
    sent = [0] * (2 * warps)
    transfers, acks, completed = {}, {}, set()
    next_copy, drained = 0, False
    trace, checksum = [], hashlib.sha256()
    count = 0

    def fail(reason):
        raise ProtocolViolation(reason, trace)

    def phase_done(slot, use):
        return empty_phase[0][slot] % 2 != use % 2

    while True:
        consumers_done = all(n == copies for n in sent)
        if drained and consumers_done and not acks and not transfers:
            if any(active) or any(finished):
                fail('unfinished_consumers')
            break
        actions = []
        for rank in (0, 1):
            if not init[rank]:
                actions.append(('init', rank))
        if mutant == 'peer_CTA_exits' and init[1] and live[1]:
            actions.append(('exit_peer', 1))
        startup_ready = init[0] and (init[1] or mutant == 'missing_startup_join')
        if startup_ready and next_copy < copies:
            slot, use = next_copy % slots, next_copy // slots
            if use == 0 or phase_done(slot, use - 1):
                actions.append(('issue', next_copy))
        for n, rank in transfers:
            actions.append(('chunk', n, rank))
        for actor in range(2 * warps):
            rank = actor // warps
            n = issued[actor]
            if init[rank] and live[rank] and n < copies and len(active[actor]) < 2:
                slot, use = n % slots, n // slots
                if full_phase[rank][slot] % 2 != use % 2:
                    actions.append(('start', actor, n))
            for n in sorted(active[actor]):
                actions.append(('finish', actor, n))
            for n in sorted(finished[actor]):
                actions.append(('send', actor, n))
        for n, actor in acks:
            actions.append(('deliver', actor, n))
        if next_copy == copies and not drained and startup_ready:
            used = range(min(slots, copies))
            if all(phase_done(x, (copies - 1 - x) // slots) for x in used):
                actions.append(('drain',))
        if not actions:
            fail('deadlock')

        if mode == 'random':
            event = rng.choice(actions)
        else:
            favored = 0 if mode == 'leader_fast' else 1

            def priority(event):
                kind = event[0]
                if kind == 'exit_peer':
                    return 100
                if kind == 'issue':
                    return 90
                if kind == 'init':
                    return 95 if event[1] == 0 else 1
                if kind == 'chunk':
                    return 80 if event[2] == favored else 10
                if kind in ('start', 'finish', 'send', 'deliver'):
                    return 70 if event[1] // warps == favored else 5
                return 60

            event = max(actions, key=priority)
        trace.append(event)
        checksum.update((repr(event) + '\n').encode())
        count += 1
        if count > 2000 + copies * 512:
            fail('event_bound')
        kind = event[0]
        if kind == 'init':
            init[event[1]] = True
        elif kind == 'exit_peer':
            live[1] = False
        elif kind == 'issue':
            n = event[1]
            slot, use = n % slots, n // slots
            for rank in (0, 1):
                if not init[rank]:
                    fail('remote_access_before_init')
                if not live[rank]:
                    fail('remote_access_after_exit')
                if full_phase[rank][slot] != use or full_remaining[rank][slot] is not None:
                    fail('rearm_before_completion')
                size = 4096 * (1 + (7 * n) % 8)
                full_remaining[rank][slot] = size // 2 if mutant == 'half_peer_tx_bytes' and rank == 1 else size
                full_arm_generation[rank][slot] = n
                transfers[n, rank] = size
            next_copy += 1
        elif kind == 'chunk':
            _, n, rank = event
            slot = n % slots
            if not live[rank]:
                fail('remote_access_after_exit')
            for actor in range(rank * warps, (rank + 1) * warps):
                if any(k != n and k % slots == slot for k in active[actor]):
                    fail('operand_overwrite')
            if full_arm_generation[rank][slot] != n or full_remaining[rank][slot] is None:
                fail('tx_generation_mismatch')
            size = min(4096, transfers[n, rank])
            if full_remaining[rank][slot] < size:
                fail('tx_underflow')
            payload[rank][slot] = n
            transfers[n, rank] -= size
            full_remaining[rank][slot] -= size
            if full_remaining[rank][slot] == 0:
                full_remaining[rank][slot] = None
                full_phase[rank][slot] += 1
            if transfers[n, rank] == 0:
                del transfers[n, rank]
                completed.add((n, rank))
        elif kind == 'start':
            _, actor, n = event
            rank, slot = actor // warps, n % slots
            if payload[rank][slot] != n or (n, rank) not in completed:
                fail('read_before_payload_complete')
            active[actor].add(n)
            issued[actor] += 1
        elif kind == 'finish':
            _, actor, n = event
            rank, slot = actor // warps, n % slots
            if payload[rank][slot] != n:
                fail('operand_overwrite')
            active[actor].remove(n)
            finished[actor].add(n)
        elif kind == 'send':
            _, actor, n = event
            finished[actor].remove(n)
            acks[n, actor] = None
            sent[actor] += 1
        elif kind == 'deliver':
            _, actor, n = event
            origin, slot, use = actor // warps, n % slots, n // slots
            target = origin if mutant == 'peer_releases_private_barrier' else 0
            if target != 0:
                fail('wrong_empty_target')
            if not live[target]:
                fail('remote_access_after_exit')
            if empty_phase[target][slot] != use:
                fail('late_ack_wrong_generation')
            empty_pending[target][slot] -= 1
            if empty_pending[target][slot] == 0:
                empty_phase[target][slot] += 1
                empty_pending[target][slot] = expected
            del acks[n, actor]
        elif kind == 'drain':
            for slot in range(min(slots, copies)):
                expected_phases = 1 + (copies - 1 - slot) // slots
                if empty_phase[0][slot] != expected_phases:
                    fail('drain_without_all_releases')
            if any(active) or any(finished) or acks or transfers or not consumers_done:
                fail('drain_without_all_releases')
            drained = True
        else:
            raise RuntimeError(kind)
    return dict(copies=copies, seed=seed, mode=mode, events=count,
                event_sha256=checksum.hexdigest(), final_empty_phases=empty_phase[0])


def main():
    rows = []
    for copies in (0, 1, 2, 3, 4, 5, 8, 16, 31, 32, 65):
        for seed in range(8 if copies > 16 else 16):
            rows.append(simulate(copies, seed))
        for mode in ('leader_fast', 'peer_fast'):
            rows.append(simulate(copies, 0, mode))
    rejected = {}
    allowed = {
        'missing_startup_join': {'remote_access_before_init'},
        'one_CTA_empty_count': {'rearm_before_completion', 'operand_overwrite', 'late_ack_wrong_generation', 'drain_without_all_releases', 'deadlock'},
        'half_peer_tx_bytes': {'read_before_payload_complete', 'tx_generation_mismatch', 'tx_underflow'},
        'peer_releases_private_barrier': {'wrong_empty_target'},
        'peer_CTA_exits': {'remote_access_after_exit'},
    }
    for mutant, reasons in allowed.items():
        try:
            simulate(8, 0, 'leader_fast', mutant)
        except ProtocolViolation as exc:
            assert exc.reason in reasons, (mutant, exc.reason)
            rejected[mutant] = dict(reason=exc.reason, tail=exc.trace)
        else:
            raise AssertionError(('Mutant survived', mutant))
    report = dict(kind='CPU event model, finite interleavings; not a hardware memory-order or complete-MoE deadlock proof',
                  model_AST_sha256=hashlib.sha256(ast.dump(ast.parse(Path(__file__).read_bytes()), include_attributes=False).encode()).hexdigest(),
                  CTAs=2, consumer_warps_per_CTA=16, slots=3, max_overlapping_reads_per_warp=2,
                  variable_copy_bytes=[4096 * n for n in range(1, 9)],
                  positive_cases=len(rows), rows=rows, rejected_mutants=rejected,
                  delayed_remote_arrivals=True, separate_multicast_receiver_completion=True,
                  last_slot_parity_and_drain_checked=True, kernel_execution=False, GPU_jobs=0,
                  limitations=['Abstracts WGMMA reads and TMA/DSM progress as causal events',
                               'Does not prove PTX scopes/proxy semantics, residency or graph capture',
                               'Does not include inter-cluster phase-A readiness or designated fixup dependencies',
                               'Complete native source still needs an exact protocol/early-return audit'])
    out = ROOT / 'results/moe-cluster-x-ring-model-20261029'
    out.mkdir(exist_ok=True)
    data = (json.dumps(report, indent=2) + '\n').encode()
    target = out / 'audit.json'
    assert not target.exists() or target.read_bytes() == data, 'Frozen evidence collision'
    target.write_bytes(data)
    print(json.dumps(dict(positive_cases=len(rows), rejected_mutants={k: v['reason'] for k, v in rejected.items()}, GPU_jobs=0)))


if __name__ == '__main__':
    main()
