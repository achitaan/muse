"""CPU scheduler/ownership audit. Finite evidence, not a GPU race proof."""
import bisect
import json
import random


def audit(counts, phase, grid):
    costs = ([112, 135, 140, 158, 173, 185, 200, 207],
             [100, 104, 115, 123, 132, 144, 156, 174])[phase]
    stages = 512 >> phase
    prefix = [0]
    for n in counts:
        prefix.append(prefix[-1] + costs[(n + 7) // 8 - 1] * stages)
    if not counts:
        return 0, 0

    def stage_at(x):
        job = bisect.bisect_right(prefix, x) - 1
        return job * stages + (x - prefix[job]) // costs[(counts[job] + 7) // 8 - 1]

    ranges = [(stage_at(prefix[-1] * c // grid),
               stage_at(prefix[-1] * (c + 1) // grid) if c + 1 < grid else len(counts) * stages)
              for c in range(grid)]
    partial = {}
    owners = {}
    for c, (a, b) in enumerate(ranges):
        for u in range(a // 16, (b + 15) // 16):
            lo, hi = max(a, u * 16), min(b, (u + 1) * 16)
            if lo >= hi or hi - lo == 16:
                continue
            endpoint = 0 if a // 16 == u else 1
            assert (c, endpoint) not in owners  # exactly one writer per endpoint
            owners[c, endpoint] = u
            partial.setdefault(u, []).append((c, lo, hi, endpoint))
    for u, pieces in partial.items():
        # Exact unit coverage, no duplicates, <=16 nonempty contributors.
        assert [s for _, a, b, _ in pieces for s in range(a, b)] == list(range(u * 16, (u + 1) * 16))
        assert len(pieces) <= 16
        assert [c for c, _, _, _ in pieces] == sorted(c for c, _, _, _ in pieces)
        for s in (u * 16, u * 16 + 15):
            job = s // stages
            xe = prefix[job] + ((s % stages) + 1) * costs[(counts[job] + 7) // 8 - 1]
            inverse = min(grid - 1, (xe - 1) * grid // prefix[-1])
            while inverse + 1 < grid and prefix[-1] * (inverse + 1) // grid < xe:
                inverse += 1
            while inverse > 0 and prefix[-1] * inverse // grid >= xe:
                inverse -= 1
            assert ranges[inverse][0] <= s < ranges[inverse][1]
    return len(partial), max((len(p) for p in partial.values()), default=0)


def main():
    rng = random.Random(91028)
    sets = [[], [1], [8], [64], [64] * 32, [1] * 32, [1, 64] * 16]
    sets += [[rng.randint(1, 64) for _ in range(rng.randint(1, 56))] for _ in range(100)]
    cases, maximum = 0, 0
    for counts in sets:
        for phase in (0, 1):
            for grid in (1, 2, 16, 64, 132, 159):
                _, contributors = audit(counts, phase, grid)
                maximum = max(maximum, contributors)
                cases += 1
    old = 175163680
    extra = 2 * 160 * 4 * (8 * 512) * 4
    total = old + extra + 59 * 768
    assert total < 256 * 1024**2
    print(json.dumps(dict(cases=cases, max_contributors_observed=maximum,
                         max_contributors_proved=16, extra_bytes=extra,
                         one_layer_bytes=old + extra, sixty_layer_bytes=total)))


if __name__ == "__main__":
    main()
