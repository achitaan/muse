"""CPU source/opportunity screen for Attention-to-O dependency overlap.

Does not import the submission, compile an extension, or launch a GPU job.
"""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
BASE = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'


def main():
    raw = (ROOT / 'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    tree = ast.parse(raw)
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)

    def constant(name):
        nodes = assignments[name]
        assert len(nodes) == 1
        node = nodes[0]
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id == '_zsrc'
            assert len(node.args) == 1 and not node.keywords
            return zlib.decompress(base64.b64decode(ast.literal_eval(node.args[0]))).decode()
        return ast.literal_eval(node)

    attention = constant('_native_attn_CUDA_SRC')
    calls = [i for i, line in enumerate(attention.splitlines(), 1)
             if line.strip() == 'pdl_launch_dependents();']
    assert len(calls) == 2
    assert 'asm volatile("griddepcontrol.wait;" ::: "memory");' in attention
    families = []
    for variable, routes, tokens, prefix in [
        ('tt_v10_sm2_SMALLM2_CUDA_TEMPLATE', 'V12_SM2_O1', 1, 'smallm2'),
        ('V18S_CUDA_TEMPLATE', 'V18_SMALL', 8, 'v18s'),
    ]:
        cuda = constant(variable)
        inst, geometry = constant(routes)['o', tokens]
        fmt, mt, rt, stages, red, diag, ntmax, cm, n, k, wr, wk, ku, splits, opt = inst
        assert fmt == 1 and diag == 0 and red == 0 and splits == 1
        assert opt & 2 and stages == ku == 2 and n == 8192 and k == 1024
        marker = 'for (int s = 0; s < ST; ++s) load(s, s);'
        assert cuda.count(marker) == 1
        assert 'if constexpr (!NOX && !RO && !(OPT & 2))' in cuda
        assert f'uint4 u = {prefix}_ldx(xp), v = {prefix}_ldx(xp + 8);' in cuda
        assert 'ld.global.nc.v4.u32' in cuda
        anchor = cuda.index(marker)
        direct = cuda.index('} else if constexpr (OPT & 2)', anchor)
        assert anchor < direct
        # The first ST units span the entire K domain for the selected geometry.
        assert stages * wk * 64 == k
        families.append(dict(tokens=tokens, source_variable=variable,
                             CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                             instance=inst, geometry=geometry,
                             initial_weight_only_units=stages,
                             full_K_prefetched=True,
                             packed_code_bytes=n * k * 3 // 2,
                             escapes_and_metadata_excluded=True,
                             activation_load_policy='ld.global.nc; review dependency/cache semantics before implementation',
                             candidate_implemented=False))
    loads_raw = (ROOT / 'results/retained-load-opportunities-20261029/loads.json').read_bytes()
    decode = json.loads(loads_raw)['studies']['decode']
    assert decode['source_sha256'] == BASE
    scenarios = []
    for affected in ((1, 8), (1, 8, 16), (1, 8, 16, 32)):
        for saving_us in (0.5, 1.0, 2.0):
            ratios = []
            for load in decode['loads'].values():
                cases = load['cases']
                assert sum(c['weight'] for c in cases) == 62
                saved_ns = sum(c['weight'] * saving_us * 1000
                               for c in cases if c['tokens'] in affected)
                old = load['retained_load_ns']
                assert old > saved_ns
                ratios.append(old / (old - saved_ns))
            scenarios.append(dict(affected_tokens=affected,
                                  assumed_net_saved_us_per_case=saving_us,
                                  hypothetical_track_ratio=math.prod(ratios) ** (1 / 6)))
    report = dict(baseline_sha256=BASE,
                  load_packet_sha256=hashlib.sha256(loads_raw).hexdigest(),
                  attention_CUDA_sha256=hashlib.sha256(attention.encode()).hexdigest(),
                  existing_attention_early_trigger_lines=calls,
                  selected_small_O=families, dilution_scenarios=scenarios,
                  scenarios_are_predictions=False,
                  additional_notes=[
                      'M16 uses m16trace next_early; M32 uses native BF16 olayer, not the old generic M32 table.',
                      'A launch attribute alone is unsafe: dependent activation reads require grid dependency synchronization.',
                      'The existing trigger does not itself publish final attention outputs.',
                      'Keep original arithmetic, BF16 materializations, packed weights, fallbacks and graph replay lifetimes.',
                      'Require independent-progress proof and graph-capture admission; concurrency must never be required for progress.',
                      'Compiler input-load dominance and actual H200 resources remain unverified.',
                      'Net savings must include added waits, launch changes and contention with attention finalization.',
                  ],
                  GPU_jobs=0, candidate_created=False,
                  decision='Continue bounded source/compiler feasibility analysis; no GPU ticket yet.')
    out = ROOT / 'results/decoder-projection-pdl-audit-20261029'
    out.mkdir(exist_ok=True)
    payload = (json.dumps(report, indent=2) + '\n').encode()
    target = out / 'audit.json'
    assert not target.exists() or target.read_bytes() == payload, 'Frozen audit collision'
    target.write_bytes(payload)
    print(json.dumps(dict(baseline_sha256=BASE, small_O_paths=len(families),
                          hypothetical_scenarios=len(scenarios), GPU_jobs=0,
                          candidate_created=False)))


if __name__ == '__main__':
    main()
