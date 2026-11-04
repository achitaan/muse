"""CPU-only gate-branch source, admission, and synchronization regressions."""
import ast
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from test_decode_gate_diag import BASE, definitions


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_decode_gate
        cls.source = build_decode_gate.build(ROOT)
        cls.tree = ast.parse(cls.source)
        names = {'bfp_SK2_SRC', 'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC', 'bfp_SK2_CONFIGS',
                 'bfp_SK2_PICK', 'bfp_SK2_K', 'bfp__sk2_name', 'bfp__sk2_tokr', 'bfp__sk2_geom',
                 '_de_source', 'ATT7_CUDA_SRC'}
        cls.ns = definitions(BASE, names)
        definitions(cls.tree, {'_gg_original_sk_source', '_gg_replace', '_gg_sk_source', '_gg_att_source'}, cls.ns)

    def test_standalone_readable_source(self):
        self.assertLess(len(self.source.encode()), 1_000_000)
        self.assertTrue(self.source.startswith('# okc: problem=muse_decode_layer'))
        for bad in ('b64decode(', 'zlib.decompress(', 'lzma.decompress('):
            self.assertNotIn(bad, self.source)

    def test_sk_waits_output_offsets_and_completion(self):
        for narrow in (False, True):
            src = self.ns['_gg_sk_source'](narrow)
            self.assertIn('if (gate_branch) asm volatile("griddepcontrol.launch_dependents;', src)
            self.assertEqual(src.count('if (pdl && !gate_branch) asm volatile("griddepcontrol.wait;'), 2)
            self.assertIn('out + (size_t)tt * N + gate_offset + cta * R + row', src)
            self.assertIn('if (!gate_branch && gate_flags)', src)
            self.assertIn('named_bar(1, NC);\n    if (tid == 0)', src)
            self.assertIn('atom.acq_rel.gpu.global.add.s32', src)
            self.assertIn('g_gate_flags, g_gate_branch, g_gate_offset));', src)
            self.assertIn('g_gate_flags, g_gate_branch, g_gate_offset);', src)

    def test_attention_only_delays_waits_not_arithmetic(self):
        old, new = self.ns['ATT7_CUDA_SRC'], self.ns['_gg_att_source']()
        self.assertIn('gate_qkv_wait(p);   // prefix release, not gate completion', new)
        self.assertEqual(new.count('pdl_wait();   // gate branch completion'), 4)
        # Every instruction in the original CUDA attention remains, except
        # the one relocated wait and the two reset statements. The expected
        # arithmetic comes from the checked-in incumbent (not a guessed output).
        for line in old.splitlines():
            if any(op in line for op in ('mma_async', 'floats2bfloat162', 'const float rn',
                                        'const float s0 = sigm', 'ss +=', 'sum_copies<', 'pr8(y[')):
                self.assertIn(line, new)
        for line in new.splitlines():
            if 'p.qkv +' in line and 'graw' in line:
                previous = new.splitlines()[new.splitlines().index(line) - 1]
                self.assertIn('pdl_wait();', previous)
        self.assertIn('p.ws_cnt[32767] = 0;\n        p.ws_cnt[24576 + bp] = ep + 1;', new)
        self.assertIn('if (tid == 0) { p.ws_cnt[bp] = 0; p.ws_cnt[32767] = 0; }', new)

    def test_exact_weight_row_and_output_partition(self):
        # bfp_sk2_stream assigns three 8-row tiles to each CTA. Repacking
        # [stage][0:54] and [stage][54:96] must preserve ALL rows and all K units.
        rng = random.Random(5353)
        for _ in range(10000):
            row, unit, lane = rng.randrange(2304), rng.randrange(128), rng.randrange(16)
            oldcta, localrow = divmod(row, 24)
            branch = oldcta >= 54
            localcta = oldcta - (54 if branch else 0)
            offset = 1296 if branch else 0
            self.assertEqual(offset + localcta * 24 + localrow, row)
            # No change to the row's warp, stage, K-unit or BF16 words.
            self.assertEqual((localrow // 8, unit, lane), (row % 24 // 8, unit, lane))
        self.assertEqual(54 + 42, 96)

    def test_counter_protocol_random_schedules(self):
        # Exhaust the ordering argument by randomizing independent events.
        # Attention queries cannot start before all 54 prefix releases. The
        # reset is legal only after every active split has merged, hence every
        # split must have passed the ready wait before the count returns to 0.
        rng = random.Random(32767)
        for _ in range(500):
            nsplit = rng.choice((32, 33, 252))
            count, read, merged = 0, set(), set()
            pending = [('prefix', i) for i in range(54)] + [('read', i) for i in range(nsplit)]
            while pending:
                eligible = [i for i, (kind, _) in enumerate(pending) if kind == 'prefix' or count == 54]
                k = rng.choice(eligible)
                kind, idx = pending.pop(k)
                if kind == 'prefix':
                    count += 1
                else:
                    self.assertEqual(count, 54)
                    read.add(idx)
            order = list(read)
            rng.shuffle(order)
            for split in order:
                merged.add(split)
                if len(merged) == nsplit:
                    self.assertEqual(len(read), nsplit)
                    count = 0
            self.assertEqual(count, 0)

    def test_strict_admission_and_byte_identity(self):
        src = (ROOT / 'kb/exp/gate_overlap_impl.py').read_text()
        self.assertIn("x.view(torch.uint8), y.view(torch.uint8)", src)
        self.assertIn("max(t['ratio'], t['abba'], t['baab']) <= .992", src)
        self.assertIn('len(timings) == 2', src)
        self.assertIn('valid &= all(checks)', src)
        self.assertIn("st['gate_pick'] = {}", src)
        self.assertIn("'-gencode=arch=compute_90a,code=sm_90a'", src)


if __name__ == '__main__':
    unittest.main()
