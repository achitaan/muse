"""CPU/static gate for FULL-CB2; no CUDA compilation/execution is implied.

Oracles: incumbent producer/address equations and independently enumerated
logical KV rows, plus the exact 8f53cd76 scored numbers. Numerical device code
is unchanged; tests prove coverage/resource/host-wrapper/admission invariants,
not that Hopper's newly instantiated NPW2 kernel runs correctly or faster.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'board_best/attention_kb1h2_scored_2.6536.py'
CANDIDATE = ROOT / 'candidates/attention_full_cb2.py'


def definitions(path):
    constants, functions = {}, {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    return constants, functions


def load(names, env, path=CANDIDATE):
    _, functions = definitions(path)
    for name in names:
        exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), str(path), 'exec'), env)
    return env


def copies(npw):
    """Actual source producer mapping, returning (ptid, row, 16-byte chunk)."""
    npt, nwp = 128 * npw, 4 * npw
    vp = (128 + 4 * nwp - 1) // (4 * nwp)
    return [(tid, i * 4 * nwp + (tid >> 5) * 4 + ((tid & 31) >> 3), tid & 7)
            for tid in range(npt) for i in range(vp)]


def stage(npw, k, v, pages, last):
    buf = np.empty((2, 128 * 128), np.uint8)
    for _, row, chunk in copies(npw):
        page = pages[min(row, last)]
        for kind, cache in enumerate((k, v)):
            sw = row & 7 if kind == 0 else (row >> 4) & 7
            off = row * 128 + ((chunk ^ sw) << 4)
            buf[kind, off:off + 16] = cache[page, chunk * 16:chunk * 16 + 16]
    return buf


class TensorMetadata:
    def __init__(self, shape, dtype, device='cuda:0', pointer=256):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim, self.is_cuda = len(shape), device.startswith('cuda')
        self.pointer, self.contiguous = pointer, True
    def data_ptr(self): return self.pointer
    def is_contiguous(self): return self.contiguous
    def numel(self): return math.prod(self.shape)
    def stride(self, dim): return math.prod(self.shape[dim + 1:])
    def view(self, *args): return self


def metadata():
    data = dict(config=dict(batch=16, q_len=8, window=0, page_size=1, num_q_heads=16,
                            num_kv_heads=2, head_dim=64, rope_theta=0., kv_scale=1.,
                            qk_scale_factor=5.4730064863875, sm_scale=.125))
    for name, shape, dtype in (
            ('qkv', (16, 8, 2304), torch.bfloat16), ('page_table', (16, 86008), torch.int32),
            ('seq_lens', (16,), torch.int32), ('cache_loc', (128,), torch.int64),
            ('k_cache', (1376128, 1, 2, 64), torch.float8_e4m3fn),
            ('v_cache', (1376128, 1, 2, 64), torch.float8_e4m3fn),
            ('k_scale', (), torch.float32), ('v_scale', (), torch.float32)):
        data[name] = TensorMetadata(shape, dtype)
    return data


def gate_case(timings=((.97, .97), (.97, .97)), candidate_error=(.03, .05), fail=None, cache_fail=False):
    base_out = torch.ones((16, 8, 1024), dtype=torch.bfloat16)
    candidate_out = base_out.clone()
    data = dict(qkv=base_out, k_cache=torch.zeros(8, dtype=torch.uint8), v_cache=torch.zeros(8, dtype=torch.uint8))
    class Graph:
        def replay(self): pass
    calls, comparisons = [], iter(timings)
    def score(data, output, pack, cache_bytes, tag, deadline):
        rel, row = (.03, .05) if tag.endswith(' base') else candidate_error
        bad = fail is not None and tag.endswith(' ' + fail)
        return (not bad or cache_fail), rel, row, not (bad and cache_fail)
    def abba(*args):
        calls.append('abba')
        ratio, paired = next(comparisons)
        return 100., 100. * ratio, ratio, paired
    env = dict(
        torch=SimpleNamespace(uint8=torch.uint8, equal=torch.equal,
                              cuda=SimpleNamespace(synchronize=lambda *_: None)),
        _fcb_capture=lambda data, candidate, deadline: (Graph(), candidate_out if candidate else base_out),
        _fcb_launch=lambda *_: candidate_out.clone(), _at_pack=lambda *_: None,
        _at_reset_cache=lambda *_: None, _at_score=score, _at_abba=abba)
    load(('_fcb_gate_cell',), env)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        keep, status = env['_fcb_gate_cell'](data, None, None, None)
    return keep, status, calls, stdout.getvalue()


class FullCB2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constants, cls.functions = definitions(CANDIDATE)
        cls.base_constants, cls.base_functions = definitions(BASE)

    def test_complete_incumbent_is_literal_unchanged_prefix(self):
        self.assertTrue(CANDIDATE.read_bytes().startswith(BASE.read_bytes()))

    def test_only_entry_instantiation_changes_and_resource_query_is_host_only(self):
        env = load(('_fcb_source',), dict(self.constants))
        old = self.constants['ZH_WST_CUDA']
        got = env['_fcb_source'](old)
        expected = old.replace('launch_ct14b<4, 2, 4, 1>((int)G', 'launch_ct14b<4, 2, 2, 1>((int)G')
        self.assertEqual(got, expected + self.constants['FCB_RESOURCE_CUDA'])
        self.assertNotIn('__global__', self.constants['FCB_RESOURCE_CUDA'])
        for bad in (old.replace('launch_ct14b<4, 2, 4, 1>((int)G', ''), old + old):
            with self.assertRaisesRegex(ValueError, 'anchor'):
                env['_fcb_source'](bad)

    def test_compiler_math_and_precision_flags_identical_to_incumbent(self):
        def flag_list(node):
            for item in ast.walk(node):
                if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'flags' for t in item.targets):
                    return ast.literal_eval(item.value)
        self.assertEqual(flag_list(self.functions['_fcb_build']), flag_list(self.base_functions['_zh_build_wst']))

    def test_producer_tile_coverage_both_allocations_heads_and_chunks(self):
        for npw in (2, 4):
            seen = np.zeros((128, 8), np.int32)
            for tid, row, chunk in copies(npw):
                self.assertLess(tid, 128 * npw)
                seen[row, chunk] += 1
            # Independent oracle: each 128B two-head row is eight 16B chunks.
            np.testing.assert_array_equal(seen, 1)
            self.assertEqual(len(copies(npw)) * 2 * 16, 32768)
            self.assertEqual(len(copies(npw)) * 2 // 32, 64)  # warp cp.async issue count

    def test_staged_bytes_match_independent_logical_oracle_and_old_producers(self):
        rng = np.random.default_rng(0xFCB2)
        k = rng.integers(0, 256, (1000, 128), dtype=np.uint8)
        v = rng.integers(0, 256, (1000, 128), dtype=np.uint8)
        pages = rng.choice(1000, size=128, replace=False)
        for last in (0, 7, 63, 127):
            a, b = stage(2, k, v, pages, last), stage(4, k, v, pages, last)
            np.testing.assert_array_equal(a, b)
            for row in range(128):
                for chunk in range(8):
                    for kind, cache in enumerate((k, v)):
                        sw = (row & 7) if kind == 0 else (row >> 4) & 7
                        off = row * 128 + ((chunk ^ sw) << 4)
                        # Oracle: unchanged FP8 cache bytes at the logical page,
                        # including source's clamped last row in partial tiles.
                        np.testing.assert_array_equal(a[kind, off:off+16], cache[pages[min(row, last)], chunk*16:chunk*16+16])

    def test_new_kv_rows_unchanged_and_each_written_once(self):
        def live_assignments(npw):
            live = []
            for ptid in range(128 * npw):
                nu, d0 = ptid >> 3, (ptid & 7) * 8
                if nu < 32:
                    live.append((nu >> 2, (nu >> 1) & 1, nu & 1, d0))
            return live
        expected = {(tok, kind, head, dim) for tok in range(8) for kind in range(2)
                    for head in range(2) for dim in range(0, 64, 8)}
        for npw in (2, 4):
            live = live_assignments(npw)
            self.assertEqual(set(live), expected)
            self.assertEqual(len(live), len(expected))

    def test_register_shared_memory_and_barrier_budgets(self):
        # Source branches, not measured register use: both budgets =61440/65536.
        self.assertEqual(512 * 40 + 256 * 160, 256 * 40 + 256 * 200)
        self.assertEqual(256 * 40 + 256 * 200, 61440)
        smem = 4 * 32768 + 4 * 9216 + 64 + 64 + ((4 * 17 + 15) & ~15) + 36864
        self.assertEqual(smem, 205008)
        self.assertLess(smem, 227 * 1024)  # one H200 opt-in block fits
        self.assertGreater(2 * smem, 228 * 1024)  # shared memory still caps one CTA/SM
        for npw in (2, 4):
            npt = 128 * npw
            self.assertEqual(npt % 128, 0)  # full register-reconfiguration WGs
            self.assertEqual(len(set(t for t, _, _ in copies(npw))), npt)
            # One cp.async-completion arrival per producer, all 256 consumers
            # release the slot. Both named-barrier counts remain warp multiples.
            self.assertEqual(npt % 32, 0)
            self.assertEqual(256 % 32, 0)

    def test_resource_preflight_rejects_insufficient_register_pool_before_launch(self):
        env = load(('_fcb_resource_ok',), {})
        good2 = [2,512,40,200,120,0,205008,1,1024]
        good4 = [4,768,40,160,80,0,205008,1,1024]
        self.assertTrue(env['_fcb_resource_ok'](good2))
        self.assertTrue(env['_fcb_resource_ok'](good4))
        for idx, value in ((4,112), (7,0), (8,256), (6,0), (3,104), (0,1)):
            bad = good2.copy(); bad[idx] = value
            self.assertFalse(env['_fcb_resource_ok'](bad), (idx,value))

    def test_wst_streamk_partials_and_ticket_protocol_unchanged(self):
        rng = np.random.default_rng(132)
        cases = [rng.integers(74000, 86001, 16).tolist() for _ in range(30)]
        cases += [[0]*16, [80000]+[0]*15]
        for seq in cases:
            prefix = [0]
            for n in seq:
                prefix.append(prefix[-1] + (n + 8 + 127)//128)
            ge, bw = min(132, prefix[-1]), 3
            vv = [p + bw*b for b, p in enumerate(prefix)]
            w = vv[-1]
            def owner(v): return ((v+1)*ge-1)//w
            totals = [0]*16
            coverage = [set() for _ in seq]
            scratch = set()
            for c in range(ge):
                v0, v1 = c*w//ge, (c+1)*w//ge
                for b in range(16):
                    begin, end = max(v0, vv[b]+bw), min(v1, vv[b+1])
                    if end <= begin: continue
                    cf, cl = owner(vv[b]+bw), owner(vv[b+1]-1)
                    totals[b] += 256 - (cl-cf+1) + 1 if c == cl else 1
                    lo, hi = (1, cl-cf-1) if cl-cf+1 >= 3 else (0, cl-cf)
                    j = c-cf
                    if lo <= j <= hi:
                        for r in range(j-lo, 64, hi-lo+1):
                            self.assertNotIn(r, coverage[b])
                            coverage[b].add(r)
                    for head in range(2):
                        for r in range(64):
                            offset = ((c+b)*2+head)*64+r
                            self.assertLess(offset, (132+16)*2*64)
                            self.assertNotIn(offset, scratch)
                            scratch.add(offset)
            # Exact expected protocol: every request's tickets +=256 per call;
            # interior folders finalize each of its 64 rows exactly once.
            self.assertEqual(totals, [256]*16)
            for rows in coverage: self.assertEqual(rows, set(range(64)))

    def test_host_wrapper_arguments_match_v16f_incumbent(self):
        data, calls = metadata(), []
        out = TensorMetadata((16, 8, 1024), torch.bfloat16)
        acc, m, l, tickets = (object() for _ in range(4))
        recorder = SimpleNamespace(wst_attn_bal=lambda *args: calls.append(args))
        env = dict(torch=SimpleNamespace(empty=lambda *a, **kw: out, bfloat16=torch.bfloat16, uint8=torch.uint8),
                   _FCB_MODULES={'mod': recorder}, _ZH={'wst': recorder}, LOG2E=1.4426950408889634,
                   _fcb_workspace=lambda _: (acc,m,l,tickets), _zh_dev_ws=lambda _: (acc,m,l),
                   _zh_tickets=lambda *_: tickets, _num_sms=lambda _: 132)
        load(('_fcb_launch', '_zh_wst'), env)
        env['_zh_wst'](data)
        env['_fcb_launch'](data)
        self.assertEqual(calls[0], calls[1])

    def test_only_target_exact_layout_is_eligible(self):
        env = load(('_fcb_layout_ok',), {'torch': torch, '_FCB_DEVICE': 'cuda:0'})
        self.assertTrue(env['_fcb_layout_ok'](metadata()))
        for name, value in (('batch',32), ('q_len',1), ('window',2048), ('page_size',64),
                            ('num_kv_heads',4), ('rope_theta',500000.), ('kv_scale',1/64),
                            ('sm_scale',.25)):
            data = metadata(); data['config'][name] = value
            self.assertFalse(env['_fcb_layout_ok'](data), name)
        for name in ('qkv','page_table','seq_lens','cache_loc','k_cache','v_cache','k_scale','v_scale'):
            data = metadata(); data[name].pointer += 1
            self.assertFalse(env['_fcb_layout_ok'](data), name)
            data = metadata(); data[name].device = 'cuda:1'
            self.assertFalse(env['_fcb_layout_ok'](data), name)
        data = metadata(); data['page_table'].shape = (16,86009)
        self.assertFalse(env['_fcb_layout_ok'](data))

    def test_runtime_dispatch_falls_back_outside_target_or_without_confirmation(self):
        env = dict(_FCB_KEPT=False, _fcb_layout_ok=lambda d: d['target'],
                   _fcb_launch=lambda _: 'experimental', _FCB_BASE=lambda _: 'incumbent')
        load(('kernel_fn',), env)
        self.assertEqual(env['kernel_fn']({'target':True}), 'incumbent')
        env['_FCB_KEPT'] = True
        self.assertEqual(env['kernel_fn']({'target':False}), 'incumbent')
        self.assertEqual(env['kernel_fn']({'target':True}), 'experimental')

    def test_prepare_skips_if_incumbent_is_not_wst4(self):
        cuda = SimpleNamespace(is_available=lambda:True, is_current_stream_capturing=lambda:False,
                               current_device=lambda:0, get_device_capability=lambda _:(9,0))
        for modules, selected, threads in (
                ({}, {}, []), ({'wst':object()}, {(16,8,0):('onehead','1h',{})}, []),
                ({'wst':object()}, {}, [SimpleNamespace(is_alive=lambda:True)])):
            env = dict(torch=SimpleNamespace(cuda=cuda, device=lambda *_:'cuda:0'),
                       _FCB_DEVICE='stale', _FCB_KEPT=True, _FCB_STATUS='not-started',
                       _ZH=modules, _AT_SELECTED=selected, _ZH_THREADS=threads)
            load(('_fcb_prepare',), env)
            env['_fcb_prepare']()
            self.assertFalse(env['_FCB_KEPT'])
            self.assertIsNone(env['_FCB_DEVICE'])
            self.assertIn('not ready WST4', env['_FCB_STATUS'])

    def test_gate_requires_all_four_speed_comparisons_and_no_worse(self):
        for search, confirm, expected in (
                ((.97,.97),(.97,.97),True), ((.981,.97),(.97,.97),False),
                ((.97,.981),(.97,.97),False), ((.97,.97),(.981,.97),False),
                ((.97,.97),(.97,.981),False), ((float('nan'),.97),(.97,.97),False)):
            keep, _, _, _ = gate_case(timings=(search,confirm))
            self.assertEqual(keep, expected, (search,confirm))
        keep, _, calls, _ = gate_case(candidate_error=(.030000000000000002,.05))
        self.assertFalse(keep)  # no epsilon was added to make a pass
        self.assertEqual(len(calls), 2)  # valid numerical failure can be timed diagnostically

    def test_gate_hard_failures_never_adopt_or_time_before_validation(self):
        for tag in ('base','eager-probe','captured','repeat','preconfirm','confirm'):
            for cache in (False, True):
                keep, _, calls, _ = gate_case(fail=tag, cache_fail=cache)
                self.assertFalse(keep, (tag,cache))
                if tag in ('base','eager-probe','captured'):
                    self.assertEqual(calls, [], (tag,cache))

    def test_scored_full_bandwidth_table(self):
        data = json.loads((ROOT/'results/attention_redesign_baseline.json').read_text())
        expected = {(16,1):3.7886932065761187, (32,1):3.954279851058259,
                    (16,8):3.337620137543563, (32,8):3.5683580779847577}
        for row in data['records']:
            if row['layer'] != 'full' or row['batch'] == 1: continue
            # Analytic unique bytes at nominal ctx=80k; not a physical-traffic measurement.
            bw = row['batch']*row['ctx']*256/(row['median_ns']/1000)/1e6
            self.assertAlmostEqual(bw, expected[row['batch'],row['q_len']], places=12)

    def test_resource_query_cpp_syntax(self):
        compiler = shutil.which('clang++')
        if compiler is None: self.skipTest('clang++ unavailable')
        shim = r'''
#include <vector>
#include <cstdint>
#include <stdexcept>
template<int,int,int,int,int,int> void wst_ct14b() {}
constexpr int STG2=32768, VIMG9=9216;
using cudaError_t=int;
constexpr int cudaSuccess=0, cudaFuncAttributeMaxDynamicSharedMemorySize=1;
const char* cudaGetErrorString(int) { return "stub"; }
struct cudaFuncAttributes { int numRegs=120; size_t localSizeBytes=0; int maxThreadsPerBlock=1024; };
template<class T> int cudaFuncSetAttribute(T,int,int) { return 0; }
template<class T> int cudaFuncGetAttributes(cudaFuncAttributes*,T) { return 0; }
template<class T> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* a,T,int,int) { *a=1; return 0; }
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("check"); } while(0)
'''
        with tempfile.TemporaryDirectory(prefix='.full-cb2-test-', dir=ROOT) as tmp:
            source = Path(tmp)/'resource.cpp'
            source.write_text(shim+self.constants['FCB_RESOURCE_CUDA'])
            subprocess.run([compiler,'-std=c++20','-fsyntax-only',str(source)],check=True,capture_output=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
