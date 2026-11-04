"""CPU regressions for compiler flags, complete diagnostics and prepare timing.

No CUDA build is simulated as a success: load_inline is intercepted to check
its actual arguments and exception behavior. Host/device frontend stubs are
a separate, explicitly limited check documented in results/MOE_RESTART.md.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).with_name('moe_restart.py')
spec = importlib.util.spec_from_file_location('restart_build', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BuildTests(unittest.TestCase):
    def test_both_compilers_use_cxx20(self):
        sentinel = object()
        with patch.object(mod, 'load_inline', return_value=sentinel) as load, contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(mod._restart_extension(), sentinel)
        flags = load.call_args.kwargs
        self.assertEqual(flags['extra_cflags'], ['-std=c++20'])
        self.assertEqual([f for f in flags['extra_cuda_cflags'] if f.startswith('-std=')], ['-std=c++20'])
        self.assertIn('-gencode=arch=compute_90a,code=sm_90a', flags['extra_cuda_cflags'])
        self.assertIn('--ftemplate-backtrace-limit=0', flags['extra_cuda_cflags'])
        self.assertFalse(flags['verbose'])  # retain combined ninja output in failures

    def test_build_exception_is_not_replaced_or_truncated(self):
        exc = RuntimeError('compiler output\n' * 2000 + 'FINAL_PTXAS_ERROR')
        with patch.object(mod, 'load_inline', side_effect=exc), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                mod._restart_extension()
        self.assertIs(caught.exception, exc)

    def test_first_errors_and_exact_tail_survive_log_truncation(self):
        first = 'cuda.cu(42): error: first NVCC diagnostic'
        text = (first + '\n' + 'template context ' * 4000 + '\n' +
                'ptxas fatal : last assembly diagnostic\n' + 'tail context ' * 180)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mod._restart_report_failure(RuntimeError(text))
        logged = out.getvalue()
        self.assertIn(text, logged)  # full output, before any summary slicing
        tail = logged[-16000:]
        self.assertIn('RSTREAM_ERROR_FIRST\n' + first, tail)
        self.assertIn('ptxas fatal : last assembly diagnostic', tail)
        self.assertTrue(logged.endswith(text[-1500:] + '\n'))

    def test_noncompiler_exception_remains_visible(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mod._restart_report_failure(ValueError('shape mismatch'))
        self.assertIn('RSTREAM_ERROR_FIRST\nshape mismatch', out.getvalue())
        self.assertIn('RSTREAM_DISABLED ValueError', out.getvalue())

    def test_compact_summary_does_not_drop_later_errors(self):
        errors = [f'cuda.cu({i}): error: diagnostic {i}' for i in range(20)]
        errors[1] = 'ptxas cuda.ptx, line 30; error   : Arguments mismatch for instruction'
        errors[2] = 'cuda.cu(123): error #20: identifier is undefined'
        text = ('\n' + 'template context ' * 2000 + '\n').join(errors)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mod._restart_report_failure(RuntimeError(text))
        self.assertIn('RSTREAM_ERROR_MORE', out.getvalue())
        tail = out.getvalue()[-16000:]
        for error in errors:
            self.assertIn(error, tail)

    def test_deadline_with_headroom_and_no_deadline(self):
        with patch('time.monotonic', return_value=100.0):
            mod._restart_budget_check(None)
            mod._restart_budget_check(103.0)
            with self.assertRaises(TimeoutError):
                mod._restart_budget_check(101.9)

    def test_architecture_fallback_reports_total_prepare(self):
        out = io.StringIO()
        weights = {'w13': SimpleNamespace(device='cpu')}
        state = {}
        with (patch.object(mod.torch.cuda, 'get_device_capability', return_value=(8, 0)),
              patch.object(mod, '_restart_incumbent_prepare', return_value=state),
              patch('time.monotonic', side_effect=[10.0, 12.0, 13.0]),
              contextlib.redirect_stdout(out)):
            self.assertIs(mod.prepare(weights), state)
        self.assertEqual(state['restart_cases'], frozenset())
        self.assertIn('RSTREAM_PREPARE seconds=3.0', out.getvalue())
        self.assertIn('within_limit=1', out.getvalue())

    def test_build_failure_reports_shutdown_time_and_overrun(self):
        clock = [100.0]
        future = Mock()
        future.result.side_effect = RuntimeError('nvcc fatal : synthetic build failure')
        pool = Mock()
        pool.submit.return_value = future
        pool.shutdown.side_effect = lambda **kwargs: clock.__setitem__(0, 410.0)
        weights = {'w13': SimpleNamespace(device='cpu'), 'w13_sf': [], 'w2_sf': []}
        out = io.StringIO()
        with (patch.object(mod.torch.cuda, 'get_device_capability', return_value=(9, 0)),
              patch.object(mod.torch.cuda, 'synchronize'),
              patch.object(mod, '_scales_prmt_exact', return_value=True),
              patch.object(mod, '_restart_incumbent_prepare', return_value={}),
              patch('concurrent.futures.ThreadPoolExecutor', return_value=pool),
              patch('time.monotonic', side_effect=lambda: clock[0]),
              contextlib.redirect_stdout(out)):
            state = mod.prepare(weights)
        self.assertEqual(state['restart_cases'], frozenset())
        pool.shutdown.assert_called_once_with(wait=True)
        self.assertIn('RSTREAM_ERROR_FIRST\nnvcc fatal : synthetic build failure', out.getvalue())
        self.assertIn('RSTREAM_PREPARE seconds=310.0', out.getvalue())
        self.assertIn('within_limit=0', out.getvalue())

    def test_no_packing_or_admission_after_expired_setup(self):
        clock = [100.0]
        future = Mock()
        future.result.side_effect = lambda: clock.__setitem__(0, 374.0)
        pool = Mock()
        pool.submit.return_value = future
        weights = {'w13': SimpleNamespace(device='cpu'), 'w13_sf': [], 'w2_sf': []}
        with (patch.object(mod.torch.cuda, 'get_device_capability', return_value=(9, 0)),
              patch.object(mod.torch.cuda, 'synchronize'),
              patch.object(mod, '_scales_prmt_exact', return_value=True),
              patch.object(mod, '_restart_incumbent_prepare', return_value={}),
              patch.object(mod, '_restart_weights') as pack,
              patch.object(mod, '_restart_select') as select,
              patch('concurrent.futures.ThreadPoolExecutor', return_value=pool),
              patch('time.monotonic', side_effect=lambda: clock[0]),
              contextlib.redirect_stdout(io.StringIO())):
            state = mod.prepare(weights)
        self.assertEqual(state['restart_cases'], frozenset())
        pack.assert_not_called()
        select.assert_not_called()

    def test_resource_rejection_is_per_case_not_prepare_abort(self):
        state = {'grid': 132}
        out = io.StringIO()
        def graph(data, state, new):
            if new:
                raise RuntimeError('RSTREAM_RESOURCE_REJECT local_bytes=1928')
            return Mock(), None
        with (patch.object(mod, '_restart_data', side_effect=lambda m,*args,**kwargs: {'m': m}),
              patch.object(mod, '_restart_graph', side_effect=graph) as run,
              patch.object(mod.torch, 'empty', return_value=Mock()),
              patch.object(mod.torch.cuda, 'Event', return_value=Mock()),
              patch.object(mod.torch.cuda, 'synchronize'),
              contextlib.redirect_stdout(out)):
            mod._restart_select({'w13': SimpleNamespace(device='cpu')}, state)
        self.assertEqual(run.call_count, 12)
        self.assertEqual(out.getvalue().count('RSTREAM_CASE_DISABLED'), 6)
        self.assertEqual(state['restart_cases'], frozenset())

    def test_default_incumbent_prepare_skips_optional_selector(self):
        self.assertIs(mod._restart_incumbent_prepare, mod._moe_sel_original_prepare)

    def test_production_clock_code_is_compile_time_discarded(self):
        body = mod.RESTART_CUDA.split('void stamp(')[1].split('template<int ROWS')[0]
        self.assertIn('if constexpr (PROFILE)', body)
        self.assertEqual(mod.RESTART_CUDA.count('clock64()'), 3)
        for name in ('sample_start', 'sample_end'):
            body = mod.RESTART_CUDA.split(name+'(')[1].split('\n}')[0]
            self.assertIn('if constexpr (PROFILE)', body)
        self.assertIn('attrs.localSizeBytes == 0', mod.RESTART_CUDA)
        self.assertIn('const __grid_constant__ Params p', mod.RESTART_CUDA)
        for r,c in ((64,8),(128,16),(128,32),(128,64),(64,128),(64,256)):
            self.assertIn(f'rs_launch<{r},{c}>', mod.RESTART_CUDA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
