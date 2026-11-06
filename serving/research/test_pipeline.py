"""Manufactured contracts only: these records are never serving training labels."""
import copy
import datetime as dt
from pathlib import Path
import tempfile
import unittest

from engine import (predict, append_record, records, ingest, calibration,
                    promotion_review, trace_prediction, validation_plan)


def candidate():
    return dict(id='manufactured', patch_sha256='a'*64,
                evaluation_context=dict(sglang_revision='b'*40, gpu='test-device',
                                        environment={'runtime':'test'}, workload='manufactured',
                                        baseline_patch_sha256='c'*64),
                by_concurrency={'1':dict(critical_path_fraction=[.2,.2], stage_new_old_ratio=[.5,.5],
                                        added_time_fraction=[0,0], overlap_basis='nonoverlapped_critical_path')})


def observation(root):
    pred = predict(candidate())
    key = append_record(root, 'prediction', pred)
    timestamp = dt.datetime.fromisoformat(records(root)[key]['recorded_at']) + dt.timedelta(microseconds=1)
    return dict(prediction_id=key, candidate_sha256=pred['candidate_sha256'], patch_sha256='a'*64,
                **candidate()['evaluation_context'], measurement_id='manufactured-1',
                measurement_started_at=timestamp.isoformat(), lineage='family-a', session='session-a',
                fidelity='full_serving', concurrency=1, metric='latency', unit='us',
                stock=100., candidate=90., path_executed=True, source='manufactured-test-only',
                gates={k:True for k in ('rules','output_agreement','graph_safety','memory','ttft','reproducible_gain')})


class PipelineTests(unittest.TestCase):
    def test_prospective_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            o = observation(root)
            ingest(root, o)
            report = calibration(root)
            self.assertAlmostEqual(report['prospective_comparisons'][0]['midpoint_error'], 0)
            self.assertEqual(report['grouped_validation']['status'], 'insufficient_independent_groups')
            self.assertTrue(promotion_review(o)['eligible_for_human_review'])

    def test_context_units_and_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            o = observation(root)
            for field, bad_value in [('patch_sha256','x'), ('sglang_revision','x'), ('unit','ms'),
                                     ('workload','other'), ('measurement_started_at','2000-01-01T00:00:00+00:00'),
                                     ('measurement_started_at','2099-01-01T00:00:00+00:00'),
                                     ('concurrency',True), ('candidate',float('nan')), ('path_executed','yes')]:
                bad = dict(o, **{field:bad_value})
                with self.subTest(field=field), self.assertRaises(ValueError): ingest(root,bad)
            ingest(root,o)
            # Same measured job cannot become another sample via a different prediction.
            another = append_record(root,'prediction',predict(candidate()))
            repeated = dict(o,prediction_id=another,measurement_started_at=dt.datetime.now(dt.timezone.utc).isoformat())
            with self.assertRaises(ValueError): ingest(root,repeated)

    def test_failures_preserved_but_excluded_from_performance_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            o=observation(root)
            o['gates']['output_agreement']=False
            ingest(root,o)
            self.assertEqual(len(calibration(root)['excluded']),1)
            self.assertFalse(calibration(root)['prospective_comparisons'])
            self.assertFalse(promotion_review(o)['eligible_for_human_review'])

    def test_cheap_fidelity_never_becomes_serving_label(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            o=observation(root)
            o['fidelity']='integrated_component'
            ingest(root,o)
            self.assertFalse(calibration(root)['prospective_comparisons'])
            self.assertFalse(promotion_review(o)['eligible_for_human_review'])

    def test_false_gain_gate_cannot_promote_slowdown(self):
        with tempfile.TemporaryDirectory() as d:
            o=observation(Path(d))
            o['candidate']=110
            self.assertFalse(promotion_review(o)['eligible_for_human_review'])

    def test_transitive_session_lineage_grouping(self):
        rows=[dict(lineage='a',session='1'),dict(lineage='b',session='1'),
              dict(lineage='b',session='2'),dict(lineage='c',session='2'),
              dict(lineage='d',session='3')]
        plan=validation_plan(rows)
        self.assertEqual(plan['groups'],[[0,1,2,3],[4]])
        for fold in plan['folds']:
            for field in ('lineage','session'):
                self.assertFalse({rows[i][field] for i in fold['train']} & {rows[i][field] for i in fold['test']})

    def test_trace_path_switch_and_cli_prediction(self):
        # 10us branch and 20us branch join 5us consumer. Candidate branch [2,12]:
        # critical path changes; total candidate bounds are [15,17], not [7,17].
        trace=dict(resource_edges_complete=True, source='manufactured test graph',
                   nodes=[dict(id='a',duration_us=10),dict(id='b',duration_us=20),
                          dict(id='join',duration_us=5,dependencies=['a','b'])],
                   changed_duration_us={'b':[2,12]})
        self.assertEqual(trace_prediction(trace)['candidate_us'],[15,17])
        p=predict({'id':'trace-test','by_concurrency':{'1':{'trace':trace}}})
        self.assertEqual(p['concurrency']['1']['missing'],[])
        self.assertAlmostEqual(p['concurrency']['1']['latency_reduction'][0],.32)
        self.assertIsNone(p['concurrency']['1']['throughput_change'])
        trace['resource_edges_complete']=False
        with self.assertRaises(ValueError): trace_prediction(trace)

    def test_no_silent_concurrency_or_memory_coercion(self):
        for value in ({'id':'x','by_concurrency':{'3':{}}},
                      {'id':'x','additional_workspace_bytes':True},
                      {'id':'x','workspace_cap_bytes':-1}):
            with self.subTest(value=value), self.assertRaises(ValueError): predict(value)


if __name__=='__main__': unittest.main()
