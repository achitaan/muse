import copy
import tempfile
import unittest
from pathlib import Path
from engine import predict, append_record, records, calibration, queue, critical_path


class ContractTests(unittest.TestCase):
    def test_overlap_and_join(self):
        # Two concurrent branches (10 and 20 us) join a 5 us consumer: 25, not 35.
        nodes=[dict(id='a',duration_us=10),dict(id='b',duration_us=20),
               dict(id='join',duration_us=5,dependencies=['a','b'])]
        self.assertEqual(critical_path(nodes),25)
        self.assertEqual(critical_path(nodes,{'a':1}),25)
        self.assertEqual(critical_path(nodes,{'b':2}),15)

    def test_cyclic_trace_rejected(self):
        with self.assertRaises(ValueError):
            critical_path([dict(id='a',duration_us=1,dependencies=['a'])])
    def test_missing_is_not_zero(self):
        result = predict({'id':'missing'})
        self.assertIsNone(result['concurrency']['1']['latency_reduction'])
        self.assertIsNone(result['probability_improving'])

    def test_amdahl_and_overhead(self):
        # Analytic example: 20% serial stage halved -> 10% total reduction.
        c = {'id':'example', 'by_concurrency':{'1':dict(critical_path_fraction=[.2,.2],
             stage_new_old_ratio=[.5,.5], added_time_fraction=[0,0],
             overlap_basis='nonoverlapped_critical_path')}}
        p = predict(c)['concurrency']['1']
        self.assertAlmostEqual(p['latency_reduction'][0], .1)
        self.assertIsNone(p['throughput_change'])
        c['by_concurrency']['1']['added_time_fraction'] = [.2,.2]
        self.assertAlmostEqual(predict(c)['concurrency']['1']['latency_reduction'][0], -.1)

    def test_reject_summed_overlap(self):
        c={'id':'x','by_concurrency':{'1':dict(critical_path_fraction=[0,1],stage_new_old_ratio=[.5,1],added_time_fraction=[0,0])}}
        with self.assertRaises(ValueError): predict(c)

    def test_immutable_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            key=append_record(root,'prediction',predict({'id':'x'}))
            self.assertIn(key,records(root))
            (root/(key+'.json')).write_text('{}')
            with self.assertRaises(ValueError): records(root)

    def test_no_training_from_empty_data(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(calibration(Path(d))['trained_model'])

    def test_failed_candidate_blocked(self):
        self.assertEqual(queue([{'id':'bad','validity':'rejected'}])[0]['status'],'blocked')


if __name__=='__main__': unittest.main()
