import unittest
from operator_features import summarize, extract


class OperatorTests(unittest.TestCase):
    def test_observed_ratios_and_slowdown_retained(self):
        # Manufactured arithmetic: 8/10=.8 then 12/10=1.2; neither is serving evidence.
        rows=[(1,dict(event='timing',tokens=8,pattern='hot',cache='cold',offset=32,
                      block=0,medians_us={'stock_a':10,'stock_b':10,'s1':8})),
              (2,dict(event='timing',tokens=8,pattern='hot',cache='cold',offset=32,
                      block=1,medians_us={'stock_a':10,'stock_b':11,'s1':12}))]
        r=summarize(rows,'s1')[0]
        self.assertEqual(r['stage_new_old_ratio'],[.8,1.2])
        self.assertEqual(r['saved_us'],[-2,2])
        self.assertIsNone(r['serving_concurrency'])
        with self.assertRaises(ValueError): summarize(rows+rows,'s1')

    def test_saved_source_bound_features(self):
        result=extract()
        self.assertEqual(result['serving_labels'],0)
        self.assertEqual(result['component_sessions'],1)
        for family in result['families']:
            self.assertTrue(family['observations'])
            for row in family['observations']:
                self.assertEqual(sorted(row['blocks']),[0,1,2])
                self.assertIsNone(row['serving_concurrency'])


if __name__=='__main__': unittest.main()
