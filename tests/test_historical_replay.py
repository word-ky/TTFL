import copy,unittest
from src.context.historical_replay import compare_historical_episode_rows,compare_historical_table


class HistoricalReplayTest(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(client='1',bank='A',target='clean',estimator='pi_zero_soft',L1='1.2',JS='0.2809442888638962'),dict(client='2',bank='B',target='gaussian_blur',estimator='pi_zero_hard',L1='1.3',JS='0.1')]
        self.headers=list(self.rows[0])
    def compare(self,new):return compare_historical_episode_rows(new,self.rows,self.headers,self.headers)
    def test_identical_rows(self):
        r=self.compare(copy.deepcopy(self.rows));self.assertEqual(r['nonzero_delta_count'],0)
    def test_roundoff_JS_only(self):
        new=copy.deepcopy(self.rows);new[0]['JS']='0.28094428886389666'
        r=self.compare(new);self.assertEqual(r['nonzero_delta_count'],1);self.assertEqual(r['max_absolute_delta'],4.440892098500626e-16)
    def test_large_JS_fails(self):
        new=copy.deepcopy(self.rows);new[0]['JS']=str(float(new[0]['JS'])+2e-12)
        with self.assertRaises(AssertionError):self.compare(new)
    def test_non_JS_numeric_string_fails(self):
        for value in ('1.2000000000000000','1.2000000000000002'):
            new=copy.deepcopy(self.rows);new[0]['L1']=value
            with self.assertRaises(AssertionError):self.compare(new)
    def test_order_and_identifiers_fail(self):
        with self.assertRaises(AssertionError):self.compare(self.rows[::-1])
        for field in ('client','bank','target','estimator'):
            new=copy.deepcopy(self.rows);new[0][field]+='changed'
            with self.assertRaises(AssertionError):self.compare(new)
    def test_nonfinite_JS_fails(self):
        for value in ('nan','inf','-inf'):
            new=copy.deepcopy(self.rows);new[0]['JS']=value
            with self.assertRaises(AssertionError):self.compare(new)
    def test_aggregate_tables_exact(self):
        for name in ('mixture_quality.csv','scientific_gates.csv'):
            self.assertTrue(compare_historical_table(name,self.rows,self.rows,self.headers,self.headers)['exact'])
            new=copy.deepcopy(self.rows);new[0]['JS']='0.28094428886389666'
            with self.assertRaises(AssertionError):compare_historical_table(name,new,self.rows,self.headers,self.headers)
    def test_schema_and_row_count_exact(self):
        with self.assertRaises(AssertionError):self.compare(self.rows[:1])
        with self.assertRaises(AssertionError):compare_historical_episode_rows(self.rows,self.rows,self.headers[::-1],self.headers)


if __name__=='__main__':unittest.main()
