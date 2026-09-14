import hashlib,unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from src.context.bootstrap_regret import bootstrap_positions,expected_regret_objects
from src.context.bootstrap_convergence import BLOCKS,aggregate_blocks,convergence_gate


class BootstrapConvergenceTest(unittest.TestCase):
    def test_prefix_namespace(self):
        for r in range(256):
            seed=int(hashlib.sha256(f'T020|0|A|clean|{r}'.encode()).hexdigest(),16)
            np.testing.assert_array_equal(bootstrap_positions(0,'A','clean',r),np.random.Generator(np.random.PCG64(seed)).integers(0,20,20))
    def test_arbitrary_R_equivalence(self):
        rows=[[r%7,2,3,0,1] for r in range(256)]
        blocks={'H1':(0,128),'H2':(128,256),'ALL':(0,256)};o,d,states,masks=aggregate_blocks(rows,100,0,blocks)
        _,_,old=expected_regret_objects(rows,100,0)
        self.assertEqual(o['ALL']['state'],old['BER256']);self.assertEqual(o['ALL']['means_exact'],old['mean_utility256_exact'])
        self.assertEqual(d['BER_bootstrap_expected_regret_exact'],old['BER_bootstrap_expected_regret_exact'])
    def test_blocks_and_agreements(self):
        rows=[[3,0,0,0,0]]*2048+[[0,4,0,0,0]]*2048;o,_,_,_=aggregate_blocks(rows,10,0,BLOCKS)
        self.assertEqual([o[k]['state'] for k in ('Q0','Q1','Q2','Q3','H1','H2','ALL')],[0,0,1,1,0,1,1])
        result=convergence_gate({'H1':[0]*8000,'H2':[0]*7920+[1]*80,'ALL':[0]*7960+[1]*40},['A']*8000,['clean']*8000)
        self.assertTrue(result['passed']);self.assertEqual(result['global_agreement']['H1_H2'],.99)
    def test_label_blind_R4096(self):
        p=np.random.default_rng(5).random((20,5));p/=p.sum(1,keepdims=True)
        positions=np.array([bootstrap_positions(2,'B','clean',r) for r in range(4096)]);q=p[positions].mean(1)
        outputs=[]
        for stored_labels in (np.zeros(20),np.arange(20)):
            # Labels are unrelated stored metadata, never parameters of the bootstrap/aggregation.
            rows=[[int(round(v*1e9)) for v in row] for row in q]
            outputs.append(aggregate_blocks(rows,10**9,0,BLOCKS)[0])
        self.assertEqual(outputs[0],outputs[1])
    def test_ties_at_R4096(self):
        o,d,_,m=aggregate_blocks([[2,2,0,0,0]]*4096,10,1,BLOCKS)
        self.assertEqual(o['ALL']['state'],0);self.assertEqual(Fraction(d['BER_bootstrap_expected_regret_exact']),0);self.assertTrue(np.all(m==3))
    def test_no_unsealing_in_preparation(self):
        text=(Path(__file__).resolve().parents[1]/'scripts/eval_t020r.py').read_text()
        for forbidden in ('support_truth.json','half_integer_counts.npz','true_template_regrets.json','labels_in_frozen_support_order'):self.assertNotIn(forbidden,text)
