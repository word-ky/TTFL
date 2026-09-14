import hashlib,io,unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from src.context.bootstrap_regret import position_seed,bootstrap_positions,shared_integer_weights,exact_utilities_from_weights,canonical_and_mask,expected_regret_objects
from src.context.constrained_prevalence import ActiveSetCLS


class BootstrapRegretTest(unittest.TestCase):
    def test_seed_and_positions(self):
        seed=int(hashlib.sha256(b'T020|3|B|clean|7').hexdigest(),16)
        self.assertEqual(position_seed(3,'B','clean',7),seed)
        np.testing.assert_array_equal(bootstrap_positions(3,'B','clean',7),np.random.Generator(np.random.PCG64(seed)).integers(0,20,20))
    def test_label_blindness(self):
        p=np.tile(np.arange(1,11)/55,(20,1));outputs=[]
        for labels in (np.zeros(20),np.arange(20)%10):
            buffer=io.BytesIO();np.savez(buffer,probabilities=p,stored_true_labels=labels);buffer.seek(0)
            post=np.load(buffer)['probabilities'];positions=np.array([bootstrap_positions(0,'A','clean',r) for r in range(256)])
            q=post[positions].mean(1);pi=np.array([ActiveSetCLS(np.eye(10)).solve(x)[0] for x in q]);w,d=shared_integer_weights(pi)
            n=exact_utilities_from_weights(w,[list(range(10)),list(range(9,-1,-1)),[0]*10,[0]*10,[0]*10]);_,_,r=expected_regret_objects(n,d,0)
            outputs.append((positions,q,pi,r['BER256']))
        for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)
    def test_expected_regret_equivalence(self):
        n=[[i%7,2,3,0,1] for i in range(256)];_,_,r=expected_regret_objects(n,100,0)
        means=[sum(row[s] for row in n)/256 for s in range(5)];regrets=[sum(max(row)-row[s] for row in n)/256 for s in range(5)]
        self.assertEqual(r['BER256'],int(np.argmax(means)));self.assertEqual(r['BER256'],int(np.argmin(regrets)))
    def test_exact_ties(self):
        self.assertEqual(canonical_and_mask([3,3,1,0,0]),(0,3))
        _,_,r=expected_regret_objects([[2,2,0,0,0]]*256,10,1);self.assertEqual(r['BER256'],0);self.assertEqual(r['BER_bootstrap_expected_regret'],0.)
    def test_stability_detects_switch(self):
        states,masks,r=expected_regret_objects([[3,0,0,0,0]]*128+[[0,4,0,0,0]]*128,10,0)
        self.assertFalse(r['agreement']);self.assertEqual(r['BER128'],0);self.assertEqual(r['BER256'],1);self.assertEqual(r['max_vote_frequency_difference'],.5)
    def test_exact_integer_utility(self):
        pi=np.array([[.1,.9],[.8,.2]]);w,d=shared_integer_weights(pi);nums=exact_utilities_from_weights(w,[[3,5],[1,2]])
        for row,p in zip(nums,pi):self.assertEqual(Fraction(row[0],d),3*Fraction(float(p[0]))+5*Fraction(float(p[1])))
    def test_preparation_has_no_privileged_query_reader(self):
        source=(Path(__file__).resolve().parents[1]/'scripts/eval_t020.py').read_text()
        for forbidden in ('support_truth.json','half_integer_counts.npz','true_template_regrets.json','labels_in_frozen_support_order'):
            self.assertNotIn(forbidden,source)
    def test_source_dispatch_only_fixed_ids(self):
        # Each client/slot chooses a frozen context index; no detector is called.
        choices=np.arange(100*2*5*8).reshape(100,2,5,8);context=np.full((100,2,8),3)
        from src.context.bootstrap_regret import frozen_context_dispatch
        selected=frozen_context_dispatch(choices,context)
        np.testing.assert_array_equal(selected[:, :, 0, :],choices[:,:,3,:])
