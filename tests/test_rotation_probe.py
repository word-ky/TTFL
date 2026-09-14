import hashlib,unittest
import numpy as np
from src.context.rotation_probe import folds,prepare,evaluate,independent_score,rotation_labels,scramble_labels

class RotationProbeTest(unittest.TestCase):
    def test_ids_and_scramble(self):
        ids=list(range(100,120));a,b=folds(7,ids)
        expected=sorted(range(20),key=lambda k:hashlib.sha256(f'T023|fold|7|{ids[k]}'.encode()).hexdigest())
        self.assertEqual(list(a)+list(b),expected);self.assertEqual(len(set(a)&set(b)),0)
        y=scramble_labels(7,ids,3)
        np.testing.assert_array_equal(y.sum(1),np.ones((20,4)))
        np.testing.assert_array_equal(y,scramble_labels(7,ids,3))
    def test_minimum_norm_and_heldout_labels(self):
        rng=np.random.default_rng(4);h=rng.normal(size=(20,4,512));split=folds(2,list(range(20)));y=rotation_labels()
        score,err,rank=evaluate(prepare(h,split),y);ref,re,rr=independent_score(h,split,y)
        self.assertAlmostEqual(score,ref,places=12);self.assertEqual(rank,rr)
        expected=np.mean(re);self.assertEqual(ref,expected)
    def test_rotation_signal_beats_scramble(self):
        h=np.tile(np.eye(4),(20,1)).reshape(20,4,4);split=folds(0,list(range(20)));p=prepare(h,split)
        correct,_,_=evaluate(p,rotation_labels());wrong,_,_=evaluate(p,scramble_labels(0,list(range(20)),0))
        self.assertLess(correct,1e-20);self.assertGreater(wrong,.1)
