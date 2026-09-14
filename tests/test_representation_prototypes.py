import unittest
import numpy as np
from src.context.representation_prototypes import excluded_prototype,matched_mean


class RepresentationPrototypesTest(unittest.TestCase):
    def test_target_labels_and_features_excluded(self):
        rng=np.random.default_rng(21);x=rng.normal(size=(4,20,12));y=np.tile(np.arange(20)%10,(4,1))
        m,n,owners=excluded_prototype(x,y,2);x[2]=1e6;y[2]=9
        mm,nn,oo=excluded_prototype(x,y,2)
        np.testing.assert_array_equal(m,mm);np.testing.assert_array_equal(n,nn);self.assertNotIn(2,owners);self.assertEqual(owners,oo)
    def test_matched_exact_counts_and_seed(self):
        x=np.arange(4*20*12).reshape(4,20,12);y=np.tile(np.arange(20)%10,(4,1));counts=[4,0,3,1,2,1,2,1,3,3]
        m,ids=matched_mean(x,y,1,counts,'H','B','clean',127);mm,ii=matched_mean(x,y,1,counts,'H','B','clean',127)
        np.testing.assert_array_equal(m,mm);np.testing.assert_array_equal(ids,ii);self.assertTrue(np.all(ids//20!=1))
        np.testing.assert_array_equal(np.bincount(y.reshape(-1)[ids],minlength=10),counts)
