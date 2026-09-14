import hashlib,unittest
import numpy as np
from src.context.observation_audit import class_seed,draw_source_ids,decompose,null_comparison,single_class_repair


class ObservationAuditTest(unittest.TestCase):
    def test_seed_exact(self):
        expected=int(hashlib.sha256(b'T019|3|B|clean|7|2').hexdigest(),16)
        self.assertEqual(class_seed(3,'B','clean',7,2),expected)
    def test_count_conditioning_and_target_exclusion(self):
        labels=np.tile(np.arange(10),8);counts=np.array([11,9]+[0]*8)
        ids=draw_source_ids(1,'A','clean',0,counts,labels)
        self.assertFalse(np.any(ids//20==1));np.testing.assert_array_equal(np.bincount(labels[ids],minlength=10),counts)
        np.testing.assert_array_equal(ids,draw_source_ids(1,'A','clean',0,counts,labels))
    def test_source_id_pairing(self):
        labels=np.tile(np.arange(10),8);ids=draw_source_ids(0,'B','gaussian_blur',5,[2]*10,labels)
        clean=np.arange(80);shift=clean+1000
        np.testing.assert_array_equal(shift[ids]-clean[ids],np.full(20,1000))
    def test_decomposition_and_absent_classes(self):
        p=np.array([[.7,.3]+[0.]*8,[.2,.8]+[0.]*8]);matrix=np.eye(10)
        counts,mu,d,w,q,r=decompose(p,np.array([0,1]),matrix)
        np.testing.assert_allclose(w.sum(0),r,atol=1e-12,rtol=0)
        self.assertTrue(np.isnan(mu[2:]).all());self.assertTrue(np.isnan(d[2:]).all())
    def test_all_repaired_identity(self):
        p=np.full((20,10),.1);labels=np.repeat([0,1],10);C=np.eye(10)
        n,mu,d,w,q,r=decompose(p,labels,C)
        np.testing.assert_allclose(q-w.sum(0),C@(n/20),atol=1e-12,rtol=0)
        np.testing.assert_array_equal(single_class_repair(q,w,0),q-w[0])
    def test_percentile_strict(self):
        r=null_comparison(1.,np.ones(128))
        self.assertEqual(r['percentile'],1.);self.assertFalse(r['above_p95'])
    def test_no_mutation(self):
        p=np.full((20,10),.1);before=p.copy();decompose(p,np.repeat([0,1],10),np.eye(10))
        np.testing.assert_array_equal(p,before)
