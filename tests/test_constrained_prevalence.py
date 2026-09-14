import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from src.context.constrained_prevalence import cls_s,direct_kkt,face_reference
from src.context.confusion_prevalence import project_simplex
from src.context.matched_channel import exact_utilities


class ConstrainedPrevalenceTest(unittest.TestCase):
    def test_projection_unchanged(self):
        x=np.array([.8,.5,-.3]);before=x.copy();np.testing.assert_allclose(project_simplex(x),[.65,.35,0.],atol=1e-15)
        np.testing.assert_array_equal(x,before)
    def test_simplex(self):
        C=.8*np.eye(3)+.2/3;pi,r=cls_s(C,np.array([.8,.15,.05]))
        self.assertTrue(r['converged']);self.assertTrue(np.all(pi>=0));self.assertAlmostEqual(pi.sum(),1.,places=14)
    def test_noise_free(self):
        C=.7*np.eye(4)+.3/4;true=np.array([.6,.3,.1,0.]);pi,r=cls_s(C,C@true)
        np.testing.assert_allclose(pi,true,rtol=0,atol=1e-14);self.assertLessEqual(r['objective'],1e-16)
    def test_objective_dominance(self):
        C=np.array([[.7,.2,.1],[.2,.6,.2],[.1,.2,.7]])
        for q in ([.9,.05,.05],[.1,.8,.1],[.2,.3,.5]):
            pi,r=cls_s(C,q);self.assertTrue(r['converged']);self.assertLessEqual(r['objective'],r['initial_objective']+5e-13)
    def test_deterministic_and_inputs_unchanged(self):
        C=.8*np.eye(3)+.2/3;q=np.array([.8,.15,.05]);old=C.copy();before=q.copy()
        a,ra=cls_s(C,q);b,rb=cls_s(C,q);np.testing.assert_array_equal(a,b);self.assertEqual(ra,rb);np.testing.assert_array_equal(C,old);np.testing.assert_array_equal(q,before)
    def test_independent_face_solution(self):
        C=np.array([[.7,.2,.1],[.2,.6,.2],[.1,.2,.7]]);q=np.array([.95,.04,.01])
        p,r=cls_s(C,q);ref,obj=face_reference(C,q)
        self.assertLess(abs(obj-r['objective']),1e-10);np.testing.assert_allclose(p,ref,rtol=0,atol=1e-7);self.assertLess(direct_kkt(C,q,p),1e-10)
    def test_raw_prevalence_selection_no_override(self):
        template=[[Fraction(k,10),Fraction(9-k,10),Fraction(0),Fraction(0),Fraction(0)] for k in range(10)]
        pi,r=cls_s(np.eye(10),np.array([0.]*9+[1.]));values,arg=exact_utilities(pi,template);self.assertEqual(arg,[0]);np.testing.assert_array_equal(pi,[0.]*9+[1.])
    def test_exact_tie_and_no_model_calls(self):
        template=[[Fraction(1),Fraction(1),Fraction(0),Fraction(0),Fraction(0)] for _ in range(10)]
        pi,r=cls_s(np.eye(10),np.full(10,.1));self.assertEqual(exact_utilities(pi,template)[1],[0,1])
        import src.context.constrained_prevalence as module
        source=Path(module.__file__).read_text();self.assertNotIn('torch',source);self.assertNotIn('write_',source)


if __name__=='__main__':unittest.main()
