import unittest
from fractions import Fraction
import numpy as np
from src.context.noise_free_ties import noise_free_sanity
from src.context.exact_template_lookup import compile_template,exact_lookup
from src.context.matched_channel import exact_utilities


class NoiseFreeTieTest(unittest.TestCase):
    def test_perturbation_within_exact_set(self):
        p=np.array([1.,0.]);q=np.array([1.-1e-15,1e-15])
        self.assertTrue(noise_free_sanity(q,p,1,[Fraction(1),Fraction(1),Fraction(0)],0)['tied'])
    def test_outside_argmax_fails(self):
        with self.assertRaises(AssertionError):noise_free_sanity([1.,0.],np.array([1.,0.]),2,[Fraction(1),Fraction(1),Fraction(0)],0)
    def test_nonzero_exact_regret_fails(self):
        with self.assertRaises(AssertionError):noise_free_sanity([1.,0.],np.array([1.,0.]),1,[Fraction(1),Fraction(1)-Fraction(1,10**30)],0)
    def test_singleton_exact_identity(self):
        self.assertFalse(noise_free_sanity([1.,0.],np.array([1.,0.]),0,[Fraction(1),Fraction(0)],0)['tied'])
        with self.assertRaises(AssertionError):noise_free_sanity([1.,0.],np.array([1.,0.]),1,[Fraction(1),Fraction(0)],0)
    def test_no_mutation(self):
        p=np.array([1.-1e-15,1e-15]);before=p.copy();values=[Fraction(1),Fraction(1)]
        noise_free_sanity(p,np.array([1.,0.]),1,values,0)
        np.testing.assert_array_equal(p,before);self.assertEqual(values,[Fraction(1),Fraction(1)])
    def test_integer_lookup_exact_fraction_equivalence(self):
        rng=np.random.default_rng(4)
        template=[[Fraction(int(rng.integers(-100,100)),int(rng.integers(1,1000))) for s in range(5)] for k in range(10)]
        coefficients,den=compile_template(template);counts=np.array([8,6,4,2,0,0,0,0,0,0]);true,_=exact_utilities([Fraction(int(n),20) for n in counts],template)
        for _ in range(20):
            pi=rng.dirichlet(np.ones(10));values,arg=exact_utilities(pi,template);selected,du=exact_lookup(pi,coefficients,den,counts)
            self.assertEqual(selected,arg[0]);self.assertEqual(du,float(max(abs(x-y) for x,y in zip(values,true))))
    def test_integer_lookup_exact_tie(self):
        template=[[Fraction(k,7),Fraction(k,7),Fraction(-1),Fraction(-2),Fraction(-3)] for k in range(10)]
        co,d=compile_template(template);self.assertEqual(exact_lookup([.1]*10,co,d,[2]*10)[0],0)


if __name__=='__main__':unittest.main()
