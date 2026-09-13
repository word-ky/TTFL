import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from src.context.matched_channel import emission_pools,replica_seed,bootstrap_observation,exact_utilities,mismatch_residual,selected_half_count
from src.context.confusion_prevalence import estimate


class MatchedChannelTest(unittest.TestCase):
    def test_target_pool_exclusion(self):
        for target in range(3):
            def loader(j):
                self.assertNotEqual(j,target)
                return np.eye(10),np.arange(10)
            pools,used=emission_pools(target,range(3),loader)
            self.assertEqual(used,[j for j in range(3) if j!=target]);self.assertTrue(all(len(p)==2 for p in pools))
    def test_pool_means(self):
        channel=.8*np.eye(10)+.02*np.ones((10,10))
        pools,_=emission_pools(0,range(4),lambda j:(channel.T,np.arange(10)))
        np.testing.assert_allclose(np.stack([p.mean(0) for p in pools],axis=1),channel,atol=1e-15)
    def test_seeded_bootstrap(self):
        rng=np.random.default_rng(12);pools=[rng.dirichlet(np.ones(10),size=30) for _ in range(10)];counts=[2]*10
        seed=replica_seed(1,'A','clean',20,0)
        np.testing.assert_array_equal(bootstrap_observation(pools,counts,20,seed)[0],bootstrap_observation(pools,counts,20,seed)[0])
        self.assertNotEqual(seed,replica_seed(1,'A','clean',20,1))
    def test_composition_fixed(self):
        pools=[np.eye(10)[[k]] for k in range(10)];counts=np.array([8,6,4,2,0,0,0,0,0,0])
        for K in (20,40,80,160):
            q,n=bootstrap_observation(pools,counts,K,3)
            np.testing.assert_array_equal(n,counts*(K//20));np.testing.assert_array_equal(q,counts/20)
    def test_noise_free_full_rank(self):
        C=.7*np.eye(10)+.03*np.ones((10,10));pi=np.array([.4,.3,.2,.1,0,0,0,0,0,0])
        recovered,_=estimate(C,C@pi);np.testing.assert_allclose(recovered,pi,rtol=0,atol=1e-14)
    def test_average_and_simplex(self):
        pools=[np.tile(np.arange(1,11)/55,(3,1)) for _ in range(10)]
        q,_=bootstrap_observation(pools,[2]*10,40,4);np.testing.assert_allclose(q,np.arange(1,11)/55,atol=1e-15)
        pi,_=estimate(np.eye(10),q);self.assertTrue(np.all(pi>=0));self.assertAlmostEqual(pi.sum(),1.,places=14)
    def test_exact_template_and_count_lookup(self):
        template=[[Fraction(0),Fraction(k,10),Fraction(k,10),Fraction(0),Fraction(0)] for k in range(10)]
        u,arg=exact_utilities([Fraction(1,10)]*10,template);self.assertEqual(arg,[1,2])
        counts=np.arange(50).reshape(5,10);np.testing.assert_array_equal(selected_half_count(counts,arg[0]),counts[1])
    def test_mismatch_percentile(self):
        out=mismatch_residual([.9,.1],np.array([.5,.5]),np.array([[.5,.5],[.6,.4],[.7,.3]]))
        self.assertTrue(out['above_p95']);self.assertEqual(out['percentile'],1.)
        out=mismatch_residual([.5,.5],np.array([.5,.5]),np.array([[.5,.5],[.6,.4],[.7,.3]]))
        self.assertFalse(out['above_p95']);self.assertEqual(out['percentile'],1/3)
    def test_no_model_or_artifact_mutation(self):
        import src.context.matched_channel as module
        source=Path(module.__file__).read_text()
        self.assertNotIn('torch',source);self.assertNotIn('write_',source)


if __name__=='__main__':unittest.main()
