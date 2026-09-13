import inspect
import unittest
import numpy as np
from src.context.confusion_prevalence import (probabilities,project_simplex,
    channel_from_other_clients,normalized_channel,estimate,source_estimate)


class ConfusionPrevalenceTest(unittest.TestCase):
    def test_target_exclusion_and_exact_columns(self):
        for target in range(4):
            accessed=[]
            def loader(client):
                self.assertNotEqual(client,target);accessed.append(client)
                return np.arange(10),np.arange(10)
            h,s,d,used=channel_from_other_clients(target,range(4),loader)
            self.assertEqual(used,accessed);self.assertNotIn(target,used)
            np.testing.assert_array_equal(h.sum(0),d)
            np.testing.assert_array_equal(normalized_channel(h,d),np.eye(10))
        with self.assertRaises(AssertionError):
            channel_from_other_clients(0,[0,1],lambda _: (np.array([0]),np.array([0])))

    def test_simplex(self):
        p=np.arange(1,11,dtype=float)/55
        np.testing.assert_allclose(project_simplex(p),p,atol=1e-15,rtol=0)
        for x in [np.zeros(10),np.arange(-5,5),np.array([100.]+[-1.]*9)]:
            a=project_simplex(x);self.assertTrue(np.all(a>=0))
            self.assertAlmostEqual(a.sum(),1.,places=13)
            np.testing.assert_array_equal(a,project_simplex(x))

    def test_known_confusion_recovery(self):
        C=.8*np.eye(10)+.02*np.ones((10,10));pi=np.arange(1,11)/55
        p,z=estimate(C,C@pi)
        np.testing.assert_allclose(p,pi,rtol=0,atol=1e-14)

    def test_soft_channel_and_label_isolation(self):
        C=.8*np.eye(10)+.02*np.ones((10,10))
        labels={i:np.arange(10) for i in range(3)}
        def build():
            return channel_from_other_clients(1,range(3),lambda j:(C.T,labels[j]))
        first=build();labels[1]=np.zeros(10,dtype=int);second=build()
        np.testing.assert_array_equal(first[1],second[1])
        np.testing.assert_array_equal(first[2],second[2])
        np.testing.assert_allclose(normalized_channel(first[1],first[2]),C,atol=1e-15)

    def test_source_branch_no_hidden_context(self):
        channels={'clean':np.eye(10),'blur':.8*np.eye(10)+.02*np.ones((10,10))}
        q=np.arange(1,11)/55
        a=source_estimate(channels,'clean',q)[0]
        channels['blur']=np.zeros((10,10))
        np.testing.assert_array_equal(a,source_estimate(channels,'clean',q)[0])
        self.assertEqual(list(inspect.signature(source_estimate).parameters),['channels','source_context','q'])

    def test_raw_logits_baseline(self):
        z=np.tile(np.log(np.arange(1,11)/55),(20,1))
        np.testing.assert_allclose(probabilities(z).mean(0),np.arange(1,11)/55,atol=1e-15)
        np.testing.assert_array_equal(np.bincount(probabilities(z).argmax(1),minlength=10),[0]*9+[20])


if __name__=='__main__':unittest.main()
