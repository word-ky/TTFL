import inspect
import unittest
import torch
from src.adaptation.moments import (estimate_global_reference_moments,
    write_affine_from_support_moments,moments,EPS)
from src.models.pfllib import ContextFedAvgCNN
from src.eval.common import state_hash


class MomentsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model=ContextFedAvgCNN(3,10,1600).eval()
        self.x=torch.randn(5,3,32,32)

    def test_reference_stream_matches_direct(self):
        ref=estimate_global_reference_moments(self.model,self.x.split(2))
        self.assertEqual(ref['samples'],5)
        h=self.x
        for layer,block in enumerate((self.model.conv1,self.model.conv2)):
            h=block(h)
            mu,sigma=moments(h)
            self.assertTrue(torch.allclose(ref['layers'][layer]['mu'],mu,atol=1e-7))
            self.assertTrue(torch.allclose(ref['layers'][layer]['sigma'],sigma,atol=1e-7))
            self.assertEqual(ref['layers'][layer]['activation_count'],h.shape[0]*h.shape[2]*h.shape[3])

    def test_sequential_formula_frozen_no_labels_or_optimizer(self):
        self.assertEqual(list(inspect.signature(write_affine_from_support_moments).parameters),
                         ['model','images','reference'])
        ref=estimate_global_reference_moments(self.model,[self.x])
        before=state_hash(self.model)
        support=self.x*.4-.3
        state,diag=write_affine_from_support_moments(self.model,support,ref)
        h=support
        for layer,block in enumerate((self.model.conv1,self.model.conv2)):
            h=block(h)
            mu,sigma=moments(h)
            scale=(ref['layers'][layer]['sigma']/(sigma+EPS)).clamp(.25,4)
            self.assertTrue(torch.equal(state[2*layer],(scale-1).float()))
            self.assertTrue(torch.equal(state[2*layer+1],(ref['layers'][layer]['mu']-scale*mu).float()))
            h=h*(1+state[2*layer][None,:,None,None])+state[2*layer+1][None,:,None,None]
            self.assertLessEqual(diag[layer]['moment_mismatch_after'],diag[layer]['moment_mismatch_before']+1e-5)
        self.assertEqual(sum(s.numel() for s in state),192)
        self.assertFalse(any(s.requires_grad for s in state))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(before,state_hash(self.model))
        repeat,_=write_affine_from_support_moments(self.model,support,ref)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(state,repeat)))

    def test_clamp_and_constant_variance(self):
        mu,sigma=moments(torch.ones(2,3,4,4))
        self.assertTrue(torch.equal(mu,torch.ones_like(mu)))
        self.assertTrue(torch.allclose(sigma,torch.full_like(sigma,EPS**.5)))
        ref=estimate_global_reference_moments(self.model,[self.x])
        for layer in ref['layers']:
            layer['sigma']*=100
        state,diag=write_affine_from_support_moments(self.model,self.x,ref)
        self.assertEqual(diag[0]['clamp_high_fraction'],1.)
        self.assertTrue(torch.equal(state[0],torch.full_like(state[0],3.)))


if __name__=='__main__':
    unittest.main()
