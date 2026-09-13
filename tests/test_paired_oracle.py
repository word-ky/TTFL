import inspect
import unittest
import torch
from src.adaptation.paired_affine_oracle import fit_channel_affine,paired_clean_oracle,EPS
from src.models.pfllib import ContextFedAvgCNN
from src.eval.common import state_hash


class PairedOracleTest(unittest.TestCase):
    def test_known_affine_and_cap(self):
        x=torch.tensor([-2.,-1.,1.,2.]).reshape(1,1,2,2)
        a,b,_=fit_channel_affine(x,3*x+2)
        self.assertAlmostEqual(a.item(),3.,places=4)
        self.assertAlmostEqual(b.item(),2.,places=5)
        a,b,_=fit_channel_affine(x,-20*x+2)
        self.assertEqual(a.item(),-8.)
        self.assertEqual(b.item(),2.)

    def test_identity_formula_and_constant(self):
        x=torch.tensor([-2.,-1.,1.,2.]).reshape(1,1,2,2)
        a,b,_=fit_channel_affine(x,x)
        self.assertAlmostEqual(a.item(),2.5/(2.5+EPS),places=7)
        self.assertEqual(b.item(),0.)
        x=torch.ones(2,3,4,4)
        a,b,_=fit_channel_affine(x,x)
        self.assertTrue(torch.equal(a[None,:,None,None]*x+b[None,:,None,None],x))

    def test_sequential_frozen_api(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=ContextFedAvgCNN(3,10,1600).eval()
        clean=torch.randn(4,3,32,32);source=.4*clean-.2
        before=state_hash(model)
        state,ds=paired_clean_oracle(model,clean,source)
        self.assertEqual(list(inspect.signature(paired_clean_oracle).parameters),
                         ['model','clean_support','source_support'])
        x,z=source,clean
        for layer,block in enumerate((model.conv1,model.conv2)):
            x,z=block(x),block(z)
            a,b,_=fit_channel_affine(x,z)
            self.assertTrue(torch.equal(state[layer*2],a-1))
            self.assertTrue(torch.equal(state[layer*2+1],b))
            x=x*(1+state[layer*2][None,:,None,None])+b[None,:,None,None]
            self.assertLess(ds[layer]['mse_after'],ds[layer]['mse_before'])
        self.assertEqual(sum(s.numel() for s in state),192)
        self.assertEqual(state_hash(model),before)
        self.assertFalse(any(s.requires_grad for s in state))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
