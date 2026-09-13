import inspect
import unittest
import torch
from src.adaptation.paired_affine_oracle_neutral import fit_channel_residual,paired_clean_oracle_neutral
from src.models.pfllib import ContextFedAvgCNN
from src.eval.common import state_hash


class NeutralOracleTest(unittest.TestCase):
    def test_exact_identity_three_variances(self):
        torch.manual_seed(7)
        for x in (torch.randn(4,3,8,8),torch.ones(4,3,8,8),torch.ones(4,3,8,8)+1e-6*torch.randn(4,3,8,8)):
            gamma,beta,raw,_=fit_channel_residual(x,x)
            self.assertTrue(torch.equal(gamma,torch.zeros_like(gamma)))
            self.assertTrue(torch.equal(beta,torch.zeros_like(beta)))
            self.assertTrue(torch.equal(x*(1+gamma[None,:,None,None])+beta[None,:,None,None],x))

    def test_known_residual_and_cap(self):
        x=torch.tensor([-2.,-1.,1.,2.]).reshape(1,1,2,2);z=1.2*x+.3
        g,b,raw,_=fit_channel_residual(x,z)
        self.assertTrue(torch.isfinite(g).all() and torch.isfinite(b).all())
        self.assertGreater(g.item(),0);self.assertLess(g.item(),.2)
        self.assertAlmostEqual(b.item(),.3,places=6)
        self.assertLess(((x*(1+g)+b-z)**2).mean().item(),((x-z)**2).mean().item()/100)
        self.assertLess(raw.abs().max().item(),8)
        g,_,_,_=fit_channel_residual(x,-20*x+2);self.assertEqual((1+g).item(),-8)

    def test_model_identity_and_sequential_contract(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=ContextFedAvgCNN(3,10,1600).eval();x=torch.randn(4,3,32,32)
        h=state_hash(model);state,_=paired_clean_oracle_neutral(model,x,x)
        self.assertEqual(sum(v.numel() for v in state),192)
        self.assertTrue(all(torch.equal(v,torch.zeros_like(v)) for v in state))
        self.assertTrue(torch.equal(model(x),model(x,state)))
        source=.4*x-.2;state,_=paired_clean_oracle_neutral(model,x,source)
        z=x
        for i,block in enumerate((model.conv1,model.conv2)):
            source,z=block(source),block(z);g,b,_,_=fit_channel_residual(source,z)
            self.assertTrue(torch.equal(g,state[2*i]));self.assertTrue(torch.equal(b,state[2*i+1]))
            source=source*(1+g[None,:,None,None])+b[None,:,None,None]
        self.assertEqual(state_hash(model),h)
        self.assertFalse(any(v.requires_grad for v in state))
        self.assertEqual(list(inspect.signature(paired_clean_oracle_neutral).parameters),['model','clean_support','source_support'])


if __name__=='__main__':unittest.main()
