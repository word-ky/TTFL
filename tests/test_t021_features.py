import unittest
import torch
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from scripts.t021_features import extract


class FeaturesTest(unittest.TestCase):
    def test_exact_existing_forward_with_affine(self):
        torch.manual_seed(21);model=ContextFedAvgCNN(3,10,1600).eval();x=torch.randn(3,3,32,32)
        state=[torch.randn_like(p)*.1 for p in new_state(model)]
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            h,l=extract(model,x,state);expected=model(x,state)
        self.assertEqual(tuple(h.shape),(3,512));self.assertEqual(tuple(l.shape),(3,10));self.assertTrue(torch.equal(l,expected))
        self.assertFalse(h.requires_grad)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
