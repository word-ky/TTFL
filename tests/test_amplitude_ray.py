import inspect
import unittest
import torch
from src.context.amplitude_ray import ALPHAS,scale_state,choose_ray
from src.adaptation.affine import new_state
from src.models.pfllib import ContextFedAvgCNN
from src.eval.common import state_hash

class AmplitudeRayTest(unittest.TestCase):
    def test_scaling_endpoints_and_immutability(self):
        torch.manual_seed(15);model=ContextFedAvgCNN(3,10,1600).eval();x=torch.randn(3,3,32,32)
        state=[torch.randn_like(v)*.1 for v in new_state(model)];copies=[v.clone() for v in state];before=state_hash(model)
        self.assertEqual(list(inspect.signature(scale_state).parameters),['state','alpha'])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x),model(x,scale_state(state,0.))))
            self.assertTrue(torch.equal(model(x,state),model(x,scale_state(state,1.))))
            for alpha in ALPHAS:
                a=scale_state(state,alpha);b=scale_state(state,alpha)
                self.assertTrue(all(torch.equal(v,w) for v,w in zip(a,b)))
                self.assertTrue(all(torch.equal(v,alpha*w) for v,w in zip(a,state)))
                self.assertTrue(torch.isfinite(model(x,a)).all())
        self.assertEqual(before,state_hash(model));self.assertTrue(all(torch.equal(v,w) for v,w in zip(state,copies)))
    def test_fixed_selectors(self):
        scores={a:dict(J=1.,consistency_js=1.,flip_top1_agreement=.5) for a in ALPHAS}
        self.assertEqual(choose_ray(scores),dict(J_ray=0.,PC_ray_safe=0.))
        scores[.25]=dict(J=.2,consistency_js=.1,flip_top1_agreement=.4)
        scores[.5]=dict(J=.2,consistency_js=.2,flip_top1_agreement=.5)
        scores[.75]=dict(J=.2,consistency_js=.2,flip_top1_agreement=.5)
        self.assertEqual(choose_ray(scores),dict(J_ray=.25,PC_ray_safe=.5))

if __name__=='__main__':unittest.main()
