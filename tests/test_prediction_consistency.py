import inspect
import unittest
import torch
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.eval.common import state_hash
from src.context.prediction_consistency import score_state,select_pc_safe,js_divergence,CONTEXTS

class PredictionConsistencyTest(unittest.TestCase):
    def test_zero_determinism_immutability(self):
        torch.manual_seed(12);model=ContextFedAvgCNN(3,10,1600).eval();x=torch.randn(4,3,32,32)
        state=new_state(model);before=state_hash(model);copies=[s.clone() for s in state]
        a,aa=score_state(model,x,state);b,bb=score_state(model,x,state)
        self.assertEqual(a,b)
        for k in aa:self.assertTrue(torch.equal(aa[k],bb[k]))
        self.assertTrue(torch.equal(aa['logits'],model(x)))
        self.assertEqual(before,state_hash(model))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(state,copies)))
        self.assertEqual(list(inspect.signature(score_state).parameters),['model','pixels','state'])
    def test_js_known_values(self):
        p=torch.tensor([[1.,0.]],dtype=torch.float64);q=p.flip(-1)
        self.assertEqual(float(js_divergence(p,p)),0.)
        self.assertAlmostEqual(float(js_divergence(p,q)),float(torch.tensor(2.).double().log()))
    def test_admissibility_and_fixed_tie(self):
        scores={c:dict(consistency_js=1.,flip_top1_agreement=.5) for c in CONTEXTS}
        self.assertEqual(select_pc_safe(scores)['selected'],'clean')
        scores['brightness_dark']['consistency_js']=.1;scores['brightness_dark']['flip_top1_agreement']=.4
        self.assertEqual(select_pc_safe(scores)['selected'],'clean')
        scores['contrast_low']['consistency_js']=.2;scores['gaussian_noise']['consistency_js']=.2
        self.assertEqual(select_pc_safe(scores)['selected'],'contrast_low')
        self.assertEqual(select_pc_safe(scores),select_pc_safe(scores))

if __name__=='__main__':unittest.main()
