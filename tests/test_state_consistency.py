import unittest
import torch
from src.context.state_consistency import post_state_signature,choose_state
from src.context.source_signature import source_moments,source_signature
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.eval.common import state_hash


class StateConsistencyTest(unittest.TestCase):
    def test_zero_signature_and_score_exact(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=ContextFedAvgCNN(3,10,1600).eval();ref=source_moments(model,torch.randn(4,3,32,32));zero=new_state(model)
        for x in (torch.randn(4,3,32,32),torch.ones(4,3,32,32)):
            psi=post_state_signature(model,x,zero,ref);phi=source_signature(model,x,ref)
            self.assertTrue(torch.equal(psi,phi));self.assertEqual(float(psi.square().sum()),float(phi.square().sum()))
            self.assertTrue(torch.equal(model(x,zero),model(x)))

    def test_sequential_path_and_immutability(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=ContextFedAvgCNN(3,10,1600).eval();x=torch.randn(4,3,32,32)
        ref=source_moments(model,x);state=[.05*torch.randn_like(v) for v in new_state(model)]
        copies=[v.clone() for v in state];before=state_hash(model);inputs=[];outputs=[]
        h1=model.conv2.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
        h2=model.conv2.register_forward_hook(lambda module,args,out:outputs.append(out.detach().clone()))
        psi=post_state_signature(model,x,state,ref);logits=model(x,state)
        h1.remove();h2.remove();self.assertTrue(torch.equal(inputs[0],inputs[1]))
        z=outputs[0]*(1+state[2][None,:,None,None])+state[3][None,:,None,None]
        self.assertTrue(torch.equal(model.fc(model.fc1(z.flatten(1))),logits))
        mu=z.double().mean((0,2,3));sigma=(z.double().var((0,2,3),unbiased=False)+1e-6).sqrt()
        expected=torch.cat(((mu-ref[1][0])/(ref[1][1]+1e-6),torch.log((sigma+1e-6)/(ref[1][1]+1e-6))))
        self.assertTrue(torch.equal(psi[64:],expected));self.assertEqual(state_hash(model),before)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(state,copies)))

    def test_zero_first_tie_order(self):
        scores=dict.fromkeys(('gaussian_blur','gaussian_noise','contrast_low','brightness_dark','clean'),1.)
        self.assertEqual(choose_state(scores)['selected'],'clean')
        scores['gaussian_blur']=.2;self.assertAlmostEqual(choose_state(scores)['zero_improvement'],.8)


if __name__=='__main__':unittest.main()
