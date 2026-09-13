import unittest
import torch
from src.models.pfllib import FedAvgCNN, ContextFedAvgCNN
from src.adaptation.affine import adapt,new_state
from src.eval.common import state_hash,evaluate


class PFLModelTest(unittest.TestCase):
    def test_four_shapes_neutral_frozen_and_classes(self):
        torch.set_num_threads(2)
        for channels,size,classes,dim in [(1,28,10,1024),(3,32,10,1600),(3,32,100,1600),(3,64,200,10816)]:
            torch.manual_seed(7)
            baseline = FedAvgCNN(channels,classes,dim).eval()
            model = ContextFedAvgCNN(channels,classes,dim).eval()
            model.load_state_dict(baseline.state_dict())
            x,y = torch.randn(4,channels,size,size),torch.tensor([0,1,classes-1,classes-1])
            self.assertTrue(torch.equal(baseline(x),model(x,new_state(model))))
            m,state,losses = adapt(model,x,y,1,.01)
            self.assertEqual(state_hash(model),state_hash(m))
            self.assertTrue(any(torch.count_nonzero(p) for p in state))
            result,_ = evaluate(m,x,y,state,num_classes=classes)
            self.assertEqual(len(result['per_class_accuracy']),classes)
            self.assertIsNotNone(result['per_class_accuracy'][-1])


if __name__=='__main__':
    unittest.main()
