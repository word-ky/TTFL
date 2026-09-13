import inspect
import unittest
import torch
from src.context.source_signature import source_moments,source_signature,nearest_prototype
from src.data.microbatches import balanced_microbatches
from src.models.pfllib import ContextFedAvgCNN
from src.eval.common import state_hash


class SourceSignatureTest(unittest.TestCase):
    def test_exact_bank_reference_zero_and_heldout_formula(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=ContextFedAvgCNN(3,10,1600).eval();x=torch.randn(4,3,32,32)
        before=state_hash(model);reference=source_moments(model,x)
        self.assertTrue(torch.equal(source_signature(model,x,reference),torch.zeros(192,dtype=torch.float64)))
        shifted=x*.4-.2;moments=source_moments(model,shifted)
        expected=torch.cat([v for (m,s),(m0,s0) in zip(moments,reference)
                            for v in ((m-m0)/(s0+1e-6),torch.log((s+1e-6)/(s0+1e-6)))])
        self.assertTrue(torch.equal(source_signature(model,shifted,reference),expected))
        self.assertGreater(expected.square().sum().item(),0)
        self.assertEqual(state_hash(model),before)
        self.assertFalse(expected.requires_grad)
        self.assertEqual(list(inspect.signature(source_signature).parameters),['model','source','bank_reference'])
        self.assertEqual(list(inspect.signature(nearest_prototype).parameters),['signature','prototypes'])

    def test_squared_distance_margin_and_tie_order(self):
        prototypes={'clean':torch.tensor([0.,0.]),'dark':torch.tensor([3.,0.]),'noise':torch.tensor([0.,5.])}
        r=nearest_prototype(torch.tensor([2.,0.]),prototypes)
        self.assertEqual((r['selected'],r['second'],r['distance'],r['margin']),('dark','clean',1.,3.))
        r=nearest_prototype(torch.tensor([1.5,0.]),prototypes);self.assertEqual(r['selected'],'clean')

    def test_microbatch_partition_exact_and_ordered(self):
        labels=[c for c in range(10) for _ in range(8)];batches=balanced_microbatches(labels)
        self.assertEqual(sorted(i for b in batches for i in b),list(range(80)))
        for b,indices in enumerate(batches):
            self.assertEqual(len(indices),20)
            self.assertTrue(all(sum(labels[i]==c for i in indices)==2 for c in range(10)))
            self.assertEqual([i for i in indices if labels[i]==0],[2*b,2*b+1])


if __name__=='__main__':unittest.main()
