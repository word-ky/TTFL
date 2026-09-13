import inspect
import unittest
import torch
from src.data.pairing import target_derangement
from src.models.pfllib import ContextFedAvgCNN


class PairingTest(unittest.TestCase):
    def test_derangement_deterministic_label_free(self):
        self.assertEqual(list(inspect.signature(target_derangement).parameters),['size','client','seed'])
        for size in (1,2,3,64):
            for cid in (0,1,99):
                perm,offset=target_derangement(size,cid)
                self.assertTrue(torch.equal(perm.sort().values,torch.arange(size)))
                if size>1:
                    self.assertTrue((perm!=torch.arange(size)).all())
                    self.assertGreater(offset,0)
                self.assertTrue(torch.equal(perm,target_derangement(size,cid)[0]))

    def test_target_multiset_both_layers_preserved(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=ContextFedAvgCNN(3,10,1600).eval()
        clean=torch.randn(4,3,32,32)
        perm,_=target_derangement(4,2)
        inverse=perm.argsort()
        h,hp=clean,clean[perm]
        self.assertTrue(torch.equal(hp[inverse],h))
        for block in (model.conv1,model.conv2):
            h,hp=block(h),block(hp)
            self.assertTrue(torch.equal(hp[inverse],h))
