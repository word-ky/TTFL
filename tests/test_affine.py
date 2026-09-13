import unittest
import torch
from src.models import DigitCNN
from src.adaptation.affine import adapt, new_state, bn_from_support
from src.eval.common import state_hash


class AffineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        self.m = DigitCNN().eval()
        self.x = torch.randn(20, 3, 32, 32)
        self.y = torch.arange(10).repeat(2)

    def test_neutral_logits(self):
        state = new_state(self.m)
        self.assertLess((self.m(self.x) - self.m(self.x, state)).abs().max().item(), 1e-6)
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in state))

    def test_only_fast_state_and_reset(self):
        before = state_hash(self.m)
        adapted, state, losses = adapt(self.m, self.x, self.y, 3, .1)
        self.assertEqual(before, state_hash(adapted))
        self.assertEqual(before, state_hash(self.m))
        self.assertTrue(any(torch.count_nonzero(p) > 0 for p in state))
        self.assertTrue(all(p.grad is None for p in adapted.parameters()))
        self.assertLess(losses[-1], losses[0])
        _, state2, _ = adapt(self.m, self.x, self.y, 3, .1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(state, state2)))

    def test_bn_only_moments(self):
        before = state_hash(self.m)
        bn = bn_from_support(self.m, self.x)
        self.assertEqual(before, state_hash(self.m))
        for a, b in zip(self.m.parameters(), bn.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(self.m.blocks[0][1].running_var,
                                     bn.blocks[0][1].running_var))

    def test_same_checkpoint_all_methods(self):
        initial = state_hash(self.m)
        for full in (False, True):
            model, state, _ = adapt(self.m, self.x, self.y, 0, .01, full=full)
            self.assertEqual(initial, state_hash(model))
            self.assertTrue(torch.equal(self.m(self.x), model(self.x, state)))


if __name__ == '__main__':
    unittest.main()
