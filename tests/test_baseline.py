import io
import unittest
import torch
import torch.nn.functional as F
from src.models import DigitCNN
from src.eval.common import evaluate, state_hash


class BaselineTest(unittest.TestCase):
    def test_train_checkpoint_and_metrics(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        m = DigitCNN()
        x, y = torch.randn(20, 3, 32, 32), torch.arange(10).repeat(2)
        m.train()
        before = state_hash(m)
        optimizer = torch.optim.SGD(m.parameters(), lr=.01)
        loss = F.cross_entropy(m(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertNotEqual(before, state_hash(m))
        m.eval()
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)
        buffer.seek(0)
        loaded = DigitCNN().eval()
        loaded.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(m(x), loaded(x)))
        result, pred = evaluate(loaded, x, y)
        self.assertAlmostEqual(result['accuracy'], (pred == y).float().mean().item()*100)
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))


if __name__ == '__main__':
    unittest.main()
