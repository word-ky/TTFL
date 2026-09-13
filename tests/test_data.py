import unittest
import torch
from src.data.digits import indices, balanced_indices, normalized


class DataTest(unittest.TestCase):
    def test_balanced_and_disjoint(self):
        y = torch.arange(10).repeat_interleave(100)
        support = balanced_indices(y, 20, 31)
        query = indices(500, 200, 101)
        support_ids = {f'SVHN/train/{i}' for i in support.tolist()}
        query_ids = {f'SVHN/test/{i}' for i in query.tolist()}
        self.assertFalse(support_ids & query_ids)
        self.assertEqual(len(support.unique()), 200)
        self.assertTrue(torch.equal(torch.bincount(y[support]), torch.full((10,), 20)))
        self.assertTrue(torch.equal(support, balanced_indices(y, 20, 31)))

    def test_preprocess(self):
        x = normalized(torch.zeros(2, 16, 16, dtype=torch.uint8))
        self.assertEqual(x.shape, (2, 3, 32, 32))
        self.assertTrue(torch.equal(x, -torch.ones_like(x)))


if __name__ == '__main__':
    unittest.main()
