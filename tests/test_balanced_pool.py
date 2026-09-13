import unittest
from collections import Counter
from src.data.balanced_pool import calibration_pools, balanced_pairings


class BalancedPoolTest(unittest.TestCase):
    def test_disjoint_balanced_spread_order_independent(self):
        candidates = [dict(client_id=c, train_index=y*2+j, original_id=c*100+y*2+j, **{'class':y})
                      for c in range(12) for y in range(10) for j in range(2)]
        pools = calibration_pools(candidates)
        self.assertEqual(pools, calibration_pools(candidates[::-1]))
        self.assertFalse({r['original_id'] for r in pools['A']} & {r['original_id'] for r in pools['B']})
        for rows in pools.values():
            self.assertEqual(Counter(r['class'] for r in rows), dict.fromkeys(range(10),8))
            for c in range(10):
                self.assertEqual(len({r['client_id'] for r in rows if r['class']==c}),8)

    def test_exact_semantic_controls(self):
        labels = [c for c in range(10) for _ in range(8)]
        for name,p in balanced_pairings(labels).items():
            self.assertEqual(sorted(p),list(range(80)))
            self.assertTrue(all(i!=j for i,j in enumerate(p)))
            self.assertEqual(sum(labels[i]==labels[j] for i,j in enumerate(p)),80 if name.startswith('same') else 0)


if __name__ == '__main__': unittest.main()
