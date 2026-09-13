import unittest
import torch
from src.data.semantic_pairing import semantic_pairings,random_offsets


class SemanticPairingTest(unittest.TestCase):
    def test_extrema_ties_and_target_histogram(self):
        y=torch.tensor([0,0,0,1,1,1])
        rows=semantic_pairings(y,7)
        self.assertEqual(rows[0]['offset'],3)
        self.assertEqual(rows[0]['same_label_fraction'],0)
        self.assertEqual(rows[1]['offset'],1)
        self.assertAlmostEqual(rows[1]['same_label_fraction'],2/3)
        for r in rows:
            perm=torch.tensor(r['permutation'])
            self.assertTrue((perm!=torch.arange(len(y))).all())
            self.assertTrue(torch.equal(y[perm].sort().values,y.sort().values))
        pure=semantic_pairings(torch.zeros(6,dtype=torch.long),7)
        self.assertEqual([r['offset'] for r in pure[:2]],[1,1])

    def test_random_distinct_deterministic_label_independent(self):
        for n in (2,3,6,30,64):
            chosen=random_offsets(n,9)
            self.assertEqual(len(chosen),min(8,n-1))
            self.assertEqual(len(set(chosen)),len(chosen))
            self.assertEqual(chosen,random_offsets(n,9))
            self.assertTrue(all(0<offset<n for offset in chosen))
            p1=semantic_pairings(torch.zeros(n,dtype=torch.long),9)
            p2=semantic_pairings(torch.arange(n)%3,9)
            self.assertEqual([r['offset'] for r in p1[2:]],[r['offset'] for r in p2[2:]])
