import inspect
import unittest
from fractions import Fraction
from src.context.disjoint_factorization import query_halves,factor_utilities,exact_argmax

class DisjointFactorizationTest(unittest.TestCase):
    def test_label_blind_partition(self):
        ids=[41,2,99,6,12,34,78]
        a=query_halves(ids,'T013-S0',9)
        self.assertEqual(a,query_halves(ids,'T013-S0',9));self.assertFalse(set(a[0])&set(a[1]))
        self.assertEqual(sorted(a[0]+a[1]),list(range(7)));self.assertLessEqual(abs(len(a[0])-len(a[1])),1)
        self.assertEqual(list(inspect.signature(query_halves).parameters),['original_ids','salt','client'])
    def test_equal_client_leave_one_out_and_ties(self):
        rows=[[[0]*5 for _ in range(5)] for _ in range(3)]
        rows[0][0]=[0,1,1,0,0];rows[0][1]=[0,2,2,0,0]
        rows[1][1]=[0,1,1,0,0];rows[2][1]=[0,9,9,0,0]
        p,c=factor_utilities(rows,[2,2,18])
        self.assertEqual(c[0][1][1],Fraction(1,2))
        self.assertEqual(exact_argmax([p[0][s]+c[0][1][s] for s in range(5)]),[1,2])
        rows[0][1]=[0,0,0,0,0]
        self.assertEqual(factor_utilities(rows,[2,2,18])[1][0],c[0])
        self.assertEqual(list(inspect.signature(factor_utilities).parameters),['train_correct','train_totals'])

if __name__=='__main__':unittest.main()
