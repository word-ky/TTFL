import unittest
from fractions import Fraction
from src.context.rotation_alignment import spearman,gates,C

class RotationAlignmentTest(unittest.TestCase):
    def test_exact_tied_ranks(self):
        self.assertAlmostEqual(spearman([4,3,2,1,0],[0,1,2,3,4]),1.)
        self.assertAlmostEqual(spearman([0,0,1,2,3],[4,4,3,2,1]),1.)
        self.assertIsNone(spearman([1]*5,[0,1,2,3,4]))
    def test_inherited_boundaries(self):
        aa=[dict(context=t,capture=.8,delta_vs_BBSE_pp=-.5) for t in C]
        rr=[dict(context=t,baseline_mean=1.,new_mean=.85,baseline_p90=1.,new_p90=1.05) for t in C]
        self.assertEqual(gates(aa,rr,rr)['diagnosis'],'T023-R')
        aa[1]['delta_vs_BBSE_pp']=-.50001
        self.assertEqual(gates(aa,rr,rr)['diagnosis'],'T023-X')
