import unittest
from scripts.summarize_t006 import decision_flags,correlation


class SemanticSummaryTest(unittest.TestCase):
    def test_tau_and_strict_marginal_bounds(self):
        r=decision_flags(34,30,33,32.5,31,32)
        self.assertEqual(r['tau_pp'],1)
        self.assertTrue(r['S_A'])
        self.assertEqual(r['recovery_correct'],.75)
        r=decision_flags(34,33,33.5,33,33,33)
        self.assertEqual(r['tau_pp'],.5)
        self.assertFalse(r['S_C'])  # abs marginal exactly tau, requires strict <.

    def test_correlation_constant_and_centered_example(self):
        self.assertIsNone(correlation([0,0,0],[1,2,3]))
        self.assertAlmostEqual(correlation([-.1,0,.1,-.1,0,.1],[-1,0,1,-2,0,2]),.948683298,places=7)
