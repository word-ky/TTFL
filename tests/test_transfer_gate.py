import unittest
from scripts.eval_t007r import primary_gate


class TransferGateTest(unittest.TestCase):
    def test_recovery_and_both_context_margins(self):
        self.assertTrue(primary_gate(40.,20.,30.,25.,25.)['passed'])
        self.assertFalse(primary_gate(40.,20.,29.99,20.,20.)['passed'])
        self.assertFalse(primary_gate(40.,20.,30.,25.01,20.)['passed'])
        self.assertFalse(primary_gate(40.,20.,30.,20.,25.01)['passed'])

    def test_nonpositive_headroom_and_half_point_floor(self):
        self.assertIsNone(primary_gate(20.,20.,30.,10.,10.)['recovery'])
        self.assertFalse(primary_gate(20.,20.,30.,10.,10.)['passed'])
        self.assertEqual(primary_gate(20.1,20.,20.05,10.,10.)['tau_pp'],.5)


if __name__=='__main__':unittest.main()
