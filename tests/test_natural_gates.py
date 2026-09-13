import unittest
from scripts.eval_t009_natural_context import natural_id_gate,clean_safety_gate


class NaturalGateTest(unittest.TestCase):
    def test_total_and_context_floor(self):
        self.assertTrue(natural_id_gate([90,90,90,90,90]))
        self.assertFalse(natural_id_gate([100,100,100,100,79]))
        self.assertFalse(natural_id_gate([89,90,90,90,90]))

    def test_clean_half_point(self):
        self.assertTrue(clean_safety_gate(34.,33.5));self.assertFalse(clean_safety_gate(34.,33.499))


if __name__=='__main__':unittest.main()
