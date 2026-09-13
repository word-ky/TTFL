import unittest
from scripts.eval_t008_source_identifiability import identification_gate,retrieval_gate


class IdentificationGateTest(unittest.TestCase):
    def test_total_and_per_context(self):
        self.assertTrue(identification_gate([4,4,4,3,3]))
        self.assertFalse(identification_gate([4,4,4,4,2]))
        self.assertFalse(identification_gate([4,4,3,3,3]))

    def test_retrieval_both_bounds_and_nonpositive_gain(self):
        self.assertTrue(retrieval_gate(20,30,29,25,4)['passed'])
        self.assertFalse(retrieval_gate(20,30,28.99,20,4)['passed'])
        self.assertFalse(retrieval_gate(20,30,29,25.01,4)['passed'])
        self.assertIsNone(retrieval_gate(20,20,29,20,4)['retained_recovery'])


if __name__=='__main__':unittest.main()
