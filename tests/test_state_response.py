import unittest
import numpy as np
from src.context.state_response import state_response


class StateResponseTest(unittest.TestCase):
    def test_exact_order_and_dimensions(self):
        for d in (10,512):
            x=np.arange(5*20*d,dtype=np.float32).reshape(5,20,d)
            phi=state_response(x);self.assertEqual(phi.shape,(20,4*d))
            for k in range(4):np.testing.assert_array_equal(phi[:,k*d:(k+1)*d],x[k+1]-x[0])
    def test_common_component_cancels_without_rescaling(self):
        x=np.arange(5*20*10,dtype=np.float64).reshape(5,20,10)
        common=np.arange(20*10).reshape(20,10)
        np.testing.assert_array_equal(state_response(x+common),state_response(x))
        np.testing.assert_array_equal(state_response(3*x),3*state_response(x))
