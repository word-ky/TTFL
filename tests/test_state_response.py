import unittest
import numpy as np
import hashlib
from src.context.state_response import state_response
from src.context.representation_prototypes import class_pools,sample_class_pools


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
    def test_t022_exact_count_namespace(self):
        x=np.arange(4*20*40,dtype=float).reshape(4,20,40);labels=np.tile(np.arange(20)%10,(4,1));counts=[2]*10
        pools,ids=class_pools(x,labels,1);mean,draw=sample_class_pools(pools,ids,1,counts,'L','A','clean',0,task='T022')
        seed=int(hashlib.sha256(b'T022|matched|L|1|A|clean|0').hexdigest(),16);rng=np.random.Generator(np.random.PCG64(seed))
        expected=np.concatenate([p[rng.integers(0,len(p),size=2)] for p in ids])
        np.testing.assert_array_equal(draw,expected);np.testing.assert_array_equal(mean,x.reshape(-1,40)[draw].mean(0))
