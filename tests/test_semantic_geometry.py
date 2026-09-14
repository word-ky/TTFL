import hashlib,unittest
from fractions import Fraction as F
from src.context.semantic_geometry import *

class SemanticGeometryTest(unittest.TestCase):
    def test_counter_and_scaling(self):
        p=[F(1)]+[F(0)]*9;q,k,deg=counter_state(p)
        self.assertEqual(k,1);self.assertFalse(deg);self.assertEqual(q[1],1)
        for lam in LAMBDAS:self.assertEqual(tv(p,interpolate(p,q,lam)),lam*tv(p,q))
        self.assertTrue(counter_state([F(1,10)]*10)[2])
    def test_pair_and_ties(self):
        ids=list(range(100));pair=pairing('T013-S0',0,ids)
        self.assertEqual(set(pair.values()),set(ids));self.assertTrue(all(i!=j for i,j in pair.items()))
        order=sorted(ids,key=lambda i:hashlib.sha256(f'T024|pair|T013-S0|0|{i}'.encode()).hexdigest());self.assertEqual(pair[order[0]],order[1])
        self.assertEqual(argmax([F(0),F(1,3),F(1,3),F(0),F(-1)]),[1,2])
    def test_exact_gate_boundaries(self):
        rows=[]
        for bank in ('A','B'):
            for salt in SALTS:
                for lam in LAMBDAS:
                    for i in range(10):rows.append(dict(bank=bank,salt=salt,lambda_=lam,degenerate=False,stale_regret=F(1,100) if lam==1 else F(1,200),mismatch_regret=F(1,100),state_switch=i<3))
        for r in rows:r['lambda']=r.pop('lambda_')
        self.assertEqual(gate_summary(rows)['diagnosis'],'T024-S')
        for r in rows:
            if r['bank']=='A':r['state_switch']=False
        self.assertEqual(gate_summary(rows)['diagnosis'],'T024-N')
    def test_current_is_oracle(self):
        p=[F(1)]+[F(0)]*9;q=[F(0),F(1)]+[F(0)]*8;m=[[F(0)]*5 for _ in range(10)];m[0][1]=F(1,5);m[1][2]=F(1,5)
        r=evaluate(p,q,p,m);self.assertEqual((r['s_hist'],r['s_current'],r['s_mismatch']),(1,2,1));self.assertEqual(r['stale_regret'],F(1,5));self.assertEqual(r['mismatch_regret'],F(1,5))
