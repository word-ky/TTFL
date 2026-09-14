import unittest
from src.context.representation_gates import inherited_gates,regret_gate,diagnosis,C


class RepresentationGatesTest(unittest.TestCase):
    def test_inherited_both_banks_all_salts_safety(self):
        rows=[dict(context=t,bank=b,salt=s,capture=.8,capture_median=.8,delta_vs_BBSE_pp=-.5,clean_delta_vs_zero=-.5) for t in C for b in ('A','B') for s in range(4)]
        self.assertTrue(inherited_gates(rows,rows,rows)['REP_SRC_A'])
        bad=[dict(r) for r in rows];bad[8]['delta_vs_BBSE_pp']=-.5001
        self.assertFalse(inherited_gates(rows,bad)['REP_REAL_A'])
        bad[8]['capture']=.7999;bad[16]['capture']=.7999
        self.assertFalse(inherited_gates(rows,bad)['REP_REAL_A'])
    def test_regret_requires_both_conditions(self):
        rows=[dict(context=t,baseline_mean=.1,new_mean=.084,baseline_p90=.2,new_p90=.21) for t in C for _ in range(8)]
        self.assertTrue(regret_gate(rows)['passed']);rows[8]['new_p90']=.211;rows[16]['baseline_mean']=0
        self.assertFalse(regret_gate(rows)['passed'])
    def test_preferred_logits_and_no_forced_success(self):
        good=dict(REP_MATCH_A=True,REP_REAL_A=True);bad=dict(REP_MATCH_A=True,REP_REAL_A=False)
        self.assertEqual(diagnosis(dict(L=good,H=good)),'T021-L')
        self.assertEqual(diagnosis(dict(L=bad,H=good)),'T021-H')
        self.assertEqual(diagnosis(dict(L=bad,H=bad)),'T021-M')
