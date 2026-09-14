import json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.context.constrained_prevalence import ActiveSetCLS,direct_kkt


class CycleFallbackTest(unittest.TestCase):
    def test_exact_two_cycles(self):
        rows=json.loads((Path(__file__).parent/'fixtures/t020r_cycles.json').read_text())
        for row in rows:
            C=np.array(row['C']);q=np.array(row['q']);solver=ActiveSetCLS(C)
            pi,r=solver.solve(q)
            self.assertEqual(r['solver_path'],'cycle_face_fallback');self.assertEqual(r['fallback_faces_tested'],1023)
            self.assertIn(r['cycle_active_set'],[x['active'] for x in r['trace']])
            self.assertIn('repeated='+str(tuple(r['cycle_active_set'])),row['old_error'])
            self.assertLessEqual(direct_kkt(C,q,pi),1e-10);self.assertLessEqual(abs(pi.sum()-1),1e-12);self.assertGreaterEqual(pi.min(),-1e-12)
            np.testing.assert_allclose(pi,row['reference_pi'],rtol=0,atol=1e-9)
            self.assertLessEqual(abs(r['objective']-row['reference_objective']),1e-12)
            pp,rr=solver.solve(q);np.testing.assert_array_equal(pi,pp);self.assertEqual(r,rr)
    def test_normal_path_does_not_enumerate(self):
        with patch('src.context.constrained_prevalence.face_reference',side_effect=AssertionError('unexpected fallback')):
            pi,r=ActiveSetCLS(np.eye(10)).solve(np.full(10,.1))
        self.assertEqual(r['solver_path'],'active_set');np.testing.assert_allclose(pi,.1,rtol=0,atol=1e-15)
    def test_uncertified_fallback_stops(self):
        row=json.loads((Path(__file__).parent/'fixtures/t020r_cycles.json').read_text())[0]
        with patch('src.context.constrained_prevalence.face_reference',return_value=(np.ones(10),0.)):
            with self.assertRaisesRegex(RuntimeError,'certification failed'):ActiveSetCLS(row['C']).solve(row['q'])
