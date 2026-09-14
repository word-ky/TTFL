import unittest
import numpy as np
from src.context.constrained_prevalence import ActiveSetCLS, direct_kkt, face_reference


def synthetic_bank():
    rng = np.random.default_rng(180018)
    for condition in (3., 50., 100., 200.):
        u = np.linalg.qr(rng.normal(size=(10, 10)))[0]
        v = np.linalg.qr(rng.normal(size=(10, 10)))[0]
        C = u @ np.diag(np.geomspace(1., 1./condition, 10)) @ v.T
        for _ in range(3):
            truth = rng.dirichlet(np.ones(10))
            yield C, C @ truth + rng.normal(0, .015, 10)


class ActiveSetTest(unittest.TestCase):
    def test_full_rank_noise_free(self):
        rng = np.random.default_rng(18)
        for _ in range(6):
            C = np.eye(10) + rng.normal(0, .03, (10, 10))
            true = rng.dirichlet(np.ones(10))
            p, r = ActiveSetCLS(C).solve(C @ true)
            np.testing.assert_allclose(p, true, atol=1e-8, rtol=0)
            self.assertLessEqual(r['direct_KKT'], 1e-10)

    def test_ill_conditioned_reference_and_insert_remove(self):
        actions = set()
        for C, q in synthetic_bank():
            before = C.copy(); old = q.copy()
            solver = ActiveSetCLS(C); p, r = solver.solve(q)
            ref, obj = face_reference(C, q)
            np.testing.assert_allclose(p, ref, atol=1e-7, rtol=0)
            self.assertLessEqual(abs(r['objective']-obj), 1e-10)
            self.assertLessEqual(direct_kkt(C, q, p), 1e-10)
            self.assertLessEqual(abs(p.sum()-1), 1e-12)
            self.assertGreaterEqual(p.min(), -1e-12)
            self.assertLessEqual(2*r['objective'], 2*r['initial_objective']+1e-12)
            replay, rr = solver.solve(q)
            np.testing.assert_array_equal(p, replay); self.assertEqual(r, rr)
            np.testing.assert_array_equal(C, before); np.testing.assert_array_equal(q, old)
            actions.update(x['action'] for x in r['trace'])
        self.assertIn('add', actions); self.assertIn('remove', actions)

    def test_boundary_optima(self):
        for zeros in (1, 2, 6):
            q = np.full(10, 1/(10-zeros)); q[:zeros] = -.1
            p, r = ActiveSetCLS(np.eye(10)).solve(q)
            self.assertEqual(int((p == 0).sum()), zeros)
            self.assertLessEqual(r['direct_KKT'], 1e-10)


if __name__ == '__main__':
    unittest.main()
