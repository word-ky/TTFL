import unittest
import torch
from src.data.covariate import corrupt, TARGETS


class CovariateTest(unittest.TestCase):
    def test_weighted_gate_and_paired_summary(self):
        from scripts.eval_pfllib_covariate_context import summarize
        rows=[]
        for target in TARGETS:
            for ctx, scores in {'none':[10,10], 'correct':[20,12],
                                'wrong_clean':[10,10], 'wrong_alt':[10,10],
                                'shuffled':[20,20], 'noise_image':[11,11]}.items():
                for cid,(acc,n) in enumerate(zip(scores,[1,3])):
                    rows.append(dict(target=target,context=ctx,client=cid,accuracy=acc,n_query=n,
                        support_loss_before=1.,support_loss_after=.5,gamma_norm=.1,beta_norm=.2))
        summaries,pairs=summarize(rows)
        correct=next(r for r in summaries if r['context']=='correct')
        self.assertEqual(correct['accuracy'],14.)
        self.assertEqual(correct['macro_client_accuracy'],16.)
        self.assertTrue(all(p['passed'] for p in pairs))
        self.assertEqual(pairs[0]['wrong_clean_paired']['mean'],6.)
        for row in rows:
            if row['context']=='noise_image':
                row['accuracy']=14.
        self.assertFalse(any(p['passed'] for p in summarize(rows)[1]))

    def test_formulas_and_identity(self):
        x = torch.linspace(-1, 1, 96).reshape(2, 3, 4, 4)
        self.assertTrue(torch.equal(corrupt(x, 'clean', [4, 9], 0), x))
        z = (x+1)/2
        self.assertTrue(torch.allclose(corrupt(x, 'brightness_dark', [4, 9], 0),
                                       2*(z*.45)-1))
        expected = 2*((z-z.mean((-2,-1),keepdim=True))*.35 +
                      z.mean((-2,-1),keepdim=True))-1
        self.assertTrue(torch.allclose(corrupt(x, 'contrast_low', [4, 9], 0), expected))
        constant = torch.full((2,3,8,8), .3)
        self.assertTrue(torch.allclose(corrupt(constant, 'gaussian_blur', [4,9], 0), constant))

    def test_deterministic_by_sample_not_order_and_no_mutation(self):
        x = torch.linspace(-1, 1, 384).reshape(2,3,8,8)
        before = x.clone()
        for name in (*TARGETS, 'noise_image'):
            result = corrupt(x, name, [4,9], 2)
            repeat = corrupt(x, name, [4,9], 2)
            reverse = corrupt(x.flip(0), name, [9,4], 2).flip(0)
            self.assertTrue(torch.equal(result, repeat))
            self.assertTrue(torch.equal(result, reverse))
            self.assertEqual(result.shape, x.shape)
            self.assertGreaterEqual(result.min().item(), -1.000001)
            self.assertLessEqual(result.max().item(), 1.000001)
        self.assertTrue(torch.equal(x, before))
        self.assertFalse(torch.equal(corrupt(x, 'gaussian_noise', [4,9], 2),
                                     corrupt(x, 'gaussian_noise', [4,9], 3)))


if __name__ == '__main__':
    unittest.main()
