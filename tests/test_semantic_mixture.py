import inspect
import hashlib,json,tempfile
from pathlib import Path
import unittest
from fractions import Fraction
import torch
from src.context.semantic_mixture import mixtures_from_logits,choose_utility,require_phaseA_freeze
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state

class SemanticMixtureTest(unittest.TestCase):
    def test_privileged_phase_requires_frozen_choices(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(FileNotFoundError):require_phaseA_freeze(p)
            (p/'phaseA_policy_choices.json.gz').write_bytes(b'frozen')
            (p/'phaseA_freeze.json').write_text(json.dumps(dict(choices_sha256=hashlib.sha256(b'frozen').hexdigest())))
            require_phaseA_freeze(p)
            (p/'phaseA_policy_choices.json.gz').write_bytes(b'changed')
            with self.assertRaises(AssertionError):require_phaseA_freeze(p)
    def test_normalization_branch_isolation(self):
        torch.manual_seed(23);logits={c:torch.randn(20,10) for c in ('clean','dark','blur')}
        a=mixtures_from_logits(logits,'dark','dark');b=mixtures_from_logits(logits,'dark','blur')
        for k,v in a.items():
            self.assertAlmostEqual(sum(v),1.,places=14);self.assertTrue(all(x>=0 and torch.isfinite(torch.tensor(x)) for x in v))
            if k!='pi_source_state_soft':self.assertEqual(v,b[k])
        self.assertNotEqual(a['pi_source_state_soft'],b['pi_source_state_soft'])
        self.assertEqual(list(inspect.signature(mixtures_from_logits).parameters),['logits','true_context','source_context'])
    def test_clean_zero_and_determinism(self):
        torch.manual_seed(3);model=ContextFedAvgCNN(3,10,1600).eval();x=torch.randn(3,3,32,32)
        with torch.no_grad():
            a=model(x);b=model(x,new_state(model));self.assertTrue(torch.equal(a,b))
        m=mixtures_from_logits({'clean':b},'clean','clean')
        self.assertEqual(m['pi_zero_soft'],m['pi_oracle_state_soft']);self.assertEqual(m['pi_zero_soft'],m['pi_source_state_soft'])
    def test_choice_no_labels_and_exact_ties(self):
        template=[[Fraction(0),Fraction(k,10),Fraction(k,10),Fraction(0),Fraction(0)] for k in range(10)]
        result=choose_utility([.1]*10,template)
        self.assertEqual(result[:2],(1,[1,2]));self.assertEqual(result,choose_utility([.1]*10,template))
        self.assertEqual(list(inspect.signature(choose_utility).parameters),['pi','template'])

if __name__=='__main__':unittest.main()
