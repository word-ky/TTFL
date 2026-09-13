import inspect
import unittest
from fractions import Fraction
import numpy as np
from src.context.class_factorization import class_templates,predict_utilities
from src.context.disjoint_factorization import exact_argmax

class ClassFactorizationTest(unittest.TestCase):
    def setUp(self):
        self.total=np.full((3,10),10,dtype=int);self.correct=np.zeros((3,5,5,10),dtype=int)
        self.correct[1,:,1,0]=4;self.correct[2,:,2,1]=6
    def test_leave_one_out_changes(self):
        before=[class_templates(self.correct,self.total,i) for i in range(3)]
        self.correct[0,1,3,2]=3
        after=[class_templates(self.correct,self.total,i) for i in range(3)]
        self.assertEqual(before[0],after[0]);self.assertNotEqual(before[1],after[1]);self.assertNotEqual(before[2],after[2])
        totals=self.total.copy();totals[0,2]+=1
        self.assertEqual(after[0],class_templates(self.correct,totals,0))
    def test_clean_controls_and_no_eval_api(self):
        template=class_templates(self.correct,self.total,0);actual=[Fraction(0),Fraction(1,10),Fraction(1,5),Fraction(0),Fraction(0)]
        own=[9,1]+[0]*8;other=[1,9]+[0]*8
        p,c,h=predict_utilities(template,actual,own)
        self.assertEqual(h[0],actual);self.assertEqual(exact_argmax(h[0]),exact_argmax(actual))
        self.assertTrue(all(template[0][k][s]-template[0][k][s]==0 for k in range(10) for s in range(5)))
        self.assertNotEqual(c,predict_utilities(template,actual,other)[1])
        self.assertEqual(predict_utilities(template,actual,[1]*10),predict_utilities(template,actual,[10]*10))
        self.assertEqual(list(inspect.signature(class_templates).parameters),['train_correct','train_totals','target'])
        self.assertEqual(list(inspect.signature(predict_utilities).parameters),['template','actual_persistent','composition'])
        # Evaluation-half labels are deliberately absent; changing them cannot enter either function.
        evaluation_labels=[0,1];first=predict_utilities(template,actual,own);evaluation_labels[:]=[9,9]
        self.assertEqual(first,predict_utilities(template,actual,own))
    def test_zero_denominator_observed_contract(self):
        self.total[:,0]=0
        with self.assertRaises(AssertionError):class_templates(self.correct,self.total,0)

if __name__=='__main__':unittest.main()
