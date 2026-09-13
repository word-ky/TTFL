"""T014 exact leave-one-client-out class templates and composition utilities."""
from fractions import Fraction

def class_templates(train_correct,train_totals,target):
    correct=train_correct.sum(axis=0)-train_correct[target]
    totals=train_totals.sum(axis=0)-train_totals[target]
    assert (totals>0).all()
    return [[[Fraction(int(correct[c,s,k])-int(correct[c,0,k]),int(totals[k])) for s in range(5)] for k in range(10)] for c in range(5)]

def predict_utilities(template,actual_persistent,composition):
    pi=[Fraction(int(n),int(sum(composition))) for n in composition]
    classes=[[sum((pi[k]*template[c][k][s] for k in range(10)),Fraction()) for s in range(5)] for c in range(5)]
    hybrid=[[actual_persistent[s]+classes[c][s]-classes[0][s] for s in range(5)] for c in range(5)]
    return classes[0],classes,hybrid
