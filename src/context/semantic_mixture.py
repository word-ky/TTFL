"""T015 fixed posterior mixtures and immutable class-utility retrieval."""
from fractions import Fraction
import hashlib
import json
import torch

def require_phaseA_freeze(output):
    receipt=json.loads((output/'phaseA_freeze.json').read_text())
    assert hashlib.sha256((output/'phaseA_policy_choices.json.gz').read_bytes()).hexdigest()==receipt['choices_sha256']

def mixtures_from_logits(logits,true_context,source_context):
    probabilities={c:x.double().softmax(-1) for c,x in logits.items()}
    zero=probabilities['clean']
    return dict(pi_zero_soft=zero.mean(0).tolist(),
        pi_zero_hard=(torch.bincount(zero.argmax(-1),minlength=10).double()/len(zero)).tolist(),
        pi_oracle_state_soft=probabilities[true_context].mean(0).tolist(),
        pi_source_state_soft=probabilities[source_context].mean(0).tolist())

def choose_utility(pi,template):
    # Preserve exact template fractions; binary float64 mixture masses are converted exactly.
    weights=[v if isinstance(v,Fraction) else Fraction(float(v)) for v in pi]
    values=[sum((weights[k]*template[k][s] for k in range(10)),Fraction()) for s in range(5)]
    mx=max(values);arg=[i for i,x in enumerate(values) if x==mx]
    return arg[0],arg,values
