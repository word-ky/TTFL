"""T017R noise-free sanity only; never intervenes in estimated states or prevalence."""
import numpy as np


def noise_free_sanity(pi_star,pi_true,selected,true_values,canonical):
    maximum=max(true_values);arg=[s for s,u in enumerate(true_values) if u==maximum]
    assert canonical==arg[0]
    assert np.max(np.abs(np.asarray(pi_star)-pi_true))<=1e-8
    assert selected in arg
    assert maximum-true_values[selected]==0
    if len(arg)==1:assert selected==canonical
    return dict(argmax_size=len(arg),tied=len(arg)>1,canonical_agreement=selected==canonical,exact_regret_zero=True)
