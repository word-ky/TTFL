"""Exact rational state utilities using a shared denominator, for batched diagnostics."""
from math import lcm


def compile_template(template):
    denominator=lcm(*(v.denominator for row in template for v in row))
    coefficients=[[template[k][s].numerator*(denominator//template[k][s].denominator) for k in range(10)] for s in range(5)]
    return coefficients,denominator


def exact_lookup(pi,coefficients,denominator,true_counts):
    ratios=[float(v).as_integer_ratio() for v in pi]
    binary_denominator=max(d for n,d in ratios)
    weights=[n*(binary_denominator//d) for n,d in ratios]
    numerators=[sum(w*c for w,c in zip(weights,row)) for row in coefficients]
    maximum=max(numerators);selected=numerators.index(maximum)
    true_numerators=[sum(int(n)*c for n,c in zip(true_counts,row)) for row in coefficients]
    error=max(abs(20*a-binary_denominator*b) for a,b in zip(numerators,true_numerators))
    return selected,error/(20*binary_denominator*denominator)
