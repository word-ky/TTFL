"""T020 label-blind position bootstrap and exact expected-utility decisions."""
import hashlib
from fractions import Fraction
import numpy as np


def position_seed(client,bank,context,replica):
    return int.from_bytes(hashlib.sha256(f'T020|{client}|{bank}|{context}|{replica}'.encode()).digest(),'big')


def bootstrap_positions(client,bank,context,replica):
    rng=np.random.Generator(np.random.PCG64(position_seed(client,bank,context,replica)))
    return rng.integers(0,20,size=20)


def shared_integer_weights(prevalences):
    ratios=[[float(v).as_integer_ratio() for v in row] for row in prevalences]
    denominator=max(d for row in ratios for n,d in row)
    return [[n*(denominator//d) for n,d in row] for row in ratios],denominator


def exact_utilities_from_weights(weights,coefficients):
    return [[sum(w*c for w,c in zip(row,col)) for col in coefficients] for row in weights]


def canonical_and_mask(row):
    maximum=max(row);indices=[j for j,v in enumerate(row) if v==maximum]
    return indices[0],sum(1<<j for j in indices)


def expected_regret_objects(numerators,denominator,point_state):
    """All replicas share an exact utility denominator; no labels/true utilities."""
    assert len(numerators)==256
    sums128=[sum(row[s] for row in numerators[:128]) for s in range(5)]
    sums256=[sum(row[s] for row in numerators) for s in range(5)]
    ber128=canonical_and_mask(sums128)[0];ber256=canonical_and_mask(sums256)[0]
    states,masks=zip(*(canonical_and_mask(row) for row in numerators))
    freq128=np.bincount(states[:128],minlength=5)/128;freq256=np.bincount(states,minlength=5)/256
    mean128=[Fraction(n,128*denominator) for n in sums128];mean256=[Fraction(n,256*denominator) for n in sums256]
    regrets=[Fraction(sum(max(row)-row[s] for row in numerators),256*denominator) for s in range(5)]
    assert regrets[ber256]==min(regrets)
    diffs=[(row[ber256]-row[point_state])/denominator for row in numerators]
    ordered=sorted(sums256,reverse=True)
    diagnostics=dict(BER128=ber128,BER256=ber256,point_state=int(point_state),agreement=ber128==ber256,
        mean_utility128=[float(x) for x in mean128],mean_utility256=[float(x) for x in mean256],mean_utility128_exact=[str(x) for x in mean128],mean_utility256_exact=[str(x) for x in mean256],
        max_mean_utility_difference=float(max(abs(x-y) for x,y in zip(mean128,mean256))),
        vote128=freq128.tolist(),vote256=freq256.tolist(),max_vote_frequency_difference=float(np.max(np.abs(freq128-freq256))),
        p_mode=float(freq256.max()),vote_entropy=float(-sum(v*np.log(v) for v in freq256 if v>0)),
        point_bootstrap_expected_regret=float(regrets[point_state]),BER_bootstrap_expected_regret=float(regrets[ber256]),
        point_bootstrap_expected_regret_exact=str(regrets[point_state]),BER_bootstrap_expected_regret_exact=str(regrets[ber256]),
        top_mean_utility_margin=float(Fraction(ordered[0]-ordered[1],256*denominator)),
        BER_replica_optimal_fraction=sum(bool(mask&(1<<ber256)) for mask in masks)/256,
        BER_minus_point_utility_p10=float(np.quantile(diffs,.1)),BER_minus_point_utility_median=float(np.median(diffs)),BER_minus_point_utility_p90=float(np.quantile(diffs,.9)))
    return np.array(states,dtype=np.uint8),np.array(masks,dtype=np.uint8),diagnostics


def frozen_context_dispatch(choices,context_ids):
    """T020 Section 6 literal dispatch to precomputed chosen-context BER state."""
    return np.take_along_axis(choices,context_ids[:,:,None,:],axis=2)
