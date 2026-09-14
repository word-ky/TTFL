"""Fixed T020R blocks; exact same expected-utility functional at arbitrary R."""
from fractions import Fraction
import numpy as np
from src.context.bootstrap_regret import canonical_and_mask

BLOCKS={'Q0':(0,1024),'Q1':(1024,2048),'Q2':(2048,3072),'Q3':(3072,4096),'H1':(0,2048),'H2':(2048,4096),'ALL':(0,4096)}


def aggregate_blocks(numerators,denominator,point_state,blocks):
    objects={}
    for name,(start,end) in blocks.items():
        rows=numerators[start:end];R=end-start
        sums=[sum(row[s] for row in rows) for s in range(5)];state=canonical_and_mask(sums)[0]
        means=[Fraction(n,R*denominator) for n in sums]
        maxsum=sum(max(row) for row in rows);regrets=[Fraction(maxsum-n,R*denominator) for n in sums]
        objects[name]=dict(state=state,means_exact=[str(x) for x in means],means=[float(x) for x in means],regrets_exact=[str(x) for x in regrets],
            utility_numerator_sums=[str(n) for n in sums],max_utility_numerator_sum=str(maxsum),mean_denominator=str(R*denominator))
        assert regrets[state]==min(regrets)
    rows=numerators[blocks['ALL'][0]:blocks['ALL'][1]];R=len(rows);state=objects['ALL']['state']
    states,masks=zip(*(canonical_and_mask(row) for row in rows));freq=np.bincount(states,minlength=5)/R
    diffs=[(row[state]-row[point_state])/denominator for row in rows]
    ordered=sorted([Fraction(x) for x in objects['ALL']['means_exact']],reverse=True)
    diagnostics=dict(vote_frequencies=freq.tolist(),p_mode=float(freq.max()),vote_entropy=float(-sum(v*np.log(v) for v in freq if v>0)),
        point_bootstrap_expected_regret_exact=objects['ALL']['regrets_exact'][point_state],BER_bootstrap_expected_regret_exact=objects['ALL']['regrets_exact'][state],
        top_mean_utility_margin_exact=str(ordered[0]-ordered[1]),top_mean_utility_margin=float(ordered[0]-ordered[1]),BER_replica_optimal_fraction=sum(bool(m&(1<<state)) for m in masks)/R,
        BER_minus_point_utility_p10=float(np.quantile(diffs,.1)),BER_minus_point_utility_median=float(np.median(diffs)),BER_minus_point_utility_p90=float(np.quantile(diffs,.9)))
    for name,(start,end) in blocks.items():
        canonical=states[start-blocks['ALL'][0]:end-blocks['ALL'][0]]
        objects[name]['votes']=(np.bincount(canonical,minlength=5)/(end-start)).tolist()
    return objects,diagnostics,np.array(states,dtype=np.uint8),np.array(masks,dtype=np.uint8)


def convergence_gate(choices,banks,contexts):
    h1=np.array(choices['H1']);h2=np.array(choices['H2']);allc=np.array(choices['ALL'])
    agreement=lambda a,b:float(np.mean(a==b))
    global_values=dict(H1_H2=agreement(h1,h2),H1_ALL=agreement(h1,allc),H2_ALL=agreement(h2,allc))
    groups=[dict(bank=b,context=c,agreement=agreement(h1[(np.array(banks)==b)&(np.array(contexts)==c)],h2[(np.array(banks)==b)&(np.array(contexts)==c)])) for b in sorted(set(banks)) for c in sorted(set(contexts))]
    pooled=[dict(context=c,agreement=agreement(h1[np.array(contexts)==c],h2[np.array(contexts)==c])) for c in sorted(set(contexts))]
    passed=global_values['H1_H2']>=.99 and global_values['H1_ALL']>=.995 and global_values['H2_ALL']>=.995 and all(r['agreement']>=.985 for r in groups) and all(r['agreement']>=.99 for r in pooled)
    return dict(passed=passed,global_agreement=global_values,bank_context=groups,pooled_context=pooled)
