"""T017 privileged empirical matched-channel diagnostic, fixed sampling only."""
import hashlib
from fractions import Fraction
import numpy as np


def emission_pools(target,clients,load_client):
    grouped=[[] for _ in range(10)]
    used=[]
    for j in clients:
        if j==target:continue
        probabilities,labels=load_client(j)
        for k in range(10):grouped[k].append(probabilities[np.asarray(labels)==k])
        used.append(j)
    pools=[np.concatenate(g,axis=0) for g in grouped]
    assert all(len(g)>0 for g in pools)
    return pools,used


def replica_seed(client,bank,context,K,replica):
    return int.from_bytes(hashlib.sha256(f'T017|{client}|{bank}|{context}|{K}|{replica}'.encode()).digest(),'big')


def bootstrap_observation(pools,counts,K,seed):
    counts=np.asarray(counts,dtype=np.int64)
    assert counts.sum()==20 and K in (20,40,80,160)
    expanded=counts*(K//20);rng=np.random.default_rng(seed)
    samples=[pool[rng.integers(len(pool),size=int(n))] for pool,n in zip(pools,expanded) if n]
    return np.concatenate(samples,axis=0).mean(0),expanded


def exact_utilities(pi,template):
    weights=[x if isinstance(x,Fraction) else Fraction(float(x)) for x in pi]
    values=[sum((weights[k]*template[k][s] for k in range(10)),Fraction()) for s in range(5)]
    arg=[s for s,v in enumerate(values) if v==max(values)]
    return values,arg


def mismatch_residual(actual,expected,bootstrap):
    actual=float(np.abs(np.asarray(actual)-expected).sum())
    distances=np.abs(np.asarray(bootstrap)-expected).sum(-1)
    return dict(D_actual=actual,p50=float(np.quantile(distances,.5)),p90=float(np.quantile(distances,.9)),p95=float(np.quantile(distances,.95)),maximum=float(distances.max()),percentile=float(np.mean(distances<=actual)),above_p95=bool(actual>np.quantile(distances,.95)))


def selected_half_count(correct,selected):
    return np.asarray(correct)[selected].copy()
