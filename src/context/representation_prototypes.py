"""T021 raw first-moment prototypes with explicit target exclusion."""
import hashlib
import numpy as np


def excluded_prototype(features,labels,target):
    owners=[j for j in range(len(features)) if j!=target]
    x=np.concatenate([features[j] for j in owners]).astype(np.float64)
    y=np.concatenate([labels[j] for j in owners])
    counts=np.bincount(y,minlength=10);assert np.all(counts>0)
    return np.stack([x[y==k].mean(0) for k in range(10)],axis=1),counts,owners


def matched_mean(features,labels,target,counts,representation,bank,context,replica):
    pools,ids=class_pools(features,labels,target)
    return sample_class_pools(pools,ids,target,counts,representation,bank,context,replica)


def class_pools(features,labels,target):
    owners=[j for j in range(len(features)) if j!=target]
    x=np.concatenate([features[j] for j in owners]).astype(np.float64)
    y=np.concatenate([labels[j] for j in owners])
    ids=np.concatenate([np.arange(20)+20*j for j in owners])
    return [x[y==k] for k in range(10)],[ids[y==k] for k in range(10)]


def sample_class_pools(pools,ids,target,counts,representation,bank,context,replica):
    seed=int.from_bytes(hashlib.sha256(f'T021|matched|{representation}|{target}|{bank}|{context}|{replica}'.encode()).digest(),'big')
    rng=np.random.Generator(np.random.PCG64(seed));draws=[];chosen=[]
    for k,n in enumerate(counts):
        take=rng.integers(0,len(pools[k]),size=int(n))
        draws.append(pools[k][take]);chosen.extend(ids[k][take])
    return np.concatenate(draws).mean(0),np.array(chosen,dtype=np.int32)
