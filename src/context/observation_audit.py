"""Privileged T019 residual decomposition and fixed-count empirical null."""
import hashlib
import numpy as np


def class_seed(client, bank, context, replica, label):
    return int.from_bytes(hashlib.sha256(f'T019|{client}|{bank}|{context}|{replica}|{label}'.encode()).digest(),'big')


def draw_source_ids(client,bank,context,replica,counts,flat_labels):
    """Pool order is source client then frozen support position; replacement as T017."""
    flat_labels=np.asarray(flat_labels)
    ids=np.arange(len(flat_labels));parts=[]
    for y,n in enumerate(counts):
        if n:
            pool=ids[(ids//20!=client)&(flat_labels==y)]
            rng=np.random.default_rng(class_seed(client,bank,context,replica,y))
            parts.append(pool[rng.integers(len(pool),size=int(n))])
    return np.concatenate(parts)


def decompose(prob,labels,matrix):
    labels=np.asarray(labels);counts=np.bincount(labels,minlength=10)
    means=np.full((10,10),np.nan);difference=np.full_like(means,np.nan);weighted=np.zeros_like(means)
    for y in np.flatnonzero(counts):
        means[y]=prob[labels==y].mean(0);difference[y]=means[y]-matrix[:,y]
        weighted[y]=(counts[y]/len(labels))*difference[y]
    q=prob.mean(0);residual=q-matrix@(counts/len(labels))
    assert np.max(np.abs(residual-weighted.sum(0)))<=1e-12
    return counts,means,difference,weighted,q,residual


def null_comparison(actual,null):
    null=np.asarray(null)
    return dict(actual=float(actual),null_median=float(np.median(null)),null_p95=float(np.quantile(null,.95)),
        percentile=float(np.mean(null<=actual)),above_p95=bool(actual>np.quantile(null,.95)))


def single_class_repair(q,weighted,y):
    return np.asarray(q)-weighted[y]
