"""Disposable label-free rotation probes with a fixed numerical SVD cutoff."""
import hashlib
import numpy as np

def folds(client,ids):
    order=sorted(range(len(ids)),key=lambda k:hashlib.sha256(f'T023|fold|{client}|{ids[k]}'.encode()).digest())
    return np.array(order[:10]),np.array(order[10:])

def rotation_labels():return np.tile(np.eye(4),(20,1)).reshape(20,4,4)

def scramble_labels(client,ids,replica):
    values=[]
    for sid in ids:
        seed=int.from_bytes(hashlib.sha256(f'T023|scramble|{client}|{sid}|{replica}'.encode()).digest(),'big')
        values.append(np.eye(4)[np.random.Generator(np.random.PCG64(seed)).permutation(4)])
    return np.stack(values)

def prepare(features,split):
    x=np.concatenate([features.astype(np.float64),np.ones(features.shape[:2]+(1,))],axis=2)
    prepared=[]
    for train,val in (split,split[::-1]):
        a=x[train].reshape(40,-1);b=x[val].reshape(40,-1)
        u,s,vt=np.linalg.svd(a,full_matrices=False);keep=s>1e-12*s[0]
        pinv=(vt[keep].T/s[keep])@u[:,keep].T
        prepared.append((pinv,b,train,val,int(keep.sum())))
    return prepared

def evaluate(prepared,labels):
    errors=[];ranks=[]
    for pinv,b,train,val,rank in prepared:
        w=pinv@labels[train].reshape(40,4);pred=b@w
        error=float(np.mean((pred-labels[val].reshape(40,4))**2))
        assert np.isfinite(w).all() and np.isfinite(error)
        errors.append(error);ranks.append(rank)
    return float(np.mean(errors)),errors,ranks

def independent_score(features,split,labels):
    errors=[];ranks=[]
    for train,val in (split,split[::-1]):
        a=np.column_stack((features[train].reshape(40,-1).astype(np.float64),np.ones(40)))
        b=np.column_stack((features[val].reshape(40,-1).astype(np.float64),np.ones(40)))
        w,_,rank,_=np.linalg.lstsq(a,labels[train].reshape(40,4),rcond=1e-12)
        assert np.isfinite(w).all()
        errors.append(float(np.mean((b@w-labels[val].reshape(40,4))**2)));ranks.append(int(rank))
    return float(np.mean(errors)),errors,ranks
