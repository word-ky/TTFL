"""T016 fixed leave-one-client-out observation channels; no fitted hyperparameters."""
import numpy as np


def probabilities(logits):
    x=np.asarray(logits,dtype=np.float64)
    e=np.exp(x-x.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def project_simplex(z):
    z=np.asarray(z,dtype=np.float64)
    u=np.sort(z)[::-1]
    thresholds=(np.cumsum(u)-1)/np.arange(1,len(z)+1)
    rho=np.flatnonzero(u>thresholds)[-1]
    return np.maximum(z-thresholds[rho],0.)


def channel_from_other_clients(target,clients,load_calibration):
    """The loader is never called on target. Returns unrounded sufficient statistics."""
    hard=np.zeros((10,10),dtype=np.int64)
    soft=np.zeros((10,10),dtype=np.float64)
    denominator=np.zeros(10,dtype=np.int64)
    used=[]
    for client in clients:
        if client==target:
            continue
        observations,labels=load_calibration(client)
        observations=np.asarray(observations)
        labels=np.asarray(labels)
        counts=np.bincount(labels,minlength=10)
        denominator+=counts
        if observations.ndim==1:
            np.add.at(hard,(observations,labels),1)
        else:
            for k in range(10):
                soft[:,k]+=observations[labels==k].sum(axis=0)
        used.append(client)
    assert np.all(denominator>0)
    return hard,soft,denominator,used


def normalized_channel(numerator,denominator):
    C=np.asarray(numerator,dtype=np.float64)/denominator[None,:]
    assert np.allclose(C.sum(axis=0),1.,rtol=0.,atol=1e-12)
    return C


def estimate(C,q):
    # Intentionally use NumPy's default pinv rcond, with no alternatives or tuning.
    z=np.linalg.pinv(C)@q
    return project_simplex(z),z


def source_estimate(channels,source_context,q):
    # True context and support labels are not accepted by this API.
    return estimate(channels[source_context],q)
