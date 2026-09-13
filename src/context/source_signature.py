"""T008 zero-state source moments and fixed nearest-prototype selection."""
import torch


@torch.no_grad()
def source_moments(model,source):
    moments=[]
    for block in (model.conv1,model.conv2):
        source=block(source)
        x=source.double();mu=x.mean((0,2,3))
        sigma=(x.var((0,2,3),unbiased=False)+1e-6).sqrt()
        moments.append((mu,sigma))
    return moments


@torch.no_grad()
def source_signature(model,source,bank_reference):
    parts=[]
    for (mu,sigma),(mu0,sigma0) in zip(source_moments(model,source),bank_reference):
        parts.extend(((mu-mu0)/(sigma0+1e-6),torch.log((sigma+1e-6)/(sigma0+1e-6))))
    phi=torch.cat(parts)
    assert phi.numel()==192 and torch.isfinite(phi).all()
    return phi


def nearest_prototype(signature,prototypes):
    names=list(prototypes)
    distances={n:float((signature-prototypes[n]).square().sum()) for n in names}
    assert all(torch.isfinite(torch.tensor(v)) for v in distances.values())
    order=sorted(names,key=lambda n:(distances[n],names.index(n)))
    return dict(selected=order[0],second=order[1],distance=distances[order[0]],
        second_distance=distances[order[1]],margin=distances[order[1]]-distances[order[0]],distances=distances)
