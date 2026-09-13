"""T010 post-state source moments along the actual sequential affine path."""
import torch


@torch.no_grad()
def post_state_signature(model,source,state,bank_reference):
    parts=[]
    for layer,block in enumerate((model.conv1,model.conv2)):
        source=block(source)
        gamma,beta=state[2*layer:2*layer+2]
        source=source*(1+gamma[None,:,None,None])+beta[None,:,None,None]
        x=source.double();mu=x.mean((0,2,3));sigma=(x.var((0,2,3),unbiased=False)+1e-6).sqrt()
        mu0,sigma0=bank_reference[layer]
        parts.extend(((mu-mu0)/(sigma0+1e-6),torch.log((sigma+1e-6)/(sigma0+1e-6))))
    psi=torch.cat(parts)
    assert psi.numel()==192 and torch.isfinite(psi).all()
    return psi


def choose_state(scores):
    order=('clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur')
    ranked=sorted(order,key=lambda c:(scores[c],order.index(c)))
    return dict(selected=ranked[0],second=ranked[1],best_score=scores[ranked[0]],second_score=scores[ranked[1]],
        score_margin=scores[ranked[1]]-scores[ranked[0]],zero_improvement=scores['clean']-scores[ranked[0]],ranking=ranked)
