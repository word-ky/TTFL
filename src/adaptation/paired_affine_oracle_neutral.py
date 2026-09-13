"""T007R paired oracle: ridge prior on zero residual, not zero absolute scale."""
import torch
from src.adaptation.paired_affine_oracle import EPS


def fit_channel_residual(x,z):
    xd,zd=x.double(),z.double()
    r=zd-xd
    mu_x,mu_r=xd.mean((0,2,3)),r.mean((0,2,3))
    xc,rc=xd-mu_x[None,:,None,None],r-mu_r[None,:,None,None]
    variance=xc.square().mean((0,2,3))
    raw_scale=1+(xc*rc).mean((0,2,3))/(variance+EPS)
    gamma=raw_scale.clamp(-8,8)-1
    beta=mu_r-gamma*mu_x
    return gamma.to(x.dtype),beta.to(x.dtype),raw_scale,variance


@torch.no_grad()
def paired_clean_oracle_neutral(model,clean_support,source_support):
    x,z=source_support,clean_support
    state,diagnostics=[],[]
    for layer,block in enumerate((model.conv1,model.conv2)):
        x,z=block(x),block(z)
        gamma,beta,raw,variance=fit_channel_residual(x,z)
        before=(x.double()-z.double()).square().mean().item()
        x=x*(1+gamma[None,:,None,None])+beta[None,:,None,None]
        after=(x.double()-z.double()).square().mean().item()
        assert torch.isfinite(x).all() and torch.isfinite(gamma).all() and torch.isfinite(beta).all()
        state.extend((gamma,beta))
        diagnostics.append(dict(layer=layer+1,mse_before=before,mse_after=after,
            restoration_ratio=after/before if before>0 else None,
            gamma_abs_mean=gamma.abs().mean().item(),gamma_abs_max=gamma.abs().max().item(),
            beta_abs_mean=beta.abs().mean().item(),beta_abs_max=beta.abs().max().item(),
            cap_fraction=(raw.abs()>=8).double().mean().item(),
            negative_scale_fraction=((1+gamma)<0).double().mean().item(),
            minimum_source_variance=variance.min().item(),
            a=(1+gamma).cpu().tolist(),b=beta.cpu().tolist(),nonfinite_count=0))
    return state,diagnostics
