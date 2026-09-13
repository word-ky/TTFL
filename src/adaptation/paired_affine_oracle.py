"""Non-deployable T004 clean-paired, label-free affine capacity oracle."""
import torch

EPS=1e-5


def fit_channel_affine(x,z):
    xd,zd=x.double(),z.double()
    mu_x,mu_z=xd.mean((0,2,3)),zd.mean((0,2,3))
    xc,zc=xd-mu_x[None,:,None,None],zd-mu_z[None,:,None,None]
    variance=xc.square().mean((0,2,3))
    covariance=(xc*zc).mean((0,2,3))
    raw=covariance/(variance+EPS)
    a=raw.clamp(-8,8)
    b=mu_z-a*mu_x
    return a.to(x.dtype),b.to(x.dtype),raw


@torch.no_grad()
def paired_clean_oracle(model,clean_support,source_support):
    """Only paired support pixels enter state construction; no query or labels."""
    x,z=source_support,clean_support
    state,diagnostics=[],[]
    for layer,block in enumerate((model.conv1,model.conv2)):
        x,z=block(x),block(z)
        a,b,raw=fit_channel_affine(x,z)
        before=(x.double()-z.double()).square().mean().item()
        gamma=a-1
        x=x*(1+gamma[None,:,None,None])+b[None,:,None,None]
        after=(x.double()-z.double()).square().mean().item()
        assert torch.isfinite(x).all() and torch.isfinite(a).all() and torch.isfinite(b).all()
        state.extend((gamma,b))
        diagnostics.append(dict(layer=layer+1,mse_before=before,mse_after=after,
            restoration_ratio=after/before if before>0 else None,
            gamma_abs_mean=gamma.abs().mean().item(),gamma_abs_max=gamma.abs().max().item(),
            beta_abs_mean=b.abs().mean().item(),beta_abs_max=b.abs().max().item(),
            cap_fraction=(raw.abs()>=8).double().mean().item(),
            negative_scale_fraction=(a<0).double().mean().item(),
            a=a.cpu().tolist(),b=b.cpu().tolist(),nonfinite_count=0))
    return state,diagnostics
