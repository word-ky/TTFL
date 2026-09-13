"""T003: label-free, optimizer-free sequential channel-moment writing."""
import torch

EPS = 1e-5
SCALE_BOUNDS = (.25, 4.)


def moments(h):
    z = h.double()
    mu = z.mean((0, 2, 3))
    variance = (z.square().mean((0, 2, 3)) - mu.square()).clamp_min(0)
    return mu, (variance + EPS).sqrt()


@torch.no_grad()
def estimate_global_reference_moments(model, image_batches):
    """All batches contain images only. Both layers use the unadapted model."""
    sums, squares, counts = [None, None], [None, None], [0, 0]
    samples = 0
    for x in image_batches:
        samples += len(x)
        for layer, block in enumerate((model.conv1, model.conv2)):
            x = block(x)
            z = x.double()
            first, second = z.sum((0, 2, 3)), z.square().sum((0, 2, 3))
            sums[layer] = first if sums[layer] is None else sums[layer] + first
            squares[layer] = second if squares[layer] is None else squares[layer] + second
            counts[layer] += x.shape[0]*x.shape[2]*x.shape[3]
    layers = []
    for first, second, count in zip(sums, squares, counts):
        mu = first/count
        variance = (second/count - mu.square()).clamp_min(0)
        sigma = (variance + EPS).sqrt()
        assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()
        layers.append(dict(mu=mu.cpu(), sigma=sigma.cpu(), activation_count=count))
    return dict(layers=layers, samples=samples, eps=EPS)


@torch.no_grad()
def write_affine_from_support_moments(model, images, reference):
    """No labels accepted; state is detached and written analytically in order."""
    state, diagnostics = [], []
    h = images
    for layer, block in enumerate((model.conv1, model.conv2)):
        h = block(h)
        mu, sigma = moments(h)
        ref = reference['layers'][layer]
        mu_g, sigma_g = ref['mu'].to(h.device), ref['sigma'].to(h.device)
        raw_scale = sigma_g/(sigma+EPS)
        scale = raw_scale.clamp(*SCALE_BOUNDS)
        gamma = (scale-1).to(h.dtype)
        beta = (mu_g-scale*mu).to(h.dtype)
        h = h*(1+gamma[None,:,None,None])+beta[None,:,None,None]
        after_mu, after_sigma = moments(h)
        before = ((mu-mu_g).abs()+(sigma-sigma_g).abs()).mean().item()
        after = ((after_mu-mu_g).abs()+(after_sigma-sigma_g).abs()).mean().item()
        assert torch.isfinite(h).all() and torch.isfinite(after_sigma).all()
        # Epsilon-regularized std and float32 application can differ at tiny scale.
        assert after <= before + 1e-5, (layer, before, after)
        state.extend((gamma, beta))
        diagnostics.append(dict(layer=layer+1,channels=len(gamma),
            gamma_abs_mean=gamma.abs().mean().item(),gamma_abs_max=gamma.abs().max().item(),
            beta_abs_mean=beta.abs().mean().item(),beta_abs_max=beta.abs().max().item(),
            clamp_low_fraction=(raw_scale<=SCALE_BOUNDS[0]).double().mean().item(),
            clamp_high_fraction=(raw_scale>=SCALE_BOUNDS[1]).double().mean().item(),
            clamp_fraction=((raw_scale<=SCALE_BOUNDS[0])|(raw_scale>=SCALE_BOUNDS[1])).double().mean().item(),
            moment_mismatch_before=before,moment_mismatch_after=after,
            support_mu=mu.cpu().tolist(),support_sigma=sigma.cpu().tolist(),
            calibrated_mu=after_mu.cpu().tolist(),calibrated_sigma=after_sigma.cpu().tolist()))
    return state, diagnostics
