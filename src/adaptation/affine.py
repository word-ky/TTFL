import copy
import torch
from torch import nn
import torch.nn.functional as F


def new_state(model, random_std=0., seed=0):
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    return [nn.Parameter(torch.randn(c, device=device, generator=gen)*random_std)
            for c in model.channels for _ in range(2)]


def adapt(model, x, y, steps, lr, full=False):
    """Fresh model/state per support. BN always eval in gradient adaptations."""
    m = copy.deepcopy(model).eval()
    m.requires_grad_(full)
    state = None if full else new_state(m)
    opt = torch.optim.SGD(m.parameters() if full else state, lr=lr)
    losses = [F.cross_entropy(m(x, state), y).item()]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(m(x, state), y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            losses.append(F.cross_entropy(m(x, state), y).item())
    return m, state, losses


@torch.no_grad()
def bn_from_support(model, x):
    """One full support batch replaces all BN running moments (no gradients)."""
    m = copy.deepcopy(model).eval()
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.train()
            layer.momentum = 1.
    m(x)
    return m.eval()


def state_norms(state):
    if state is None:
        return {'gamma_norm': 0., 'beta_norm': 0.}
    return {name: torch.cat([p.detach().flatten() for p in state[offset::2]]).norm().item()
            for offset, name in enumerate(('gamma_norm', 'beta_norm'))}
