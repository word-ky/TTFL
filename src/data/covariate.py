"""Fixed T002 transforms on saved PFLlib [-1,1] image tensors."""
import hashlib
import torch
from torchvision.transforms.functional import gaussian_blur

TARGETS = ('brightness_dark', 'contrast_low', 'gaussian_noise', 'gaussian_blur')
WRONG_ALT = dict(zip(TARGETS, ('gaussian_noise', 'gaussian_blur',
                             'brightness_dark', 'contrast_low')))


def sample_seed(seed, client, sample, corruption):
    key = f'{seed}:{client}:{sample}:{corruption}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)


def corrupt(x, name, ids, client, seed=7):
    if name == 'clean':
        return x.clone()
    z = (x + 1) / 2
    if name == 'brightness_dark':
        z = z * .45
    elif name == 'contrast_low':
        mean = z.mean((-2, -1), keepdim=True)
        z = (z - mean) * .35 + mean
    elif name == 'gaussian_blur':
        z = gaussian_blur(z, [5, 5], [1.5, 1.5])
    elif name in ('gaussian_noise', 'noise_image'):
        # CPU RNG makes each image independent of batching/order and CUDA RNG state.
        noise = torch.stack([
            (torch.randn if name == 'gaussian_noise' else torch.rand)(
                x.shape[1:], generator=torch.Generator().manual_seed(
                    sample_seed(seed, client, int(sid), name))) for sid in ids
        ]).to(x.device)
        z = (z + .15 * noise).clamp(0, 1) if name == 'gaussian_noise' else noise
    else:
        raise ValueError(name)
    return z * 2 - 1
