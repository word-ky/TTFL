from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def indices(n, count, seed):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:count]


def balanced_indices(labels, per_class, seed):
    gen = torch.Generator().manual_seed(seed)
    groups = []
    for c in range(10):
        ids = (labels == c).nonzero().flatten()
        assert len(ids) >= per_class
        groups.append(ids[torch.randperm(len(ids), generator=gen)[:per_class]])
    return torch.cat(groups)


def normalized(x):
    x = x.float() / 255.
    if x.ndim == 3:
        x = x[:, None]
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False) * 2 - 1


def load_domain(root, domain, train=True, image_root=None):
    from torchvision.datasets import MNIST, USPS, SVHN, ImageFolder
    root = Path(root)
    if domain == 'MNISTM':
        from PIL import Image
        ds = ImageFolder(image_root)
        x = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')).copy()
                                      for p, _ in ds.samples])).permute(0, 3, 1, 2)
        y = torch.tensor(ds.targets)
        ids = [str(Path(p).relative_to(root)) for p, _ in ds.samples]
    elif domain == 'SVHN':
        ds = SVHN(str(root / domain), split='train' if train else 'test', download=False)
        x, y = torch.from_numpy(ds.data), torch.tensor(ds.labels)
        ids = [f'{domain}/{"train" if train else "test"}/{i}' for i in range(len(y))]
    else:
        cls = {'MNIST': MNIST, 'USPS': USPS}[domain]
        ds = cls(str(root / domain), train=train, download=False)
        x, y = torch.as_tensor(np.asarray(ds.data).copy()), torch.tensor(ds.targets)
        ids = [f'{domain}/{"train" if train else "test"}/{i}' for i in range(len(y))]
    return x, y.long(), ids
