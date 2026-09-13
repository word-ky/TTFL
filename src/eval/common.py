import hashlib
import torch
import torch.nn.functional as F


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, x, y, state=None, bias=None, batch_size=256, num_classes=10):
    model.eval()
    logits = torch.cat([model(b, state) for b in x.split(batch_size)])
    if bias is not None:
        logits = logits + bias
    pred = logits.argmax(1)
    classes = [(pred[y == c] == c).float().mean().item() * 100
               if (y == c).any() else None for c in range(num_classes)]
    return {'accuracy': (pred == y).float().mean().item() * 100,
            'loss': F.cross_entropy(logits, y).item(),
            'worst_class_accuracy': min(a for a in classes if a is not None),
            'per_class_accuracy': classes}, pred.cpu()
