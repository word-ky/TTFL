"""Label-free deterministic image derangement for the T005 target control."""
import torch
from src.data.covariate import sample_seed


def target_derangement(size,client,seed=7):
    offset=0 if size<=1 else 1+sample_seed(seed,client,0,'target_pairing')%(size-1)
    return torch.roll(torch.arange(size),int(offset)),int(offset)
