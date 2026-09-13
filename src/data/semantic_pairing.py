"""T006 diagnostic controls: labels select pairings, never fit model state."""
import torch
from src.data.covariate import sample_seed


def random_offsets(size,client,seed=7):
    available=list(range(1,size))
    chosen=[]
    for k in range(min(8,len(available))):
        index=sample_seed(seed,client,k,'t006_random_pairing')%len(available)
        chosen.append(available.pop(index))
    return chosen


def semantic_pairings(support_labels,client,seed=7):
    n=len(support_labels)
    scores={offset:int((support_labels==support_labels.roll(offset)).sum())
            for offset in range(1,n)}
    low=min(scores,key=lambda offset:(scores[offset],offset))
    high=min(scores,key=lambda offset:(-scores[offset],offset))
    choices=[('low_semantic_derangement',low),('high_semantic_derangement',high)]
    choices += [(f'random_derangement_{k}',offset) for k,offset in enumerate(random_offsets(n,client,seed))]
    return [dict(context=name,offset=offset,permutation=torch.arange(n).roll(offset).tolist(),
                 same_label_count=scores[offset],same_label_fraction=scores[offset]/n)
            for name,offset in choices]
