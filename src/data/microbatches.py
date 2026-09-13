"""Only labels from the committed balanced pool construct diagnostic batches."""


def balanced_microbatches(labels):
    groups=[[i for i,y in enumerate(labels) if y==c] for c in range(10)]
    assert len(labels)==80 and all(len(g)==8 for g in groups)
    return [[i for group in groups for i in group[2*b:2*b+2]] for b in range(4)]
