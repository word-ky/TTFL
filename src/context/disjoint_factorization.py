"""T013 label-blind halves and exact supervised diagnostic utility factorization."""
import hashlib
from fractions import Fraction

SALTS=('T013-S0','T013-S1','T013-S2','T013-S3')
OFFSETS=(7,13,23,37,41,53,71,89)

def query_halves(original_ids,salt,client):
    order=sorted(range(len(original_ids)),key=lambda i:(hashlib.sha256(f'T013|{salt}|{client}|{original_ids[i]}'.encode()).digest(),i))
    return [order[::2],order[1::2]]

def factor_utilities(train_correct,train_totals):
    """Only training-half integer counts: [client, context, candidate]."""
    n=len(train_totals)
    delta=[[[Fraction(int(v)-int(row[0]),int(train_totals[i])) for v in row] for row in client] for i,client in enumerate(train_correct)]
    persistent=[r[0] for r in delta]
    residual=[[[delta[i][c][s]-persistent[i][s] for s in range(5)] for c in range(5)] for i in range(n)]
    total=[[sum((residual[i][c][s] for i in range(n)),Fraction()) for s in range(5)] for c in range(5)]
    context=[[[ (total[c][s]-residual[i][c][s])/(n-1) for s in range(5)] for c in range(5)] for i in range(n)]
    return persistent,context

def exact_argmax(values):
    top=max(values)
    return [i for i,v in enumerate(values) if v==top]
