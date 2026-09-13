"""T009 support ordering from split IDs only; no label input exists."""
import hashlib
import random


def natural_support_manifest(clients,calibration_ids):
    excluded=set(calibration_ids)
    query={sid for r in clients.values() for sid in r['test']}
    eligible={}
    for cid in range(100):
        ids=clients[str(cid)]['train']
        rows=[dict(train_index=i,original_id=int(sid)) for i,sid in enumerate(ids) if sid not in excluded and sid not in query]
        seed=int.from_bytes(hashlib.sha256(f'7:{cid}:T009-natural-support'.encode()).digest(),'big')
        random.Random(seed).shuffle(rows)
        eligible[str(cid)]=rows
    counts={cid:len(rs) for cid,rs in eligible.items()}
    k=next((k for k in (64,32,20,16) if min(counts.values())>=k),None)
    return dict(seed=7,ordering='random.Random(SHA256(7:client:T009-natural-support)).shuffle(train-order eligible IDs)',
        chosen_K=k,eligible_counts=counts,secondary_K=20 if k is not None and k>=32 else None,
        clients={cid:dict(client=int(cid),eligible_count=len(rs),selected=rs[:k] if k is not None else []) for cid,rs in eligible.items()})
