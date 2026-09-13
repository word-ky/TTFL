"""T007 diagnostic pools; labels are confined to selection and pairing."""
import hashlib


def calibration_pools(candidates):
    pools, used = {}, set()
    for pool in ('A', 'B'):
        selected = []
        for label in range(10):
            eligible = [r for r in candidates if r['class'] == label and r['original_id'] not in used]
            eligible.sort(key=lambda r: (hashlib.sha256(
                f"t007:{pool}:{label}:{r['client_id']}:{r['original_id']}".encode()).hexdigest(),
                r['client_id'], r['original_id']))
            chosen, clients = [], set()
            for r in eligible:
                if r['client_id'] not in clients:
                    chosen.append(r)
                    clients.add(r['client_id'])
                if len(chosen) == 8:
                    break
            chosen_ids = {r['original_id'] for r in chosen}
            for r in eligible:
                if len(chosen) == 8:
                    break
                if r['original_id'] not in chosen_ids:
                    chosen.append(r)
                    chosen_ids.add(r['original_id'])
            if len(chosen) != 8:
                raise ValueError(f'Pool {pool} class {label}: only {len(chosen)} eligible samples')
            selected.extend(chosen)
            used.update(r['original_id'] for r in chosen)
        pools[pool] = selected
    return pools


def balanced_pairings(labels):
    indices = {c: [i for i, y in enumerate(labels) if y == c] for c in range(10)}
    assert all(len(v) == 8 for v in indices.values())
    same, cross = list(range(80)), list(range(80))
    for c, group in indices.items():
        for j, i in enumerate(group):
            same[i] = group[(j+1) % 8]
            cross[i] = indices[(c+1) % 10][j]
    return dict(same_class_derangement=same, cross_class_derangement=cross)
