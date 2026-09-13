"""One-off arithmetic and provenance check of the completed fixed T001 receipt."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
run = root/'research_log/remote_runs/20260913-113757-ttfl-t001-full'
data = run/'artifacts/t001'
rows = json.loads((data/'records.json').read_text())
baseline = json.loads((data/'baseline.json').read_text())
config = json.loads((data/'config.json').read_text())
splits = json.loads((data/'split_manifest.json').read_text())
pred = np.load(data/'predictions.npz')
y = pred['query_labels']
checkpoint_hash = hashlib.file_digest((data/'global.pt').open('rb'), 'sha256').hexdigest()
assert checkpoint_hash == baseline['checkpoint_sha256']
assert len(rows) == 446 and len(y) == 3000
assert len({(r['method'], r['context'], r['support_seed'], r['lr'], r['steps']) for r in rows}) == 446
errors = []
class_errors = []
for r in rows:
    p = pred[r['prediction_key']]
    errors.append(abs(float(np.mean(p == y)*100)-r['accuracy']))
    class_acc = [float(np.mean(p[y == c] == c)*100) for c in range(10)]
    class_errors += [abs(a-b) for a,b in zip(class_acc, r['per_class_accuracy'])]
    assert r['checkpoint_sha256'] == checkpoint_hash
    assert r['source_tensor_sha256'] == baseline['source_tensor_sha256']
assert max(errors) < 1e-5 and max(class_errors) < 1e-5
query = set(splits['query'])
assert len(query) == 3000 and all(p.startswith('SVHN/test/') for p in query)
for sources in splits['source_train'].values():
    assert len(sources) == 6000 and not query.intersection(sources)
    assert all(not p.startswith('SVHN/') for p in sources)
for seed, episode in splits['supports'].items():
    assert episode['class_counts'] == [20]*10
    for context in ['correct', 'wrong_MNIST', 'wrong_USPS', 'wrong_MNISTM']:
        assert len(set(episode[context])) == 200
        assert not query.intersection(episode[context])
    assert all(p.startswith('SVHN/train/') for p in episode['correct'])
assert baseline['neutral_max_abs_diff'] == 0
assert '[autodl] exit_code=0' in (run/'train.log').read_text()
report = dict(status='PASS', records=len(rows), queries=len(y),
              support_seeds=config['support_seeds'], checkpoint_sha256=checkpoint_hash,
              max_accuracy_recompute_error_pp=max(errors),
              max_class_accuracy_recompute_error_pp=max(class_errors),
              support_query_overlap=0, source_query_overlap=0,
              neutral_max_abs_diff=baseline['neutral_max_abs_diff'],
              scope='Receipt arithmetic, IDs, recorded shared hash and downloaded checkpoint. Runtime frozen-state assertions passed; no additional experiment.')
(root/'results/t001/verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
