"""Summarize completed PFLlib runs and check their recorded predictions/IDs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root,out=Path(a.input),Path(a.output)
out.mkdir(parents=True,exist_ok=True)
results,verification=[],[]
for dataset in ('MNIST','Cifar10','Cifar100','TinyImagenet'):
    folders=[p for p in root.glob(f'*/{dataset}') if p.parent.name!='data']
    assert len(folders)==1,(dataset,folders)
    folder=folders[0]
    cfg=json.loads((folder/'config.json').read_text())
    baseline=json.loads((folder/'baseline.json').read_text())
    rows=json.loads((folder/'context_records.json').read_text())
    summary=json.loads((folder/'context_summary.json').read_text())
    pred=np.load(folder/'context_predictions.npz')
    splits=json.loads((root/'data'/dataset/'split_manifest.json').read_text())
    supports=json.loads((folder/'support_indices.json').read_text())
    assert baseline['rounds']==100 and len(baseline['selection_history'])==100
    assert all(len(set(s))==10 for s in baseline['selection_history'])
    assert cfg['join_ratio']==.1 and cfg['num_clients']==100 and not cfg['random_join_ratio']
    assert baseline['model_device'].startswith('cuda')
    assert len(rows)==1000
    hashes={r['checkpoint_sha256'] for r in rows}
    assert len(hashes)==1
    errors=[]
    all_ids=[]
    for cid in range(100):
        split=splits['clients'][str(cid)]
        assert not set(split['train']) & set(split['test'])
        all_ids+=split['train']+split['test']
        support=supports[str(cid)]
        sx=[split['train'][i] for i in support['correct_train_indices']]
        wx=[splits['clients'][str(support['wrong_client'])]['train'][i] for i in support['wrong_train_indices']]
        assert not set(sx+wx) & set(split['test'])
    assert len(all_ids)==len(set(all_ids))==splits['total']
    for r in rows:
        y=pred[f"c{r['client']}_labels"]
        errors.append(abs(float(np.mean(pred[r['prediction_key']]==y)*100)-r['accuracy']))
        assert r['neutral_max_abs_diff']<1e-6
    assert max(errors)<1e-5
    means={(r['method'],r['context']):r for r in summary['rows']}
    assert abs(means[('none','none')]['accuracy']-baseline['final_accuracy'])<1e-5
    correct=means[('affine','correct')]['accuracy']
    result=dict(dataset=dataset,baseline=baseline['final_accuracy'],affine_correct=correct,
        affine_gain=correct-baseline['final_accuracy'],affine_wrong=means[('affine','wrong')]['accuracy'],
        specificity_pp=correct-means[('affine','wrong')]['accuracy'],
        affine_shuffled=means[('affine','shuffled')]['accuracy'],affine_noise=means[('affine','noise')]['accuracy'],
        prior_correct=means[('prior','correct')]['accuracy'],random_no_write=means[('random_no_write','random')]['accuracy'],
        train_eval_seconds=baseline['elapsed_seconds'],peak_gpu_gib=baseline['cuda_peak_bytes']/1024**3,
        commit=cfg['commit'],run_id=folder.parent.name)
    results.append(result)
    verification.append(dict(dataset=dataset,status='PASS',records=len(rows),rounds=100,
        participants=10,source_examples=len(all_ids),max_accuracy_error_pp=max(errors),
        support_query_overlap=0,checkpoint_sha256=next(iter(hashes))))
(out/'summary.json').write_text(json.dumps(results,indent=2))
(out/'verification.json').write_text(json.dumps(verification,indent=2))
with (out/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(results[0])); w.writeheader(); w.writerows(results)
lines=['# PFLlib 100 clients / 10 participants', '',
       'One seed; Dirichlet alpha=0.1; 100 rounds; one local epoch; upstream CNN/SGD. Static-client supervised context diagnostic, not dynamic-context evidence.', '',
       '| Dataset | FedAvg | Affine correct | Gain pp | Wrong client | Correct-wrong pp | Shuffled | Prior correct |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in results:
    lines.append(f"| {r['dataset']} | {r['baseline']:.2f} | {r['affine_correct']:.2f} | {r['affine_gain']:+.2f} | {r['affine_wrong']:.2f} | {r['specificity_pp']:+.2f} | {r['affine_shuffled']:.2f} | {r['prior_correct']:.2f} |")
lines+=['','Accuracy is sample-weighted across all100clients; all models run on CUDA. BN N/A because upstream CNN has no BN. Prior uses Laplace1 support counts against uniform classes. Standard PFLlib merges original labeled splits and repartitions75/25, so these are not official held-out image benchmark accuracies.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(results,indent=2))
