"""Fixed-checkpoint supervised read-operator screen. No query gradients."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from src.models import DigitCNN
from src.data.digits import load_domain, indices, balanced_indices, normalized
from src.eval.common import evaluate, state_hash
from src.adaptation.affine import adapt, bn_from_support, new_state, state_norms


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--mnistm-train', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/t001.json')
    p.add_argument('--commit', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.smoke:
        cfg.update(source_samples_per_domain=100, epochs=1, query_count=100,
                   support_per_class=2, support_seeds=[11], steps=[1], lrs=[.01], full_steps=[1])
    cfg.update(commit=args.commit, smoke=args.smoke, data_root=args.data_root,
               mnistm_train=args.mnistm_train)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg['device'])
    assert device.type == 'cuda' and torch.cuda.is_available()
    save_json(out / 'config.json', cfg)
    save_json(out / 'environment.json', {'python': platform.python_version(),
              'torch': torch.__version__, 'host': platform.node(),
              'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
              'visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
              'start_time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    train_x, train_y, pools, split_manifest = [], [], {}, {'source_train': {}}
    for k, domain in enumerate(cfg['sources']):
        raw, labels, ids = load_domain(args.data_root, domain, image_root=args.mnistm_train)
        selected = indices(len(labels), cfg['source_samples_per_domain'], cfg['seed']+k)
        train_x.append(normalized(raw[selected]))
        train_y.append(labels[selected])
        split_manifest['source_train'][domain] = [ids[i] for i in selected.tolist()]
        pools[domain] = (raw, labels, ids)
        print('source_loaded', domain, len(selected), flush=True)
    xtrain, ytrain = torch.cat(train_x).to(device), torch.cat(train_y).to(device)
    source_prior = (torch.bincount(ytrain, minlength=10).float()+1)/(len(ytrain)+10)
    model = DigitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train_lr'])
    training = []
    for epoch in range(cfg['epochs']):
        model.train()
        permutation = torch.randperm(len(ytrain), device=device)
        total_loss, correct = 0., 0
        for ids in permutation.split(cfg['batch_size']):
            opt.zero_grad()
            logits = model(xtrain[ids])
            loss = F.cross_entropy(logits, ytrain[ids])
            loss.backward()
            opt.step()
            total_loss += loss.item()*len(ids)
            correct += (logits.argmax(1) == ytrain[ids]).sum().item()
        row = {'epoch': epoch+1, 'loss': total_loss/len(ytrain), 'accuracy': correct/len(ytrain)*100}
        training.append(row)
        print('train', row, flush=True)
    model.eval()
    checkpoint = out / 'global.pt'
    torch.save(model.state_dict(), checkpoint)
    checkpoint_sha = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
    tensor_sha = state_hash(model)
    loaded = DigitCNN().to(device).eval()
    loaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    assert state_hash(loaded) == tensor_sha
    assert torch.equal(model(xtrain[:32]), loaded(xtrain[:32]))
    model = loaded
    save_json(out / 'training.json', training)
    del xtrain, ytrain, train_x, train_y

    raw, labels, ids = load_domain(args.data_root, 'SVHN', train=False)
    selected = indices(len(labels), cfg['query_count'], cfg['query_seed'])
    qx, qy = normalized(raw[selected]).to(device), labels[selected].to(device)
    query_ids = [ids[i] for i in selected.tolist()]
    split_manifest['query'] = query_ids
    pools['SVHN'] = load_domain(args.data_root, 'SVHN', train=True)
    neutral_diff = (model(qx[:32]) - model(qx[:32], new_state(model))).abs().max().item()
    assert neutral_diff < 1e-6
    baseline, baseline_pred = evaluate(model, qx, qy)
    predictions = {'query_labels': qy.cpu().numpy(), 'baseline': baseline_pred.numpy()}
    rows = []
    split_manifest['supports'] = {}

    def record(m, state, method, context, support_seed, sx, sy, lr=0., steps=0, losses=None, bias=None):
        # The shared source remains unchanged across all methods/contexts.
        assert state_hash(model) == tensor_sha
        result, pred = evaluate(m, qx, qy, state, bias)
        row = dict(method=method, context=context, support_seed=support_seed, lr=lr, steps=steps,
                   checkpoint_sha256=checkpoint_sha, source_tensor_sha256=tensor_sha,
                   baseline_accuracy=baseline['accuracy'], query_loss_before=baseline['loss'],
                   gain_pp=result['accuracy']-baseline['accuracy'], **result, **state_norms(state))
        if sx is not None:
            row['support_loss_before'] = evaluate(model, sx, sy)[0]['loss']
            row['support_loss_after'] = evaluate(m, sx, sy, state, bias)[0]['loss']
        row['support_loss_trajectory'] = losses
        row['prediction_key'] = f'r{len(rows):04d}'
        predictions[row['prediction_key']] = pred.numpy()
        rows.append(row)

    record(model, None, 'none', 'none', 0, None, None)
    for seed in cfg['support_seeds']:
        supports = {}
        manifest = {}
        for domain in ['SVHN'] + cfg['sources']:
            raw, labels, ids = pools[domain]
            selected = balanced_indices(labels, cfg['support_per_class'], seed)
            sid = [ids[i] for i in selected.tolist()]
            assert not set(sid).intersection(query_ids)
            context = 'correct' if domain == 'SVHN' else f'wrong_{domain}'
            supports[context] = normalized(raw[selected]).to(device), labels[selected].to(device)
            manifest[context] = sid
        sx, sy = supports['correct']
        gen = torch.Generator().manual_seed(seed+10000)
        shuffle = torch.randperm(len(sy), generator=gen)
        supports['shuffled'] = sx, sy[shuffle.to(device)]
        noise = torch.rand(sx.shape, generator=gen).to(device)*2-1
        supports['noise'] = noise, sy
        manifest['shuffled_label_permutation'] = shuffle.tolist()
        manifest['class_counts'] = torch.bincount(sy, minlength=10).tolist()
        split_manifest['supports'][str(seed)] = manifest
        random_state = new_state(model, cfg['random_state_std'], seed)
        record(model, random_state, 'random_no_write', 'random', seed, sx, sy)
        for context, (sx, sy) in supports.items():
            bn = bn_from_support(model, sx)
            record(bn, None, 'bn', context, seed, sx, sy)
            prior = (torch.bincount(sy, minlength=10).float()+1)/(len(sy)+10)
            bias = prior.log()-source_prior.log()
            record(model, None, 'prior', context, seed, sx, sy, bias=bias)
            for lr in cfg['lrs']:
                for steps in cfg['steps']:
                    m, state, losses = adapt(model, sx, sy, steps, lr)
                    assert state_hash(m) == tensor_sha
                    record(m, state, 'affine', context, seed, sx, sy, lr, steps, losses)
            print('context_done', seed, context, flush=True)
        sx, sy = supports['correct']
        for steps in cfg['full_steps']:
            m, state, losses = adapt(model, sx, sy, steps, cfg['full_lr'], full=True)
            record(m, state, 'full', 'correct', seed, sx, sy, cfg['full_lr'], steps, losses)
        save_json(out / 'records.json', rows)
        save_json(out / 'split_manifest.json', split_manifest)
    assert state_hash(model) == tensor_sha
    save_json(out / 'baseline.json', dict(**baseline, checkpoint_sha256=checkpoint_sha,
              source_tensor_sha256=tensor_sha, neutral_max_abs_diff=neutral_diff,
              fast_parameter_count=sum(p.numel() for p in new_state(model))))
    np.savez_compressed(out / 'predictions.npz', **predictions)
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with (out / 'records.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(flat)
    print('DONE', len(rows), 'records', 'baseline', baseline['accuracy'], flush=True)


if __name__ == '__main__':
    main()
