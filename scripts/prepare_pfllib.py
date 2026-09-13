"""PFLlib's partition/split semantics, with per-client materialization."""
import argparse
import importlib.util
import json
import random
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder

ROOT = Path(__file__).resolve().parents[1]


def partition_module():
    spec = importlib.util.spec_from_file_location('pfl_partition', ROOT/'third_party/PFLlib/dataset/utils/dataset_utils.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--mnist-root', required=True)
    a = p.parse_args()
    cfg = json.loads((ROOT/'configs/pfllib_100c.json').read_text())
    root = Path(a.root)
    raw = root/'raw'
    dest = root/'dataset'/a.dataset
    for split in ('train', 'test'):
        (dest/split).mkdir(parents=True, exist_ok=True)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    if a.dataset == 'MNIST':
        sets = [MNIST(a.mnist_root, train=b) for b in (True, False)]
        images = np.concatenate([np.asarray(d.data)[:,None] for d in sets])
        labels = np.concatenate([np.asarray(d.targets) for d in sets])
        count_train = len(sets[0])
    elif a.dataset in ('Cifar10', 'Cifar100'):
        cls = CIFAR10 if a.dataset == 'Cifar10' else CIFAR100
        sets = [cls(str(raw), train=b, download=True) for b in (True, False)]
        images = np.concatenate([d.data.transpose(0,3,1,2) for d in sets])
        labels = np.concatenate([np.asarray(d.targets) for d in sets])
        count_train = len(sets[0])
    else:
        tiny = raw/'tiny-imagenet-200'
        if not tiny.exists():
            with zipfile.ZipFile(raw/'tiny-imagenet-200.zip') as z:
                z.extractall(raw)
        train = ImageFolder(tiny/'train')
        paths = [p for p,_ in train.samples]
        labels = [y for _,y in train.samples]
        count_train = len(labels)
        for line in (tiny/'val/val_annotations.txt').read_text().splitlines():
            name, wnid, *_ = line.split()
            paths.append(str(tiny/'val/images'/name))
            labels.append(train.class_to_idx[wnid])
        labels = np.asarray(labels)
        images = None
    num_classes = int(labels.max()+1)
    mod = partition_module()
    mod.alpha, mod.train_ratio = cfg['alpha'], cfg['train_ratio']
    ids = np.arange(len(labels))
    x,y,stat = mod.separate_data((ids, labels), cfg['num_clients'], num_classes,
                                niid=True, balance=False, partition=cfg['partition'])
    train, test = mod.split_data(x,y)
    manifest = {'original_train_count': count_train, 'total': len(labels),
                'original_split_merged': True, 'clients': {}, 'config': cfg}
    for split, records in [('train',train),('test',test)]:
        for cid,r in enumerate(records):
            ix = np.asarray(r['x'], dtype=np.int64)
            if images is None:
                pixels = np.stack([np.asarray(Image.open(paths[i]).convert('RGB')).transpose(2,0,1) for i in ix])
            else:
                pixels = images[ix]
            # Same torchvision ToTensor+Normalize(.5,.5) arithmetic as standard generators.
            pixels = pixels.astype(np.float32)/255.
            pixels = (pixels-.5)/.5
            np.savez_compressed(dest/split/f'{cid}.npz', data={'x':pixels,'y':np.asarray(r['y'])})
            manifest['clients'].setdefault(str(cid),{})[split] = ix.tolist()
            print(a.dataset,split,cid,len(ix),flush=True)
    (dest/'config.json').write_text(json.dumps(dict(cfg,num_classes=num_classes,statistic=stat),indent=2))
    (dest/'split_manifest.json').write_text(json.dumps(manifest))


if __name__ == '__main__':
    main()
