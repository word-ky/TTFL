"""Export small receipts without copying datasets or model checkpoints."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--run-ids',nargs='+',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
base=Path('/media/wenchang/F/wjq/TTFL')
home=Path('/home/wenchang/asdasdsad/wjq/TTFL')
manifest=[]
with tarfile.open(a.output,'w:gz') as tar:
    for run_id in a.run_ids:
        run=base/'runs'/run_id
        if run.exists():
            for folder in run.iterdir():
                if not folder.is_dir():
                    continue
                for f in folder.iterdir():
                    if f.is_file() and (f.suffix=='.json' or f.name=='context_predictions.npz'):
                        name=f'{run_id}/{folder.name}/{f.name}'
                        tar.add(f,arcname=name)
                        manifest.append({'path':name,'bytes':f.stat().st_size,
                            'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()})
        for filename in ('train.log','meta.json','run.sh'):
            f=home/'runs'/run_id/filename
            if f.exists():
                tar.add(f,arcname=f'logs/{run_id}/{filename}')
    for dataset in ('MNIST','Cifar10','Cifar100','TinyImagenet'):
        for filename in ('config.json','split_manifest.json'):
            tar.add(base/'dataset'/dataset/filename,arcname=f'data/{dataset}/{filename}')
    tar.add(base/'pfllib_environment.txt',arcname='environment.txt')
manifest_path=Path(a.output).with_suffix('.manifest.json')
manifest_path.write_text(json.dumps(manifest,indent=2))
print(a.output,len(manifest),'files')
