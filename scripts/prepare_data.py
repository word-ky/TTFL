"""Acquire official torchvision Digits plus the documented MNIST-M image mirror."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from torchvision.datasets import MNIST, USPS, SVHN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    MNIST.mirrors = ['https://storage.googleapis.com/cvdf-datasets/mnist/']
    for cls in (MNIST, USPS):
        for train in (True, False):
            ds = cls(str(root / cls.__name__), train=train, download=True)
            print(cls.__name__, train, len(ds), flush=True)
    # Retain official Stanford HTTP URLs and torchvision's published MD5 checks.
    for split in ('train', 'test'):
        ds = SVHN(str(root / 'SVHN'), split=split, download=True)
        print('SVHN', split, len(ds), flush=True)
    url = 'https://media.githubusercontent.com/media/mashaan14/MNIST-M/main/MNIST-M.zip'
    archive = root / 'MNIST-M.zip'
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == 'bd899e2c286f1267955a73f6a374f31fc98764380bb8928dbfee37d8ab0393d6'
    with zipfile.ZipFile(archive) as z:
        print('MNIST-M archive samples', z.namelist()[:30], flush=True)
        z.extractall(root / 'MNISTM')
    manifest = {str(f.relative_to(root)): {'bytes': f.stat().st_size,
                'sha256': hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()}
                for f in root.rglob('*') if f.is_file() and f.suffix in ('.zip', '.gz', '.bz2', '.mat')}
    (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
