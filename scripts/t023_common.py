"""Artifact utilities only; no privileged evaluator or dataset-label imports."""
import gzip,hashlib,json
from pathlib import Path
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
B=['A','B'];S=['T013-S0','T013-S1','T013-S2','T013-S3']
DIR='t023_rotation_ssl_alignment'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gzsave(p,x):Path(p).write_bytes(gzip.compress(json.dumps(x,allow_nan=False).encode(),mtime=0))
def gzload(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def rawsha(x):return hashlib.sha256(x.tobytes()).hexdigest()
