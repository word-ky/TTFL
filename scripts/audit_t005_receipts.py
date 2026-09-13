"""Independent integer-prediction/headroom audit of the frozen T005 gate."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True);p.add_argument('--identity-predictions',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();raw,out=Path(a.raw),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    s=json.loads((raw/'summary.json').read_text());v=json.loads((raw/'verification.json').read_text())
    rows=json.loads((raw/'raw_records.json').read_text());pairs=json.loads((raw/'pairings.json').read_text())
    pred=np.load(raw/'predictions.npz');clean=np.load(a.identity_predictions)
    total=0;clean_correct=0
    for cid in range(100):
        labels=pred[f'c{cid}_labels'];assert np.array_equal(labels,clean[f'c{cid}_labels'])
        total+=len(labels);clean_correct+=int((clean[f'c{cid}_none']==labels).sum())
        r=pairs[str(cid)];perm=r['permutation'];ids=r['source_ids']
        assert sorted(perm)==list(range(len(ids))) and all(i!=j for i,j in enumerate(perm))
        assert r['target_ids']==[ids[i] for i in perm] and sorted(ids)==sorted(r['target_ids'])
        assert not set(ids).intersection(r['query_ids'])
    counts=[]
    for result in s['specificity']:
        target=result['target'];c={}
        for r in rows:
            if r['target']==target:
                c[r['context']]=c.get(r['context'],0)+int((pred[r['prediction_key']]==pred[f"c{r['client']}_labels"]).sum())
        headroom=clean_correct-c['none'];gain=c['oracle_correct_pair']-c['none']
        gaps={ctx:100*(c['oracle_correct_pair']-c[ctx])/total for ctx in
              ('oracle_target_permuted','oracle_wrong_alt','oracle_noise_source')}
        recovery=gain/headroom if headroom>0 else None
        passed=recovery is not None and recovery>=.5 and all(g>=2 for g in gaps.values())
        assert passed==result['passed']
        assert abs(recovery-result['recovery_fraction'])<1e-5
        assert abs(gaps['oracle_target_permuted']-result['pairing_advantage_pp'])<1e-4
        counts.append(dict(target=target,query_count=total,correct_counts=c,clean_correct=clean_correct,
            headroom_count=headroom,gain_count=gain,recovery_fraction=recovery,
            pairing_advantage_pp=gaps['oracle_target_permuted'],passed=passed))
    old_writer=subprocess.check_output(['git','show','68ce150:src/adaptation/paired_affine_oracle.py'])
    assert hashlib.sha256(old_writer).hexdigest()==v['writer_sha256']
    v.update(integer_prediction_gates_match=True,independent_clean_accuracy=100*clean_correct/total,
             raw_query_count=total,old_t004_writer_byte_hash_matches=True,
             independent_derangement_and_id_checks=True)
    for name in ('summary.csv','summary.json','per_client.csv','restoration_diagnostics.csv'):
        shutil.copyfile(raw/name,out/name)
    (out/'verification.json').write_text(json.dumps(v,indent=2))
    (out/'integer_prediction_audit.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    main()
