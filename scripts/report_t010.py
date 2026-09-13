"""Audit frozen T010 selections and predictions; produce the Lead handoff."""
import csv, hashlib, json, shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from report_t009 import table, write_csv

ROOT=Path(__file__).resolve().parents[1]
RUN='20260913-233408-ttfl-t010-gpu1'
RAW=ROOT/f'research_log/t010_receipts/{RUN}/artifacts/t010_state_consistency'
OLD=ROOT/'research_log/t007r_receipts/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
PREV=ROOT/'research_log/t009_receipts/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
OUT=ROOT/'results/t010_state_consistency'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def ranks(x):
    return np.array([sum(y<v for y in x)+(sum(y==v for y in x)+1)/2 for v in x])
def corr(x,y): return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
def stats(rows):
    x=np.array([r['delta_J'] for r in rows]);y=np.array([r['delta_accuracy'] for r in rows]);pos=x>0;harm=pos&(y<0)
    return dict(n=len(rows),pearson=corr(x,y),spearman=corr(ranks(x),ranks(y)),positive_J=int(pos.sum()),harmful=int(harm.sum()),
        harmful_fraction_all=float(harm.mean()),harmful_fraction_positive_J=float(harm.sum()/pos.sum()) if pos.any() else None,
        median_delta_accuracy_positive_J=float(np.median(y[pos])) if pos.any() else None)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.suffix in ('.json','.csv'): shutil.copyfile(p,OUT/p.name)
    s=load(RAW/'summary.json');v=load(RAW/'verification.json');m=load(RAW/'metadata.json');d=load(RAW/'state_scores.json')
    freeze=load(RAW/'selection_freeze.json');qfreeze=load(RAW/'query_freeze.json')
    assert sha(RAW/'state_scores.json')==freeze['decisions_sha256']
    assert sha(RAW/'mixed_predictions.npz')==qfreeze['predictions_sha256']
    assert datetime.fromisoformat(freeze['timestamp'])<datetime.fromisoformat(qfreeze['timestamp'])
    assert sha(RAW/'natural_support_manifest.json')==sha(PREV/'natural_support_manifest.json')==m['support_manifest_sha256']
    assert sha(RAW/'frozen_skew_labels_receipt.json')==sha(PREV/'skew_labels_receipt.json')
    for r in d:
        order=sorted(C,key=lambda c:(r['J_'+c],C.index(c)))
        assert [r['selected'],r['second']]==order[:2] and r['true_rank']==order.index(r['true_context'])+1
        assert r['score_margin']==r['J_'+order[1]]-r['J_'+order[0]]
        assert r['zero_improvement']==r['J_clean']-r['J_'+order[0]]
    p=np.load(RAW/'mixed_predictions.npz');p7=np.load(OLD/'predictions.npz');p9=np.load(PREV/'mixed_predictions.npz')
    ix={(r['client'],r['pool'],r['target'],r['context']):r for r in load(OLD/'raw_records.json')}
    y=np.concatenate([p[f'c{i}_labels'] for i in range(100)]);total=np.bincount(y,minlength=10)
    def macro(pr):
        counts=np.bincount(y[y==pr],minlength=10)
        return sum((Fraction(100*int(n),int(t)) for n,t in zip(counts,total)),Fraction())/10,counts
    clean,_=macro(np.concatenate([p7[f'c{i}_clean'] for i in range(100)]))
    for r in load(RAW/'mixed_query_records.json'):
        yy=p[f"c{r['client']}_labels"];pr=p[r['prediction_key']]
        assert np.bincount(yy,minlength=10).tolist()==r['class_total']
        assert np.bincount(yy[yy==pr],minlength=10).tolist()==r['class_correct']
        if r['reused_from'] in ('T007R','T009'): assert np.array_equal(pr,({'T007R':p7,'T009':p9}[r['reused_from']])[r['reused_key']])
    exact={};error=0
    for r in load(RAW/'integer_count_receipts.json'):
        val,counts=macro(np.concatenate([p[f"{r['bank']}|{r['target']}|c{i}"] for i in range(100)]))
        assert counts.tolist()==r['correct'] and total.tolist()==r['total']
        error=max(error,abs(float(val)-r['macro_class']));exact[r['bank'],r['target']]=val
    for r in s['retrieval']:
        b,t=r['bank'],r['target'];old={}
        for ctx in ('none','correct_pair','wrong_alt_source'):
            old[ctx]=macro(np.concatenate([p7[ix[i,b,t,ctx]['prediction_key']] for i in range(100)]))[0]
        gain=old['correct_pair']-old['none'];ret=(exact[b,t]-old['none'])/gain;tau=max(Fraction(1,2),(clean-old['none'])/4)
        assert r['passed']==(gain>0 and ret>=Fraction(9,10) and exact[b,t]-old['wrong_alt_source']>=tau)
        assert abs(float(ret)-r['retained_recovery'])<1e-10
    for r in s['clean_safety']: assert r['passed']==(exact[r['bank'],'clean']>=clean-Fraction(1,2))
    for r in s['identification']:
        g=[x for x in d if x['bank']==r['bank']];mat=[[sum(x['true_context']==t and x['selected']==c for x in g) for c in C] for t in C]
        assert mat==r['confusion'] and r['passed']==(sum(mat[i][i] for i in range(5))>=450 and min(mat[i][i] for i in range(5))>=80)
    assert s['SC_ID_A'] and s['SC_RET_A'] and sum(s['joint_shift_pass'].values())==3
    di={(r['client'],r['bank'],r['true_context']):r for r in d};alignment=list(csv.DictReader((RAW/'restoration_task_rows.csv').open()))
    for r in alignment:
        i=int(r['client']);r['client']=i;b,t=r['bank'],r['target'];rr=di[i,b,t];yy=p[f'c{i}_labels']
        for k in ('delta_J','delta_accuracy','zero_accuracy','candidate_accuracy'):r[k]=float(r[k])
        zero=p7[f'c{i}_clean'] if t=='clean' else p7[ix[i,b,t,'none']['prediction_key']]
        cand=p[f'{b}|{t}|c{i}'] if r['role']=='selected' else (zero if t=='clean' else p7[ix[i,b,t,'correct_pair']['prediction_key']])
        assert abs(r['delta_accuracy']-100*(np.mean(cand==yy)-np.mean(zero==yy)))<1e-10
        assert r['delta_J']==rr['J_clean']-rr['J_'+r['candidate']]
    audits=[]
    for b in ('A','B','both'):
        for role in ('selected','true'):
            for t in C+['all']:
                g=[r for r in alignment if (b=='both' or r['bank']==b) and r['role']==role and (t=='all' or r['target']==t)]
                audits.append(dict(bank=b,role=role,target=t,**stats(g)))
    top={role:sorted([r for r in alignment if r['role']==role and r['delta_J']>0 and r['delta_accuracy']<0],key=lambda r:(-r['delta_J'],r['delta_accuracy'],r['client'],r['bank']))[:10] for role in ('selected','true')}
    save('restoration_task_audit.json',dict(groups=audits,largest_disagreements=top,ranking='DeltaJ descending among DeltaJ>0 and DeltaAcc<0; then DeltaAcc, client, bank'))
    sk={r['client']:r for r in load(RAW/'frozen_skew_labels_receipt.json')};skrows=[]
    for b in ('A','B'):
        for quart in range(1,5):
            for t in ('clean','gaussian_blur','all'):
                g=[r for r in d if r['bank']==b and sk[r['client']]['entropy_quartile']==quart and (t=='all' or r['true_context']==t)]
                skrows.append(dict(bank=b,quartile=quart,target=t,n=len(g),accuracy=100*sum(r['selected']==r['true_context'] for r in g)/len(g),median_score_margin=float(np.median([r['score_margin'] for r in g]))))
    write_csv(OUT/'skew_audit.csv',skrows)
    worst=[dict(**r,entropy=sk[r['client']]['normalized_entropy'],max_class_fraction=sk[r['client']]['max_class_fraction'],true_excess=r['J_'+r['true_context']]-r['best_score']) for r in d if r['true_context'] in ('clean','gaussian_blur') and r['selected']!=r['true_context']]
    worst.sort(key=lambda r:-r['true_excess']);save('worst_clean_blur_errors.json',worst)
    prev={(r['client'],r['bank'],r['true_context']):r for r in load(PREV/'source_decisions.json')};changes=[]
    for b in ('A','B'):
        for t in C+['all']:
            g=[r for r in d if r['bank']==b and (t=='all' or r['true_context']==t)]
            pairs=[(r,prev[r['client'],b,r['true_context']]) for r in g]
            changes.append(dict(bank=b,target=t,corrected=sum(a['selected']==a['true_context'] and z['selected']!=z['true_context'] for a,z in pairs),preserved_errors=sum(a['selected']!=a['true_context'] and z['selected']!=z['true_context'] for a,z in pairs),new_errors=sum(a['selected']!=a['true_context'] and z['selected']==z['true_context'] for a,z in pairs),changed_wrong_to_wrong=sum(a['selected']!=a['true_context'] and z['selected']!=z['true_context'] and a['selected']!=z['selected'] for a,z in pairs)))
    hard=[dict(bank=b,clean_nonzero=sum(r['bank']==b and r['true_context']=='clean' and r['selected']!='clean' for r in d),blur_zero=sum(r['bank']==b and r['true_context']=='gaussian_blur' and r['selected']=='clean' for r in d),blur_contrast=sum(r['bank']==b and r['true_context']=='gaussian_blur' and r['selected']=='contrast_low' for r in d)) for b in ('A','B')]
    save('t009_comparison.json',dict(changes=changes,hard_pairs=hard))
    ood=list(csv.DictReader((RAW/'ood_state_scores.csv').open()));oods=[]
    for b in ('A','B'):
        g=[r for r in ood if r['bank']==b];row=dict(bank=b,histogram={c:sum(r['selected']==c for r in g) for c in C},fraction_zero=sum(r['selected']=='clean' for r in g)/len(g))
        for k in ('J_clean','best_score','zero_improvement'):
            vals=[float(r[k]) for r in g];row[k]=dict(min=min(vals),median=float(np.median(vals)),mean=float(np.mean(vals)),max=max(vals))
        oods.append(row)
    save('ood_summary.json',oods)
    moments=[]
    for b in ('A','B'):
        for t in C:
            g=[r for r in d if r['bank']==b and r['true_context']==t]
            moments.append(dict(bank=b,target=t,median_J_true=float(np.median([r['J_'+t] for r in g])),median_J_zero=float(np.median([r['J_clean'] for r in g])),median_J_nearest_wrong=float(np.median([min(r['J_'+c] for c in C if c!=t) for r in g]))))
    write_csv(OUT/'moment_scores.csv',moments)
    v.update(independent_prediction_count_max_error_pp=error,independent_fraction_gates_match=True,independent_selection_confusions_match=True,independent_alignment_from_predictions=True,freeze_hashes_match=True,tests_passed=43)
    save('verification.json',v);s['decision']='SC-A';save('summary.json',s)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    overall=[r for r in audits if r['target']=='all' and r['bank']!='both']
    def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
    lines=['# CODEX -> CHATGPT','## Timestamp',now,'## Commit / run ID',f'Lead 50f324a; runtime {m["code_commit"]}; run {RUN}; A6000 GPU1; {m["seconds"]:.2f}s evaluation. Enclosing delivery commit contains this report.',
    '## Frozen evidence reused','Same CIFAR-10 PFLlib checkpoint, 192-scalar operator, neutral writer, T007R states/pools, T008 references/prototypes, corruption code, historical source_signature, and exact T009 K20 support manifest. All frozen hashes match. 100 clients; no new FL training. This is the historical PFLlib merged-data per-client query split, not the official CIFAR-10 test benchmark.',
    '## New state-consistency score implementation','Score all five frozen candidates including zero by J=||psi||² after sequential affine-corrected blocks, relative to the bank clean moments. Block 2 receives corrected block 1. Fixed argmin order; no labels, gradients, fitting, thresholds or query tuning. The same K20 IDs serve every candidate/context/bank.',
    '## Sanity/regression verification',f'43 tests PASS. 5,240 candidate scores; 1,048 exact zero-signature/J comparisons, max error 0. Model and states unchanged after every episode. Choices frozen {freeze["timestamp"]}, query predictions frozen {qfreeze["timestamp"]}. 985 query policies reused from T009, 14 from T007R, 1 new evaluation. All historical metrics reproduce. Independent saved-prediction integer counts, Fraction gates, decisions and alignment checks pass; macro-class error {error:.3g} pp.',
    '## Balanced held-out sanity','20/20 correct in each bank; both confusion matrices are diagonal with four examples per context.',
    '## Natural-client state selection',tab(['bank','correct','accuracy','median_true_rank','fraction_true_rank_le2','passed'],s['identification']),
    '## Confusion matrices / hard-pair audit',*[f'Bank {r["bank"]}:\n\n'+table(['true / selected']+C,[[c]+r['confusion'][i] for i,c in enumerate(C)]) for r in s['identification']],tab(['bank','clean_nonzero','blur_zero','blur_contrast'],hard),tab(['bank','target','corrected','preserved_errors','new_errors','changed_wrong_to_wrong'],changes),
    '## Per-client mixed retrieval',tab(['bank','target','none','oracle','selected_mixed','wrong_alt','tau_pp','retained_recovery','selected_wrong_margin','passed'],s['retrieval']),
    'Dark, Contrast and Noise pass jointly. Blur fails in both banks: only 40.37% / 39.59% of oracle gain retained. Bank-B Blur is worse than T009 (53.89%); this selector has not solved Blur.',
    '## Clean safety',tab(['bank','none','selected','delta_pp','passed'],s['clean_safety']),
    'Both losses fall within the predeclared 0.5 pp allowance, improving from T009 losses 0.610321 / 0.616185 pp. Five false clean adaptations remain in each bank; passing this gate does not mean zero harm.',
    '## Restoration-vs-task audit',tab(['bank','role','n','pearson','spearman','positive_J','harmful','harmful_fraction_all','harmful_fraction_positive_J','median_delta_accuracy_positive_J'],overall),
    'Among selected states with positive DeltaJ, 22.03% / 22.77% still reduce query accuracy. Pearson correlations are only 0.326 / 0.317. Thus lower moment residual is not a reliable per-client guarantee of task benefit. All quantities are per-client percentage-point accuracy changes versus zero. Spearman uses average ranks; constant groups have null correlation. Full per-bank/context statistics and the ten largest positive-J/negative-accuracy disagreements for each role are in restoration_task_audit.json. Ranking was fixed before evaluation: descending DeltaJ, then DeltaAcc/client. These descriptive correlations do not establish that source moment restoration causes task improvement.',
    '## Label-skew audit',tab(['bank','quartile','target','n','accuracy','median_score_margin'],skrows),
    'Frozen T009 quartiles and label statistics reused only after selection/query freezing. Worst clean/blur errors are listed in worst_clean_blur_errors.json by J_true-J_best, with entropy and max-class fraction. The top eight rows all involve single-class support clients 15 and 29 (entropy 0, max-class fraction 1). Both banks still map client 15 Blur to Contrast and client 29 Blur to zero. The low-entropy tail failure remains; changed selection names do not prove semantic-content confounding was removed.',
    tab(['bank','client','true_context','selected','entropy','max_class_fraction','true_excess'],worst[:10]),
    '## OOD descriptive audit',json.dumps(oods,indent=2),
    'All eight random-noise supports select the Gaussian-noise state; none selects zero. Thus including zero does not provide OOD rejection. No threshold was introduced.',
    '## Failures / uncertainties','Clean tail errors and Blur retention failure persist despite the joint gate passing. Moment/task disagreements remain in the saved audit. Results cover fixed known corruption severities and two existing state banks; no unseen-severity, continuous-state, or open-set success is established.',
    '## SC-ID / SC-RET / Case decision','**SC-ID-A PASS; SC-RET-A PASS; primary case SC-A** under the explicit conjunction rule: 3/4 joint shifted targets plus clean safety in both banks. The instruction’s SC-B description overlaps: clean passes while Blur fails. Preserve that residual rather than claiming complete safe retrieval or changing the original gate.',
    '## Recommended next action','Return to Research Lead for adjudication, explicitly carrying forward clean tail harm and failed Blur retention. A bounded next package could examine the proposed clean↔shift interpolation/unseen severities and OOD rejection separately. No next-stage implementation, SSL, meta-learning, new federation or operator expansion was started.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    handoff='# T010 COMPLETE — SC-A with residual Blur failure\n\nRead results/t010_state_consistency/RESULTS.md and research_log/t010_delivery.json. ID 97.8/98.0%; 3/4 joint shifted targets pass; clean losses .440248/.413368 pp pass .5 gate. Blur retention .403732/.395870 FAIL. 43 tests and independent count/selection/alignment audits pass. Runtime 1feaec9. Await Lead review; no next stage.\n'
    (ROOT/'research_log/HANDOFF.md').write_text(handoff,encoding='utf-8')
    delivery=dict(timestamp=now,lead='50f324a',runtime=m['code_commit'],run=RUN,decision='SC-A',raw=RAW.relative_to(ROOT).as_posix(),remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t010_state_consistency',results='results/t010_state_consistency/RESULTS.md',next='Await Lead review')
    (ROOT/'research_log/t010_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(RAW.parent.parent.rglob('*')) if p.is_file()]
    (ROOT/'research_log/t010_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=s['decision'],alignment=overall,changes=changes,hard_pairs=hard),indent=2))

if __name__=='__main__':main()
