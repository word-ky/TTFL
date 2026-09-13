"""Independent T012 ray prediction/count reconstruction and predeclared diagnostics."""
import csv,gzip,hashlib,json,shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from report_t009 import table,write_csv
from report_t010 import ranks,corr
from analyze_t012_factorization import main as factorize
ROOT=Path(__file__).resolve().parents[1]
RUN='20260914-013254-ttfl-t012-gpu1'
RAW=ROOT/f'research_log/t012_receipts/{RUN}/artifacts/t012_context_amplitude'
P7=ROOT/'research_log/t007r_receipts/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
P11=ROOT/'research_log/t011_receipts/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
OUT=ROOT/'results/t012_context_amplitude'
A=[0.,.25,.5,.75,1.,1.25,1.5];T=['brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];C=['clean']+T
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
def regretstats(rows):
    v=[r['regret'] for r in rows]
    return dict(n=len(rows),alpha1_best_fraction=sum(1. in r['best_alphas'] for r in rows)/len(rows),alpha0_best_fraction=sum(0. in r['best_alphas'] for r in rows)/len(rows),
        any_below1_fraction=sum(any(a<1 for a in r['best_alphas']) for r in rows)/len(rows),any_above1_fraction=sum(any(a>1 for a in r['best_alphas']) for r in rows)/len(rows),
        regret_positive_fraction=sum(r['regret']>0 for r in rows)/len(rows),regret_gt2_fraction=sum(r['gt2'] for r in rows)/len(rows),regret_gt5_fraction=sum(r['gt5'] for r in rows)/len(rows),
        median=float(np.median(v)),p75=float(np.quantile(v,.75)),p90=float(np.quantile(v,.9)),display_alpha_histogram={str(a):sum(min(r['best_alphas'])==a for r in rows) for a in A})

def main():
    factorize()
    for p in RAW.iterdir():
        if p.suffix in ('.csv','.json'):shutil.copyfile(p,OUT/p.name)
    q=load(RAW/'amplitude_query_utility.json');src=load(RAW/'amplitude_source_scores.json');choices=load(RAW/'amplitude_source_choices.json')
    freeze=load(RAW/'source_freeze.json');qfreeze=load(RAW/'query_freeze.json');v=load(RAW/'verification.json');meta=load(RAW/'metadata.json')
    assert sha(RAW/'amplitude_source_scores.json')==freeze['scores_sha256'] and sha(RAW/'amplitude_source_choices.json')==freeze['choices_sha256']
    assert sha(RAW/'amplitude_query_utility.json')==qfreeze['matrix_sha256'] and sha(RAW/'amplitude_predictions.npz')==qfreeze['predictions_sha256']
    assert datetime.fromisoformat(freeze['timestamp'])<datetime.fromisoformat(qfreeze['timestamp'])
    assert sha(RAW/'natural_support_manifest.json')==sha(P11/'natural_support_manifest.json')==meta['support_manifest_sha256']
    assert sha(RAW/'frozen_skew_labels_receipt.json')==sha(P11/'frozen_skew_labels_receipt.json')
    si={(r['client'],r['bank'],r['target'],r['alpha']):r for r in src};qi={(r['client'],r['bank'],r['target'],r['alpha']):r for r in q};ch={(r['client'],r['bank'],r['target']):r for r in choices}
    assert len(si)==len(qi)==5600 and len(ch)==800
    for r in choices:
        i,b,t=r['client'],r['bank'],r['target'];z=si[i,b,t,0.]
        j=min(A,key=lambda a:(si[i,b,t,a]['J'],a));allowed=[a for a in A if a>0 and si[i,b,t,a]['consistency_js']<z['consistency_js'] and si[i,b,t,a]['flip_top1_agreement']>=z['flip_top1_agreement']]
        pc=min(allowed,key=lambda a:(si[i,b,t,a]['consistency_js'],a)) if allowed else 0.
        assert j==r['J_ray'] and pc==r['PC_ray_safe']
    pred=np.load(RAW/'amplitude_predictions.npz');p7=np.load(P7/'predictions.npz')
    for r in q:
        y=pred[f"c{r['client']}_labels"];p=pred[r['prediction_key']]
        assert np.bincount(y,minlength=10).tolist()==r['class_total'] and np.bincount(y[y==p],minlength=10).tolist()==r['class_correct']
        if r['alpha'] in (0.,1.):assert np.array_equal(p,p7[r['reused_key']])
        sr=si[r['client'],r['bank'],r['target'],r['alpha']]
        assert (r['source_J'],r['source_C'],r['source_agreement'])==(sr['J'],sr['consistency_js'],sr['flip_top1_agreement'])
        assert abs(r['query_accuracy']-100*sum(r['class_correct'])/len(y))<1e-10
    sk={r['client']:r for r in load(RAW/'frozen_skew_labels_receipt.json')}
    regrets=[];alignment=[]
    for r in choices:
        i,b,t=r['client'],r['bank'],r['target'];g=[qi[i,b,t,a] for a in A];mx=max(sum(x['class_correct']) for x in g);one=qi[i,b,t,1.];zero=qi[i,b,t,0.];n=sum(one['class_total'])
        reg=Fraction(100*(mx-sum(one['class_correct'])),n);best=[x['alpha'] for x in g if sum(x['class_correct'])==mx]
        regrets.append(dict(client=i,bank=b,target=t,quartile=sk[i]['entropy_quartile'],best_alphas=best,regret=float(reg),gt2=reg>2,gt5=reg>5))
        for a in A:
            s=si[i,b,t,a];z=si[i,b,t,0.];qr=qi[i,b,t,a]
            alignment.append(dict(client=i,bank=b,target=t,alpha=a,DeltaJ=z['J']-s['J'],DeltaC=z['consistency_js']-s['consistency_js'],DeltaAcc=float(Fraction(100*(sum(qr['class_correct'])-sum(zero['class_correct'])),n))))
    ri={(r['client'],r['bank'],r['target']):r for r in regrets};reggroups=[]
    for b in ('A','B'):
        for t in T:
            g=[r for r in regrets if r['bank']==b and r['target']==t]
            for quart in ('all',1,2,3,4):reggroups.append(dict(bank=b,target=t,quartile=quart,**regretstats([r for r in g if quart=='all' or r['quartile']==quart])))
    cross=[dict(target=t,best_alpha_argmax_intersection_fraction=sum(bool(set(ri[i,'A',t]['best_alphas'])&set(ri[i,'B',t]['best_alphas'])) for i in range(100))/100) for t in T]
    save('amplitude_regret.json',dict(groups=reggroups,rows=regrets,cross_bank=cross,ties='All exact integer-correct argmax alphas retained; smallest alpha used only for deterministic display/post-hoc aggregate.'))
    all_y=np.concatenate([pred[f'c{i}_labels'] for i in range(100)]);totals=np.bincount(all_y,minlength=10)
    def macro(rows):
        counts=np.sum([r['class_correct'] for r in rows],axis=0)
        return sum((Fraction(100*int(n),int(t)) for n,t in zip(counts,totals)),Fraction())/10,counts
    oldq={(r['client'],r['bank'],r['target'],r['candidate']):r for r in load(P11/'candidate_query_utility.json')}
    oldsummary=load(P7/'summary.json');oldmetrics={(r['pool'],r['target'],r['context']):r for r in oldsummary['rows']}
    t011best={(r['bank'],r['target']):r for r in load(ROOT/'results/t011_task_proximal/summary.json')['best_of_five']}
    aggregate=[];receipts=[];history_error=0.;capture_exact={};retained_exact={};amp_gain={}
    for b in ('A','B'):
        for t in T:
            policies={name:[qi[i,b,t,alpha] for i in range(100)] for name,alpha in [('zero',0.),('alpha1',1.)]}
            policies['best_ray']=[qi[i,b,t,min(ri[i,b,t]['best_alphas'])] for i in range(100)]
            for policy in ('J_ray','PC_ray_safe'):policies[policy]=[qi[i,b,t,ch[i,b,t][policy]] for i in range(100)]
            policies['best5']=[oldq[i,b,t,max(C,key=lambda c:sum(oldq[i,b,t,c]['class_correct']))] for i in range(100)]
            values={}
            for name,rows in policies.items():
                value,counts=macro(rows);values[name]=value
                receipts.append(dict(bank=b,target=t,policy=name,class_total=totals.tolist(),class_correct=counts.tolist(),macro_class=float(value),sample_weighted=100*int(counts.sum())/int(totals.sum()),macro_client=float(np.mean([r['query_accuracy'] for r in rows]))))
            for key,ctx in [('zero','none'),('alpha1','correct_pair')]:history_error=max(history_error,abs(float(values[key])-oldmetrics[b,t,ctx]['macro_class']))
            history_error=max(history_error,abs(float(values['best5'])-t011best[b,t]['query_best_policy_macro_class']))
            raygain=values['best_ray']-values['zero'];gain5=values['best5']-values['zero'];ampgain=values['best_ray']-values['alpha1'];capture=raygain/gain5 if gain5>0 else None
            capture_exact[b,t]=capture;amp_gain[b,t]=ampgain
            r=dict(bank=b,target=t,**{k:float(v) for k,v in values.items()},best5_gain=float(gain5),ray_gain=float(raygain),ray_capture=float(capture) if capture is not None else None,amplitude_gain=float(ampgain),RAY_CAPTURE=capture is not None and capture>=Fraction(4,5))
            for name in ('J_ray','PC_ray_safe'):
                retained=(values[name]-values['zero'])/raygain if raygain>0 else None
                retained_exact[b,t,name]=retained;r[name+'_retained']=float(retained) if retained is not None else None
            aggregate.append(r)
    assert history_error<1e-10
    write_csv(OUT/'amplitude_aggregate.csv',aggregate);save('integer_count_receipts.json',receipts)
    rayalignment=[]
    for b in ('A','B'):
        for t in T:
            g=[r for r in alignment if r['bank']==b and r['target']==t]
            for metric in ('DeltaJ','DeltaC'):
                x=np.array([r[metric] for r in g]);y=np.array([r['DeltaAcc'] for r in g]);positive=x>0;harm=positive&(y<0);per=[]
                for i in range(100):
                    gg=[r for r in g if r['client']==i];v0=corr(ranks([r[metric] for r in gg]),ranks([r['DeltaAcc'] for r in gg]))
                    if v0 is not None:per.append(v0)
                rayalignment.append(dict(bank=b,target=t,metric=metric,pooled_spearman=corr(ranks(x),ranks(y)),per_client_spearman_median=float(np.median(per)) if per else None,valid_clients=len(per),positive_rows=int(positive.sum()),harmful_rows=int(harm.sum()),harmful_fraction=float(harm.sum()/positive.sum()) if positive.any() else None))
    save('ray_source_alignment.json',dict(groups=rayalignment));write_csv(OUT/'ray_alignment_rows.csv',alignment)
    joint={t:all(capture_exact[b,t] is not None and capture_exact[b,t]>=Fraction(4,5) for b in ('A','B')) for t in T};ampa=sum(joint.values())>=3
    ampb=all(sum(amp_gain[b,t]>=1 and capture_exact[b,t]<Fraction(4,5) for t in T)>=2 for b in ('A','B'))
    raygates=[]
    for name in ('J_ray','PC_ray_safe'):
        jj={t:all(retained_exact[b,t,name] is not None and retained_exact[b,t,name]>=Fraction(9,10) for b in ('A','B')) for t in T}
        raygates.append(dict(selector=name,joint_shift_pass=jj,passed=sum(jj.values())>=3))
    factor=load(OUT/'context_client_factorization.json')
    summary=dict(CLIENT_LOCK_STRONG=factor['CLIENT_LOCK_STRONG'],AMP_A=ampa,AMP_descriptor='AMP-A' if ampa else ('AMP-B' if ampb else 'AMP-C/INCONCLUSIVE'),ray_capture_joint=joint,ray_source_gates=raygates,
        primary_case='T012-B' if factor['CLIENT_LOCK_STRONG'] else ('T012-A' if ampa else 'T012-C/INCONCLUSIVE'),aggregate=aggregate)
    save('summary.json',summary)
    v.update(all_5600_predictions_integer_counts_reconstructed=True,all_1600_endpoint_arrays_match_T007R=True,T011_5000_matrix_reconstructed=True,independent_source_choices_match=True,
        source_and_query_freeze_hashes_match=True,independent_fraction_gates=True,historical_aggregate_max_error_pp=history_error,tests_passed=48)
    save('verification.json',v)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    overlap=list(csv.DictReader((OUT/'context_pair_overlap.csv').open()));simrows=[]
    for r in factor['utility_similarity']:
        valid=[s['mean'] for s in r['controls'] if s['mean'] is not None]
        simrows.append(dict(bank=r['bank'],pair=r['target1']+' / '+r['target2'],same_median=r['same']['median'],same_mean=r['same']['mean'],same_valid=r['same']['valid_count'],control_mean_of_means=float(np.mean(valid)),control_min_mean=min(valid),control_max_mean=max(valid)))
    write_csv(OUT/'utility_vector_similarity.csv',simrows)
    rayhist=[dict(bank=b,target=t,selector=name,**{str(a):sum(r['bank']==b and r['target']==t and r[name]==a for r in choices) for a in A}) for b in ('A','B') for t in T for name in ('J_ray','PC_ray_safe')]
    write_csv(OUT/'source_alpha_histograms.csv',rayhist)
    # Query-best labels appear only in this report; no fitted source mapping is produced.
    lines=['# CODEX -> CHATGPT','## Timestamp / commit / run',
        f'{now}. Lead 7d42fe2; runtime {meta["code_commit"]}; run {RUN}; release 20260914-013251-ttfl-t012; NVIDIA RTX A6000 GPU1; {meta["seconds"]:.2f}s. One formal run, exit0, no launcher retry. The enclosing delivery commit contains the independent report.',
        '## Frozen objects and verification',
        '48 tests PASS (46 historical plus two scaling tests). Checkpoint, 192-scalar affine model, neutral writer, corruptions, T007R states, T008 references, exact T009 K20 manifest, T010 moment implementation and T011 consistency implementation hashes unchanged. No source/query split change, state fitting, gradient optimization, new federation or OOD rerun. The dataset remains the historical PFLlib merged-data per-client CIFAR-10 query split, not the official CIFAR-10 test benchmark.',
        '## T011 matrix reconstruction',
        'Phase I independently reconstructs all 5,000 T011 rows from saved predictions and integer class counts. Full exact argmax sets are retained. Per-context utility-vector Spearman uses count ranks: subtracting zero and dividing by the shared client/context query count leaves those five ranks exactly unchanged. No new model forward pass is used in Phase I.',
        '## Context-vs-client preference decomposition',
        'Every unordered context pair is evaluated. The second client is shifted by exactly [7,13,23,37,41,53,71,89] modulo100; no offset search. Overlap values are fractions; differences are percentage points.',
        tab(['bank','target1','target2','same_client_overlap','control_mean','control_min','control_max','difference_pp'],overlap),
        tab(['bank','quartile','n','fraction_ge3','fraction_ge4','fraction_eq5','distribution'],factor['persistence']),
        'Utility-vector Spearman below averages only valid nonconstant comparisons. Full per-offset means, medians and valid counts are saved in context_client_factorization.json; the compact control summary is the mean/min/max of the eight offset means.',
        tab(['bank','pair','same_median','same_mean','same_valid','control_mean_of_means','control_min_mean','control_max_mean'],simrows),
        '## CLIENT-LOCK decision',
        tab(['bank','mean_same_overlap','mean_control_overlap','difference_pp','fraction_ge3','passed'],factor['gates']),
        '**CLIENT-LOCK-STRONG in both banks.** Same-client overlap exceeds mismatch controls by 28.025 /28.2125 pp, above 15 pp. 79% /78% of clients have a candidate optimal in >=3/5 contexts, above50%. Frozen states exhibit persistent client preference in addition to context response. Thus T011 best-of-five headroom cannot be interpreted as pure context headroom. This does not isolate whether label composition or another client property causes the persistence.',
        '## Alpha scaling implementation and regression',
        'scale_state(state, alpha) linearly multiplies every gamma/beta tensor without clipping or mutation. Grid fixed at 0,.25,.5,.75,1,1.25,1.5. Before scoring, all eight saved state directions pass alpha0 ordinary-logit equality, alpha1 tensor/logit equality and finite states/logits across all seven alphas. Repeated construction is deterministic. An additional1,600 endpoint logit checks run on actual supports; max difference0. Saved/scaled state and model hashes remain unchanged after every source/query episode.',
        '## Source-score freeze',
        f'All5,600 source rows and800 decision rows (both selectors in each row) frozen at {freeze["timestamp"]}. Scores SHA {freeze["scores_sha256"]}; choices SHA {freeze["choices_sha256"]}. Query amplitude matrix frozen later at {qfreeze["timestamp"]}. J-ray is argminJ; PC-ray-safe is strict consistency improvement with nondecreasing agreement, otherwise0; both use smaller-alpha tie order. Original T010/T011 formulas unchanged.',
        '**The true target chooses the ray direction. This is a privileged, direction-conditioned oracle diagnostic, not deployable source-only adaptation.** Phase I used historical fixed-state query results, but neither new amplitude outcomes nor a query-fitted rule entered source selection.',
        '## Query amplitude matrix verification',
        f'5,600 policies:1,600 exact T007R alpha0/alpha1 arrays reused,4,000 missing amplitudes evaluated. Every prediction vector independently matches saved integer per-class correct/total counts; every endpoint array matches T007R. T007R endpoint aggregates and T011 best-of-five aggregates reconstruct with maximum error {history_error:.3g} pp. Freeze hashes and all source choices rechecked; gate arithmetic uses exact count fractions.',
        '## Best-ray / alpha=1 / best-of-five comparison',
        tab(['bank','target','zero','alpha1','best_ray','best5','amplitude_gain','ray_capture','RAY_CAPTURE'],aggregate),
        'Dark best-ray exceeds the five-state oracle because the grids differ: scaled states add candidates absent from the five-state bank. Capture can therefore exceed1 and is not clipped. Dark/Contrast exceed80% capture jointly; Noise/Blur do not. On Noise, capture is60.55% /56.40%; on Blur47.87% /49.19%. Noise/Blur nevertheless improve over alpha1 by2.34–2.55pp. Fixed direction amplitude helps substantially but leaves client-conditioned direction/state-bank headroom.',
        'Both upper bounds maximize on the same finite per-client query outcomes and are optimistic diagnostics, not independent test performance. Exact per-client correct-count ties use the smallest alpha (or historical fixed candidate order) for aggregate display; all tied optima are preserved separately. Macro-class is reconstructed after mixing client choices, not averaged from per-client class means.',
        '## Alpha regret and entropy-quartile audit',
        tab(['bank','target','quartile','alpha1_best_fraction','alpha0_best_fraction','any_below1_fraction','any_above1_fraction','regret_gt2_fraction','regret_gt5_fraction','median','p75','p90'],reggroups),
        'The argmax fractions overlap intentionally when there are ties. All display-alpha histograms, including every frozen entropy quartile, are stored in amplitude_regret.json. No labels/entropy were used to choose source amplitude.',
        tab(['bank','target','quartile','display_alpha_histogram'],reggroups),
        tab(['target','best_alpha_argmax_intersection_fraction'],cross),
        '## J-ray / PC-ray-safe source selection',
        tab(['bank','target','zero','alpha1','best_ray','J_ray','PC_ray_safe','J_ray_retained','PC_ray_safe_retained'],aggregate),
        tab(['bank','target','metric','pooled_spearman','per_client_spearman_median','valid_clients','positive_rows','harmful_fraction'],rayalignment),
        'Each within-context pooled correlation has700 client/alpha rows; per-client Spearman uses seven amplitudes with average ranks. Constant vectors are excluded from correlation and valid counts retained. Harm fractions are conditional on strictly positive source improvement; null means no positive rows, not zero harm.',
        'PC-ray-safe always chooses zero for Dark/Contrast because no positive amplitude lowers JS. The corresponding median per-client consistency/task correlations are strongly negative. J-ray captures about72% of best-ray gain on Dark,63–69% on Contrast,32–35% on Noise and21–23% on Blur. PC-ray-safe captures about48–49% on Noise and29–30% on Blur. Source scoring remains inadequate even with the true direction supplied.',
        '## RAY-SOURCE result',json.dumps(raygates,indent=2),
        '**RAY-SOURCE-A fails for each predeclared selector:0/4 shifts jointly retain90% of best-ray gain.** No hybrid selector was assembled from whichever method looked better on a target.',
        '## AMP-A/B/C decision',
        '**AMP-A false; AMP-B applies.** Only2/4 shifts satisfy joint RAY-CAPTURE. In each bank, both Noise and Blur improve by>=1pp over alpha1 while remaining below80% capture, exactly meeting the fixed AMP-B descriptor. AMP-C/INCONCLUSIVE is not selected because amplitude gains are material under that descriptor.',
        '**Primary case T012-B:** CLIENT-LOCK-STRONG plus incomplete true-ray capture supports a persistent client component mixed with transient context correction. T012-A does not hold; this is not evidence that scalar amplitude alone suffices. The small-amplitude-gain premise of T012-C is not met. Source-estimation failure coexists with direction/client mismatch; neither explains everything alone.',
        '## Mechanism vs implementation conclusion',
        'All tests, endpoint regressions, immutability checks and prediction/count audits pass. The states carry both context and persistent client preference, while amplitude is useful but insufficient on Noise/Blur. Current source criteria also fail to select near-oracle amplitude. The evidence motivates explicit client-persistent plus transient-context factorization before a writer; it does not justify enlarging the operator or asserting a particular learned factorization already works.',
        '## Recommended next action',
        'Return to Research Lead with the full decomposition and ray matrices. Preserve zero neutrality and the192-scalar operator. A next bounded package should separate persistent client calibration from transient context state rather than promote J/flip consistency directly into a learned writer. No next stage, new grid, SSL, meta-learning, federation or richer operator was launched.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T012 COMPLETE — T012-B, CLIENT-LOCK-STRONG, AMP-B\n\nRead results/t012_context_amplitude/RESULTS.md and research_log/t012_delivery.json. PhaseI overlapexcess28.025/28.2125pp,persistence>=3 .79/.78. AMP-Afalse:2/4jointcapture;Noise.605/.564,Blur.479/.492 capture despite>=2.34pp amplitudegain. Bothrayselectors0/4joint90% retention.48tests/5600counts/endpoints/freezes PASS. Runtime87f1bde,run20260914-013254-ttfl-t012-gpu1 exit0. AwaitLead; no nextstage.\n',encoding='utf-8')
    delivery=dict(timestamp=now,lead='7d42fe2',runtime=meta['code_commit'],run=RUN,decision='T012-B / AMP-B',raw=RAW.relative_to(ROOT).as_posix(),remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t012_context_amplitude',results='results/t012_context_amplitude/RESULTS.md',next='Await Lead')
    (ROOT/'research_log/t012_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    for folder in (RAW,OUT):
        for name in ('amplitude_source_scores.json','amplitude_query_utility.json'):
            p=folder/name;p.with_suffix('.json.gz').write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t012_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t012_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,regret=[r for r in reggroups if r['quartile']=='all'],alignment=rayalignment,crossbank=cross),indent=2))

if __name__=='__main__':main()
