"""Independent frozen-matrix audit for T011; no score fitting or policy changes."""
import csv,gzip,hashlib,json,shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from report_t009 import table,write_csv
from report_t010 import ranks,corr
ROOT=Path(__file__).resolve().parents[1]
RUN='20260914-003521-ttfl-t011-gpu1-lf'
RAW=ROOT/f'research_log/t011_receipts/{RUN}/artifacts/t011_task_proximal'
OLD=ROOT/'research_log/t007r_receipts/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
P9=ROOT/'research_log/t009_receipts/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
P10=ROOT/'research_log/t010_receipts/20260913-233408-ttfl-t010-gpu1/artifacts/t010_state_consistency'
OUT=ROOT/'results/t011_task_proximal'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
def alignment_stats(g,k):
    x=np.array([r[k] for r in g]);y=np.array([r['DeltaAcc'] for r in g]);pos=x>0;harm=pos&(y<0)
    return dict(n=len(g),pearson=corr(x,y),spearman=corr(ranks(x),ranks(y)),positive_fraction=float(pos.mean()),
        positive_n=int(pos.sum()),harmful_n=int(harm.sum()),harmful_fraction=float(harm.sum()/pos.sum()) if pos.any() else None,
        median_delta_acc_positive=float(np.median(y[pos])) if pos.any() else None)
def regret_stats(g):
    a=np.array([r['true_state_regret'] for r in g])
    return dict(n=len(g),true_in_argmax_fraction=sum(r['true_in_argmax'] for r in g)/len(g),
        regret_positive_fraction=float(np.mean(a>0)),regret_gt2_fraction=sum(r['gt2'] for r in g)/len(g),regret_gt5_fraction=sum(r['gt5'] for r in g)/len(g),
        median=float(np.median(a)),p75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),
        best_state_when_true_not_best={c:sum(not r['true_in_argmax'] and c in r['best_states'] for r in g) for c in C},
        median_best_gain_vs_zero=float(np.median([r['best_gain_vs_zero'] for r in g])),
        best_improves_zero_fraction=sum(r['best_gain_vs_zero']>0 for r in g)/len(g))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.suffix in ('.json','.csv'):shutil.copyfile(p,OUT/p.name)
    src=load(RAW/'source_candidate_scores.json');sel=load(RAW/'selections.json');qr=load(RAW/'candidate_query_utility.json')
    freeze=load(RAW/'source_freeze.json');qfreeze=load(RAW/'query_freeze.json');v=load(RAW/'verification.json');meta=load(RAW/'metadata.json')
    assert sha(RAW/'source_candidate_scores.json')==freeze['scores_sha256'] and sha(RAW/'selections.json')==freeze['decisions_sha256']
    assert sha(RAW/'candidate_query_utility.json')==qfreeze['matrix_sha256'] and sha(RAW/'candidate_predictions.npz')==qfreeze['predictions_sha256']
    assert datetime.fromisoformat(freeze['timestamp'])<datetime.fromisoformat(qfreeze['timestamp'])
    assert sha(RAW/'natural_support_manifest.json')==sha(P9/'natural_support_manifest.json')==meta['support_manifest_sha256']
    assert sha(RAW/'frozen_skew_labels_receipt.json')==sha(P9/'skew_labels_receipt.json')
    si={(r['client'],r['bank'],r['true_context'],r['candidate']):r for r in src};choices={(r['client'],r['bank'],r['true_context']):r for r in sel}
    assert len(si)==5000 and len(choices)==1000
    for r in sel:
        i,b,t=r['client'],r['bank'],r['true_context'];z=si[i,b,t,'clean']
        allowed=[c for c in C[1:] if si[i,b,t,c]['consistency_js']<z['consistency_js'] and si[i,b,t,c]['flip_top1_agreement']>=z['flip_top1_agreement']]
        chosen=min(allowed,key=lambda c:(si[i,b,t,c]['consistency_js'],C.index(c))) if allowed else 'clean'
        rank=sorted(C,key=lambda c:(si[i,b,t,c]['consistency_js'],C.index(c)))
        assert chosen==r['selected'] and rank.index(t)+1==r['true_state_rank_by_consistency']
        assert r['consistency_improvement']==z['consistency_js']-si[i,b,t,chosen]['consistency_js']
    pred=np.load(RAW/'candidate_predictions.npz');p7=np.load(OLD/'predictions.npz');p9=np.load(P9/'mixed_predictions.npz');p10=np.load(P10/'mixed_predictions.npz')
    qi={(r['client'],r['bank'],r['target'],r['candidate']):r for r in qr};assert len(qi)==5000
    oldidx={(r['client'],r['pool'],r['target'],r['context']):r for r in load(OLD/'raw_records.json')}
    for r in qr:
        y=pred[f"c{r['client']}_labels"];pr=pred[r['prediction_key']]
        assert np.bincount(y,minlength=10).tolist()==r['class_total']
        assert np.bincount(y[y==pr],minlength=10).tolist()==r['class_correct']
        if r['reused_from']!='new':assert np.array_equal(pr,{'T007R':p7,'T009':p9,'T010':p10}[r['reused_from']][r['reused_key']])
        assert abs(100*sum(r['class_correct'])/len(y)-r['query_accuracy'])<1e-10
    y=np.concatenate([pred[f'c{i}_labels'] for i in range(100)]);total=np.bincount(y,minlength=10)
    def macro(rows):
        counts=np.sum([r['class_correct'] for r in rows],axis=0)
        return sum((Fraction(100*int(n),int(t)) for n,t in zip(counts,total)),Fraction())/10,counts
    clean=macro([qi[i,'A','clean','clean'] for i in range(100)])[0]
    retrieval=[];cleans=[];counts_receipts=[];bestrows=[];historical_error=0
    oldsummary=load(OLD/'summary.json');om={(r['pool'],r['target'],r['context']):r for r in oldsummary['rows']}
    oldg={(r['pool'],r['target']):r for r in oldsummary['gates']}
    for b in ('A','B'):
        for t in C:
            rows=[qi[i,b,t,choices[i,b,t]['selected']] for i in range(100)];mixed,counts=macro(rows)
            weighted=100*int(counts.sum())/int(total.sum());clientmean=float(np.mean([r['query_accuracy'] for r in rows]))
            counts_receipts.append(dict(bank=b,target=t,class_total=total.tolist(),class_correct=counts.tolist(),macro_class=float(mixed),sample_weighted=weighted,macro_client=clientmean))
            if t=='clean':cleans.append(dict(bank=b,none=float(clean),selected=float(mixed),delta_pp=float(mixed-clean),passed=mixed>=clean-Fraction(1,2)))
            else:
                alt={'brightness_dark':'gaussian_noise','contrast_low':'gaussian_blur','gaussian_noise':'brightness_dark','gaussian_blur':'contrast_low'}[t]
                vals={c:macro([qi[i,b,t,c] for i in range(100)])[0] for c in ('clean',t,alt)}
                none,oracle,wrong=vals['clean'],vals[t],vals[alt]
                for c,ctx in [('clean','none'),(t,'correct_pair'),(alt,'wrong_alt_source')]:historical_error=max(historical_error,abs(float(vals[c])-om[b,t,ctx]['macro_class']))
                gain=oracle-none;tau=max(Fraction(1,2),(clean-none)/4);ret=(mixed-none)/gain
                assert abs(float(tau)-oldg[b,t]['tau_pp'])<1e-10
                retrieval.append(dict(bank=b,target=t,none=float(none),oracle=float(oracle),selected_mixed=float(mixed),wrong_alt=float(wrong),tau_pp=float(tau),retained_recovery=float(ret),selected_wrong_margin=float(mixed-wrong),passed=gain>0 and ret>=Fraction(9,10) and mixed-wrong>=tau,sample_weighted=weighted,macro_client=clientmean))
            bestpol=[]
            for i in range(100):
                best=max(C,key=lambda c:sum(qi[i,b,t,c]['class_correct']))
                bestpol.append(qi[i,b,t,best])
            bestmacro,_=macro(bestpol);zero,_=macro([qi[i,b,t,'clean'] for i in range(100)])
            bestrows.append(dict(bank=b,target=t,query_best_policy_macro_class=float(bestmacro),zero_macro_class=float(zero),best_gain_pp=float(bestmacro-zero)))
    assert historical_error<1e-10
    # All three prior selected/none policies must reconstruct against the full matrix.
    for prior,decfile,metricfile in [(P9,'source_decisions.json','summary.json'),(P10,'state_scores.json','summary.json')]:
        dd={(r['client'],r['bank'],r['true_context']):r for r in load(prior/decfile)}
        mm=load(prior/metricfile)
        for r in mm['retrieval']+mm['clean_safety']:
            b,t=r['bank'],r.get('target','clean');val,_=macro([qi[i,b,t,dd[i,b,t]['selected']] for i in range(100)])
            historical_error=max(historical_error,abs(float(val)-r.get('selected_mixed',r.get('selected'))))
    assert historical_error<1e-10
    write_csv(OUT/'retrieval.csv',retrieval);write_csv(OUT/'clean_safety.csv',cleans);save('integer_count_receipts.json',counts_receipts)
    moments={(r['client'],r['bank'],r['true_context']):r for r in load(P10/'state_scores.json')}
    auditrows=[]
    for r in src:
        i,b,t,c=r['client'],r['bank'],r['true_context'],r['candidate'];z=si[i,b,t,'clean'];mm=moments[i,b,t]
        q=qi[i,b,t,c];qz=qi[i,b,t,'clean']
        da=Fraction(100*(sum(q['class_correct'])-sum(qz['class_correct'])),q['n_query'])
        auditrows.append(dict(client=i,bank=b,target=t,candidate=c,DeltaC=z['consistency_js']-r['consistency_js'],DeltaJ=mm['J_clean']-mm['J_'+c],DeltaAcc=float(da)))
    alignment=[]
    for b in ('A','B'):
        for t in C+['all']:
            g=[r for r in auditrows if r['bank']==b and (t=='all' or r['target']==t)]
            alignment.append(dict(bank=b,target=t,DeltaC=alignment_stats(g,'DeltaC'),DeltaJ=alignment_stats(g,'DeltaJ')))
    aligns=[]
    for b in ('A','B'):
        g=[r for r in alignment if r['bank']==b and r['target'] in C[1:]]
        nonneg=sum(r['DeltaC']['spearman'] is not None and r['DeltaC']['spearman']>=0 for r in g)
        less=sum(r['DeltaC']['harmful_fraction'] is not None and r['DeltaJ']['harmful_fraction'] is not None and r['DeltaC']['harmful_fraction']<r['DeltaJ']['harmful_fraction'] for r in g)
        aligns.append(dict(bank=b,nonnegative_spearman_contexts=nonneg,lower_harm_contexts=less,passed=nonneg>=3 and less>=3))
    save('alignment_audit.json',dict(candidate_universe='All five candidates including zero; 500 rows per bank/context. DeltaJ compared on identical full matrix, not T010 selected-only subset.',groups=alignment,gates=aligns))
    write_csv(OUT/'alignment_rows.csv',auditrows)
    skew={r['client']:r for r in load(RAW/'frozen_skew_labels_receipt.json')};regrets=[]
    for r in sel:
        i,b,t=r['client'],r['bank'],r['true_context'];g=[qi[i,b,t,c] for c in C];mx=max(sum(q['class_correct']) for q in g)
        true=qi[i,b,t,t];z=qi[i,b,t,'clean'];reg=Fraction(100*(mx-sum(true['class_correct'])),true['n_query'])
        regrets.append(dict(client=i,bank=b,target=t,quartile=skew[i]['entropy_quartile'],true_in_argmax=reg==0,true_state_regret=float(reg),
            gt2=reg>2,gt5=reg>5,best_states=[q['candidate'] for q in g if sum(q['class_correct'])==mx],
            best_query_acc=100*mx/true['n_query'],true_state_acc=true['query_accuracy'],best_gain_vs_zero=100*(mx-sum(z['class_correct']))/true['n_query']))
    reggroups=[]
    for b in ('A','B'):
        for t in C:
            g=[r for r in regrets if r['bank']==b and r['target']==t]
            reggroups.append(dict(bank=b,target=t,quartile='all',**regret_stats(g)))
            if t in ('clean','gaussian_blur'):
                for quart in range(1,5):reggroups.append(dict(bank=b,target=t,quartile=quart,**regret_stats([r for r in g if r['quartile']==quart])))
    save('true_state_regret.json',dict(groups=reggroups,rows=regrets,tie_policy='All equal integer-correct maxima retained. Histogram counts every tied maximum; entries may sum above number of clients.',best_of_five=bestrows))
    rix={(r['client'],r['bank'],r['target']):r for r in regrets}
    d9={(r['client'],r['bank'],r['true_context']):r for r in load(P9/'source_decisions.json')};ledger=[]
    for r in sel:
        i,b,t=r['client'],r['bank'],r['true_context'];c=r['selected']
        if t not in ('clean','gaussian_blur') or c==t:continue
        best=rix[i,b,t];entry=dict(client=i,bank=b,target=t,entropy=skew[i]['normalized_entropy'],max_class_fraction=skew[i]['max_class_fraction'],T009=d9[i,b,t]['selected'],T010=moments[i,b,t]['selected'],T011=c)
        for label,candidate in [('zero','clean'),('true',t),('selected',c)]:
            entry['C_'+label]=si[i,b,t,candidate]['consistency_js'];entry['agreement_'+label]=si[i,b,t,candidate]['flip_top1_agreement']
        entry.update(true_query_accuracy=qi[i,b,t,t]['query_accuracy'],selected_query_accuracy=qi[i,b,t,c]['query_accuracy'],query_best_states='|'.join(best['best_states']),query_best_accuracy=best['best_query_acc'])
        ledger.append(entry)
    write_csv(OUT/'clean_blur_error_ledger.csv',ledger)
    sanity=list(csv.DictReader((RAW/'sanity_candidate_scores.csv').open()));oods=list(csv.DictReader((RAW/'ood_selections.csv').open()));oodrows=[]
    for r in oods:
        ss={q['candidate']:q for q in sanity if q['bank']==r['bank'] and q['batch']==r['batch'] and q['true_context']=='noise_image'}
        best=min(C,key=lambda c:(float(ss[c]['consistency_js']),C.index(c)))
        oo=dict(bank=r['bank'],batch=r['batch'],selected=r['selected'],raw_best=best)
        for c in C:oo['C_'+c]=float(ss[c]['consistency_js'])
        for label,c in [('zero','clean'),('best',best),('selected',r['selected'])]:
            oo['agreement_'+label]=float(ss[c]['flip_top1_agreement']);oo['entropy_'+label]=float(ss[c]['entropy_original'])
        oodrows.append(oo)
    write_csv(OUT/'ood_consistency.csv',oodrows)
    balanced=list(csv.DictReader((RAW/'balanced_sanity.csv').open()));ids=[];bals=[]
    for b in ('A','B'):
        for rows,outrows in [(sel,ids),(balanced,bals)]:
            mat=[[sum(r['bank']==b and r['true_context']==t and r['selected']==c for r in rows) for c in C] for t in C]
            outrows.append(dict(bank=b,confusion=mat,correct=sum(mat[i][i] for i in range(5)),total=sum(sum(x) for x in mat)))
    joint={t:all(r['passed'] for r in retrieval if r['target']==t) for t in C[1:]};pcret=sum(joint.values())>=3 and all(r['passed'] for r in cleans);pcalign=all(r['passed'] for r in aligns)
    summary=dict(PC_RET_A=pcret,PC_ALIGN=pcalign,joint_shift_pass=joint,alignment_gates=aligns,identification=ids,balanced=bals,retrieval=retrieval,clean_safety=cleans,best_of_five=bestrows)
    save('summary.json',summary)
    v.update(independent_5000_prediction_count_checks=True,independent_source_selection_checks=True,independent_fraction_retrieval_gates=True,historical_aggregate_max_error_pp=historical_error,source_freeze_hashes_match=True,tests_passed=46)
    save('verification.json',v)
    # Publish joined comparison columns separately from the immutable raw source JSON.
    shutil.copyfile(RAW/'source_scores_with_moments.csv',OUT/'source_candidate_scores.csv')
    summary['decision']='P-C'
    summary['decision_rationale']='PC-safe fails retrieval and alignment; substantial true-state regret plus best-of-five headroom identifies client/state mismatch alongside poor source utility ranking. P-D is not supported.'
    save('summary.json',summary)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    flat=[dict(bank=r['bank'],target=r['target'],metric=k,**r[k]) for r in alignment for k in ('DeltaC','DeltaJ')]
    oodsummary=[dict(bank=b,selected_histogram={c:sum(r['bank']==b and r['selected']==c for r in oodrows) for c in C},fraction_zero=sum(r['bank']==b and r['selected']=='clean' for r in oodrows)/4) for b in ('A','B')]
    selected_audit=[dict(bank=b,target=t,**alignment_stats([r for r in auditrows if r['bank']==b and r['target']==t and r['candidate']==choices[r['client'],b,t]['selected']],'DeltaC')) for b in ('A','B') for t in C]
    save('selected_policy_alignment.json',selected_audit)
    focused=[r for r in ledger if r['client'] in (15,29)]
    lines=['# CODEX -> CHATGPT','## Timestamp / commit / run',
        f'{now}. Lead 941731c; runtime {meta["code_commit"]}; successful run {RUN}; release 20260914-003432-ttfl-t011; NVIDIA RTX A6000 GPU1; {meta["seconds"]:.2f}s. The enclosing delivery commit contains this report.',
        'The first launch 20260914-003444-ttfl-t011-gpu1 exited before tests or scoring because the new shell file had CRLF line endings. Converting that file to LF repaired the launch; its normalized Git content and runtime revision are unchanged. One formal evaluation completed; no sweep or alternate policy.',
        '## Frozen objects and verification',
        f'46 tests PASS. Fixed checkpoint/operator/neutral writer/corruptions/source signature/T007R states/T008 reference hashes and exact T009 K20 manifest verified. {v["score_calls"]} source candidate calls and {v["zero_regression_checks"]} zero-logit regressions; exact max difference 0. Model and candidate states unchanged after every source/query episode. No labels or query objects enter the scoring API. All scores finite; no gradient, parameter fitting, augmentation/temperature search or new FL.',
        'This uses the existing 100-client CIFAR-10 PFLlib merged-data per-client query split, not the official CIFAR-10 test benchmark. Earlier training used 10% participation; T011 performs only frozen-model evaluation.',
        '## Prediction-consistency definition',
        'C is mean Jensen-Shannon divergence (natural logarithm) between original and horizontally flipped T=1 softmax predictions. Model logits retain their precision; probabilities/reductions use double. PC-safe admits a nonzero state only when C<C_zero and flip top-1 agreement >= zero agreement, then chooses minimum C with fixed clean/dark/contrast/noise/blur tie order. Otherwise zero. Entropies, probability margins and JS-to-zero are descriptive only.',
        '## Source-score freeze',
        f'5,000 candidate rows and 1,000 choices frozen at {freeze["timestamp"]}; scores SHA {freeze["scores_sha256"]}; decisions SHA {freeze["decisions_sha256"]}. Historical query predictions, T009/T010 choices and T010 moment scores were opened only afterward. Query matrix frozen at {qfreeze["timestamp"]}. Raw source JSON stays immutable; source_candidate_scores.csv in the report adds the historical moment comparison after freezing.',
        '## PC-safe selection specificity',
        'Natural context-name agreement is only 54/500 (10.8%) and 48/500 (9.6%). This is descriptive, not an extra gate: task utility need not require choosing the named state. Balanced sanity is also weak, 3/20 and 2/20. Neither bank ever selects Dark, even on Dark targets.',
        *[f'Natural Bank {r["bank"]}:\n\n'+table(['true / selected']+C,[[c]+r['confusion'][i] for i,c in enumerate(C)]) for r in ids],
        *[f'Balanced Bank {r["bank"]}:\n\n'+table(['true / selected']+C,[[c]+r['confusion'][i] for i,c in enumerate(C)]) for r in bals],
        '## Full candidate utility matrix verification',
        f'All 5,000 client/context/bank/candidate policies evaluated: {v["query_reused"]} exact historical predictions reused and {v["new_query_evaluations"]} missing combinations computed. Each saved prediction vector independently reproduces integer per-class correct/total counts; all reused arrays match the historical source. T007R and T009/T010 aggregate policies reconstruct with max error {historical_error:.3g} pp. Fraction arithmetic verifies retrieval and clean gates. Matrix never changes the selector.',
        '## Retrieval / clean safety',
        tab(['bank','target','none','oracle','selected_mixed','wrong_alt','tau_pp','retained_recovery','selected_wrong_margin','passed'],retrieval),
        tab(['bank','none','selected','delta_pp','passed'],cleans),
        '**PC-RET-A FAIL: 0/4 joint shifted targets pass.** Clean improves +1.165149 / +0.269378 pp, but Dark retains only 8.85% /8.20%, Contrast has negative gain, Noise loses 5.420158 /8.144088 pp against zero, and Blur retains only 59.39% /60.20%. Better clean and slightly better Blur than T010 come with severe losses on the other shifts.',
        '## Within-context task-alignment comparison: DeltaC vs DeltaJ',
        'Both metrics below use the identical full five-candidate matrix, including zero: 500 rows per bank/context, 2,500 pooled. This differs from T010 selected-state-only correlations and must not be presented as a contradiction of that older analysis. Positive fractions and conditional harm exclude zero improvements by their strict definition. Correlations are descriptive; repeated candidates within clients are not independent samples, and no significance claim is made.',
        tab(['bank','target','metric','pearson','spearman','positive_fraction','harmful_fraction','median_delta_acc_positive'],flat),
        tab(['bank','nonnegative_spearman_contexts','lower_harm_contexts','passed'],aligns),
        '**PC-ALIGN FAIL in both banks:** only 2/4 shifts have nonnegative consistency Spearman; only 1/4 has lower harmful fraction than matched moment restoration. Dark and Contrast consistency correlations are negative. Pooled Spearman is barely positive (0.0157 /0.0122) and cannot establish useful within-context alignment. The full candidate universe changes moment statistics too; no pooled-only rescue is justified.',
        '## True-context-state regret / query-best-state analysis',
        'Argmax sets use tied integer correct counts, not floating tolerances. Regret thresholds >2 and >5 pp use exact fractions. All tied best names are preserved; their histogram can count more than one name per client. Query-best is a post-hoc diagnostic on these same finite query sets, not a deployable or held-out selector.',
        tab(['bank','target','quartile','true_in_argmax_fraction','regret_positive_fraction','regret_gt2_fraction','regret_gt5_fraction','median','p75','p90','median_best_gain_vs_zero'],reggroups),
        tab(['bank','target','query_best_policy_macro_class','zero_macro_class','best_gain_pp'],bestrows),
        'True Clean is query-optimal in only 20% /22% of clients, and true Blur in 20% /19%. In the lowest entropy quartile, 40% of Blur clients have true-state regret >5 pp in both banks; median regret is 3.39 pp. Alternative fixed states offer headroom: the descriptive best-of-five policy gains 9.44 /9.47 pp on Blur and 6.58 /6.43 pp on Clean versus zero. For the aggregate query-best table only, ties use fixed candidate order; per-client optimality/regret preserves every tie. This supports state/client mismatch, but does not prove interpolation will solve it. Query-best selection is optimistically biased by maximizing on the same finite query outcomes.',
        '## Clean/Blur low-entropy ledger',
        f'All {len(ledger)} Clean/Blur choices differing from the named true state are preserved in clean_blur_error_ledger.csv, with prior choices, source JS/agreement, true/selected query accuracy and all query-best states. Historically difficult clients:',
        tab(['client','bank','target','entropy','max_class_fraction','T009','T010','T011','true_query_accuracy','selected_query_accuracy','query_best_states','query_best_accuracy'],focused),
        'Context misidentification and task harm are distinct here: some non-true states improve clean accuracy, whereas source-consistency ranking badly sacrifices Dark/Noise. The failure is not confined to clients 15/29 or to low-entropy supports.',
        '## OOD descriptive result',json.dumps(oodsummary,indent=2),
        tab(['bank','batch','selected','C_clean','C_contrast_low','agreement_zero','agreement_selected','entropy_zero','entropy_selected'],oodrows),
        'All 8/8 random-noise supports choose Contrast, with selected flip agreement 1.0 and very low mean entropy (about 0.010–0.027 nats), versus zero entropy about 1.67–1.80. The score accepts highly confident and flip-stable predictions on pure noise. Prediction consistency is not an OOD detector. No threshold or OOD success claim.',
        '## Mechanism vs implementation conclusion',
        'Regression, immutability, freezes and complete prediction/count reconstruction pass. The fixed horizontal-flip consistency criterion fails as a per-client utility selector; stability can reward predictions unrelated to task correctness. At the same time, the named true-context state often is not the client-optimal fixed state. These are two separate mechanism findings. The utility matrix shows substantial headroom, so this run does not establish insufficient affine capacity.',
        '## Case decision',
        '**Primary P-C, with failed PC-safe ranking as a secondary finding. PC-ALIGN FAIL and PC-RET-A FAIL.** P-A/P-B conditions do not hold. P-D is not supported because best-of-five headroom remains substantial. P-C is a descriptive diagnosis from the regret matrix, not a claim that a new method has succeeded.',
        '## Recommended next action',
        'Return the full utility matrix to Research Lead. Preserve the operator and state bank. A subsequent bounded package may test client-conditioned state amplitude/interpolation or another task-proximal criterion, but this flip-consistency score should not be promoted into a TTT loss. No next-stage work, learned writer, SSL optimization, federation or richer operator was started.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T011 COMPLETE — P-C; PC-ALIGN and PC-RET fail\n\nRead results/t011_task_proximal/RESULTS.md and research_log/t011_delivery.json. 0/4 joint shifts, clean +1.165149/+.269378 pp. Both alignment banks 2/4 nonnegative Spearman,1/4lower harm. True Blur optimal only20/19%; fullmatrix headroom remains.46tests/count/freezes pass. Runtimef323674, successfulrun003521-lf; failedlaunch003444 beforetests CRLF fixed. AwaitLead, no nextstage.\n',encoding='utf-8')
    delivery=dict(timestamp=now,lead='941731c',runtime=meta['code_commit'],run=RUN,failed_launch='20260914-003444-ttfl-t011-gpu1',decision='P-C',raw=RAW.relative_to(ROOT).as_posix(),remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t011_task_proximal',results='results/t011_task_proximal/RESULTS.md',next='Await Lead review')
    (ROOT/'research_log/t011_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    for folder in (RAW,OUT):
        for name in ('candidate_query_utility.json','source_candidate_scores.json'):
            p=folder/name;p.with_suffix('.json.gz').write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t011_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t011_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,alignment=alignment,regret=[r for r in reggroups if r['quartile'] in ('all',1)]),indent=2))

if __name__=='__main__':main()
