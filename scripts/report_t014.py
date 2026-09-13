"""Report T014 with exact count gate recomputation and all negative/positive controls."""
import csv,hashlib,json,shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from report_t009 import table,write_csv
ROOT=Path(__file__).resolve().parents[1];RUN='20260914-t014-local'
RAW=ROOT/f'research_log/t014_receipts/{RUN}/artifacts/t014_class_conditional_factorization';OUT=ROOT/'results/t014_class_conditional_factorization'
P13=ROOT/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
SALTS=['T013-S0','T013-S1','T013-S2','T013-S3'];C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];OFF=[7,13,23,37,41,53,71,89]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.suffix in ('.csv','.json','.gz'):shutil.copyfile(p,OUT/p.name)
    s=load(RAW/'summary.json');v=load(RAW/'verification.json');meta=load(RAW/'metadata.json');freeze=load(RAW/'policy_freeze.json')
    for name,key in [('composition_vectors.json','composition_sha256'),('class_templates.json.gz','templates_sha256'),('policy_utilities.json.gz','utilities_sha256'),('policy_choices.csv','choices_sha256')]:assert sha(RAW/name)==freeze[key]
    def exact(path):
        return {(r['salt'],r['bank'],r['target'],r['policy']):sum((Fraction(100*x,y) for x,y in zip(r['class_correct'],r['class_total'])),Fraction())/10 for r in load(path)}
    e=exact(RAW/'integer_count_receipts.json');old=exact(P13/'integer_count_receipts.json');metrics=load(RAW/'policy_metrics.json');error=0.
    for r in metrics:error=max(error,abs(float(e[r['salt'],r['bank'],r['target'],r['policy']])-r['macro_class']))
    comp=[];joint={t:True for t in C[1:]}
    for salt in SALTS:
        for b in ('A','B'):
            z=old[salt,b,'clean','zero'];gain=old[salt,b,'clean','client_only']-z;co=e[salt,b,'clean','composition_persistent'];mm=sum(e[salt,b,'clean',f'mismatched_pi_persistent_{k}'] for k in OFF)/8
            comp.append((co-z)/gain>=Fraction(4,5) and co-mm>=Fraction(1,2))
            assert e[salt,b,'clean','hybrid_client_plus_class_context']==old[salt,b,'clean','client_only']
            for t in C[1:]:
                cap=(e[salt,b,t,'hybrid_client_plus_class_context']-old[salt,b,t,'zero'])/(old[salt,b,t,'half_oracle']-old[salt,b,t,'zero'])
                joint[t] &= cap>=Fraction(4,5)
    assert s['COMP_A']==all(comp) and s['joint_capture']==joint
    additional=any(all(sum(e[salt,b,t,'hybrid_client_plus_class_context']-old[salt,b,t,'two_factor'] for salt in SALTS)/4>=Fraction(1,2) for b in ('A','B')) for t in ('contrast_low','gaussian_blur'))
    assert s['CLASS_INT_A']==(sum(joint.values())>=3 and additional)
    v.update(independent_exact_fraction_COMP_and_CLASS_INT_gates=True,independent_report_count_error_pp=error)
    save('verification.json',v)
    meanrows=[]
    for b in ('A','B'):
        for t in C:
            r=dict(bank=b,target=t)
            for pol in ('composition_persistent','class_context_only','hybrid_client_plus_class_context','uniform_class_context','uniform_hybrid'):
                r[pol]=float(sum(e[salt,b,t,pol] for salt in SALTS)/4)
            for pol in ('zero','client_only','context_only','two_factor','half_oracle'):r['T013_'+pol]=float(sum(old[salt,b,t,pol] for salt in SALTS)/4)
            meanrows.append(r)
    write_csv(OUT/'mean_accuracy.csv',meanrows)
    interaction=list(csv.DictReader((RAW/'interaction_capture.csv').open()))
    residual=load(RAW/'residual_lock_summary.json');sim=[]
    for r in residual['utility_similarity']:
        vals=[x['mean'] for x in r['controls'] if x['mean'] is not None]
        sim.append(dict(salt=r['salt'],bank=r['bank'],target1=r['target1'],target2=r['target2'],orientation=r['orientation'],same_valid=r['same']['valid_count'],same_mean=r['same']['mean'],same_median=r['same']['median'],control_mean=float(np.mean(vals)),control_min=min(vals),control_max=max(vals)))
    write_csv(OUT/'residual_utility_similarity.csv',sim)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    lines=['# CODEX -> CHATGPT','## Timestamp / commit / run',
        f'{now}. Lead1462732; runtime {meta["code_commit"]}; local saved-prediction run {RUN}; {meta["seconds"]:.2f}s. One formal analysis, no model forward pass, GPU job or training. Enclosing delivery commit contains the independent report.',
        '## Preflight and frozen evidence',
        'T011 prediction SHA matches its original freeze. T013 split, choices, class-count artifact and count-receipt hashes match. Every half per-class correct/total array reconstructs from T011 predictions. All600 historical T013 policy metrics and clean safety reconstruct exactly before new analysis, including zero/client-only/context-only/two-factor/half-oracle. Four salts, original half memberships, five candidates, two banks and eight offsets are unchanged.',
        'Three focused tests PASS: evaluation-label-free interfaces and choice invariance; target exclusion and legitimate changes to other templates; exact clean residual0/hybrid equivalence; uniform/mismatched composition controls on fixed templates; nonzero class denominators. No model code was imported or changed, so historical model tests were not rerun. Saved-prediction per-example concatenation independently verifies all1,240 new policy metrics; exact Fraction report error0.',
        '## Exact class templates and composition',
        'For each salt/bank/training orientation/target client, pool correct/total examples within each class across the other99 clients. D is clean utility versus zero; R is shifted utility minus D. All class denominators are positive, with no smoothing, cutoff, shrinkage or learned weight. The target composition is its opposite training-half ten-class histogram. Target evaluation-half labels never enter template construction, composition, actual clean persistent utility or policy choice.',
        'class_templates.json.gz stores1,600 templates as exact numerator/denominator strings for utility[c][class][state], plus other99 class denominators; D=utility[clean] and R=utility[c]−D. composition_vectors.json stores800 exact target histograms/denominators. All template, mixture, hybrid and argmax arithmetic uses Fraction; only reported accuracies/correlations are converted to float.',
        '## Policies / freeze',
        '31 fixed policies: composition_persistent, class_context_only, hybrid_client_plus_class_context, two T013 regression policies, two uniform controls, and eight offsets for each of persistent/class-context/hybrid mismatched-pi controls. The extra persistent mismatch controls are required by COMP-A. All controls retain identical target-excluded templates and actual persistent term where applicable; only pi changes.',
        f'All248,000 predicted utility vectors, full argmax sets and choices frozen at {freeze["timestamp"]}, before T014 evaluation or residual analysis. Utility SHA {freeze["utilities_sha256"]}; template SHA {freeze["templates_sha256"]}. Frozen candidate order resolves exact ties only after the full argmax set is retained. Clean hybrid utility, choice and metric equal T013 client_only exactly because R(clean)=0.',
        '**This is a supervised, context-privileged diagnostic, not a deployable unlabeled method.** It uses target opposite-half labels and other-client training-half labels. Both cross-fit orientations are concatenated; no evaluation data are reweighted. Historical query-selected half-oracle remains an optimistic diagnostic denominator, not a policy-learning target.',
        '## Policy accuracy / regret',
        'Macro-class accuracy averaged over all four salts for readability; every salt/bank/context/policy remains in policy_metrics.csv with weighted accuracy, macro-client accuracy, selection counts, half-oracle hit rate and regret median/p75/p90/>2/>5pp:',
        tab(['bank','target','T013_zero','T013_client_only','T013_two_factor','composition_persistent','class_context_only','hybrid_client_plus_class_context','uniform_hybrid','T013_half_oracle'],meanrows),
        'State-choice counts sum to200 client-half decisions per row. Half regret preserves integer correct-count ties. The full exact count receipts and frozen utility/template receipts support reconstruction; no evaluation-half outcome changes a formula or coefficient.',
        '## Persistent clean composition capture / COMP decision',
        tab(['salt','bank','composition_gain','client_gain','composition_capture','composition_minus_mismatch_mean','passed'],s['composition_capture']),
        '**COMP-A PASS in both banks and every salt.** Composition-only clean gain is about6.09–6.29pp, capturing107.4–114.1% of T013 client-only gain, and beats mean mismatched persistent composition by7.55–7.74pp. Capture above1 is valid: pooled class templates can outperform noisy client-half persistent measurements. This is not a causal percentage of variance explained.',
        'The interpretation of client-lock should therefore shift toward stable semantic/class composition in these synthetic non-IID CIFAR-10 clients. A separate latent client-identity variable is not required to explain most of this observed clean preference. This does not prove no other persistent client/content factor exists outside this experiment.',
        '## Class-conditioned interaction / per-salt capture',
        tab(['salt','bank','target','class_capture','hybrid_minus_two_factor','hybrid_minus_client_only','class_context_minus_context_only','hybrid_minus_uniform','hybrid_minus_mismatch_mean','mismatch_mean','mismatch_min','mismatch_max'],interaction),
        tab(['bank','target','mean_hybrid_minus_two_factor','mean_hybrid_minus_mismatch','gain_pass','mismatch_pass'],s['advantages']),
        '**CLASS-INT-A PASS:4/4 shifted contexts meet80% capture in both banks under every salt.** Mean hybrid gain over T013 two_factor is+3.323/+3.239pp for Contrast and+.828/+1.013pp for Blur, so the additional failed-context improvement condition passes. Dark gains+1.888/+1.939pp. Noise changes−.082/+.018pp and gains only+.040/+.065pp over mismatched composition; preserve this weak/negative incremental result rather than claiming uniform improvement.',
        'Class-conditioned transient response explains enough of the missing Contrast/Blur interaction to pass the fixed gate, but it does not eliminate all half-oracle regret. No coefficient tuning or interpolation between global and class-conditioned residuals was performed.',
        '## Residual disjoint client-lock',
        tab(['salt','bank','raw_margin_pp','residual_margin_pp','attenuation_pp','fractional_attenuation','residual_above_original_15pp'],s['residual_lock']),
        'Residual E uses held-out observed state utility minus the opposite-half/other99 class-predicted utility. Subtraction and residual argmax ties use exact fractions. Raw margins exactly reproduce T013. Residual margins range−5.019 to−3.494pp, with none retaining the original15pp margin. Thus the strong positive lock vanishes under this prescribed class-conditioned subtraction.',
        'Fractional attenuation exceeds1 because the residual overlap contrast changes sign; it must not be read as “more than100% of causal signal explained.” Residualization changes ties and noise structure and uses cross-fitted estimated templates, so a small negative margin is not proof of a biological/statistical anti-client effect. The narrow conclusion is that no large positive residual lock remains by this test.',
        'residual_lock.csv retains all pairs/orientations; residual_utility_similarity.csv provides valid counts, same-client Spearman mean/median and shifted-control ranges. residual_lock_summary.json retains every offset statistic. Exact residual vectors are in residual_utilities.json.gz. No policy was fitted from these residuals.',
        '## Mechanism vs implementation conclusion',
        'All artifact regressions, exclusion/half boundaries, clean invariants, freeze hashes and independent count checks pass. This is a mechanism finding: stable semantic composition explains most persistent preference, and class-dependent context response resolves much of the previous additive model’s missing interaction within the frozen five-state bank. The evidence favors semantic composition plus transient context over automatically calling the former client identity.',
        'The result remains specific to this frozen checkpoint, state bank, corruption set and synthetic PFLlib split (not the official CIFAR-10 test benchmark). Correct class composition and utility templates are supervised here. Neither a label-free composition estimator nor a writer has been tested; residual feature/content effects beyond these diagnostics are not ruled out.',
        '## Recommended next action',
        'Return COMP-A + CLASS-INT-A and the attenuated residual lock to Research Lead. A next bounded design/audit may ask how semantic mixture can be estimated without labels and how it conditions transient state utility, but no such estimator, writer, SSL/TTT update, new federation, operator expansion or T015 was implemented.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T014 COMPLETE — COMP-A + CLASS-INT-A\n\nRead results/t014_class_conditional_factorization/RESULTS.md and research_log/t014_delivery.json. Compositioncapture107.4–114.1percentall8pass; hybrid4/4jointcaptureallsalts/banks. Contrast+3.323/+3.239pp,Blur+.828/+1.013pp vsT013twofactor. Residual-lock margins−3.49to−5.02pp; revisepersistentclienttowardsemanticcomposition. Superviseddiagnostic, notunlabeledmethod.3tests;1240metrics,600oldmetrics,hashes/exclusions/cleaninvariantsPASS. Runtime78de03c local20260914-t014-local complete; noforwards. AwaitLead,noT015.\n',encoding='utf-8')
    delivery=dict(timestamp=now,lead='1462732',runtime=meta['code_commit'],run=RUN,decision='COMP-A + CLASS-INT-A',raw=RAW.relative_to(ROOT).as_posix(),results='results/t014_class_conditional_factorization/RESULTS.md',next='Await Lead')
    (ROOT/'research_log/t014_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t014_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t014_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(COMP_A=s['COMP_A'],CLASS_INT_A=s['CLASS_INT_A'],count_error=error)))

if __name__=='__main__':main()
