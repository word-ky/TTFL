"""Independent original-model, LAPACK probe and exact integer/utility replay."""
import argparse,csv,hashlib,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
from scipy.linalg import lstsq
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t023_common import *
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.eval.common import state_hash
from eval_t015_semantic_mixture import state_digest
from src.context.matched_channel import exact_utilities
from eval_t017_bootstrap import macro

def rows(p):return list(csv.DictReader(Path(p).open()))
def probe(h,split,y):
    err=[];ranks=[]
    for train,val in (split,split[::-1]):
        x=np.column_stack([h[train].reshape(40,512).astype(np.float64),np.ones(40)])
        z=np.column_stack([h[val].reshape(40,512).astype(np.float64),np.ones(40)])
        w,_,rank,_=lstsq(x,y[train].reshape(40,4),cond=1e-12,lapack_driver='gelsd')
        assert np.isfinite(w).all();err.append(float(np.mean((z@w-y[val].reshape(40,4))**2)));ranks.append(rank)
    return float(np.mean(err)),err,ranks

def main():
    ap=argparse.ArgumentParser()
    for k in ('project','scoring','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);score=Path(a.scoring);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    ss=load(score/'scoring_start.json');phase=Path(ss['phase_a']);pf=load(phase/'phaseA_choices_freeze.json');ext=Path(pf['extraction']);ef=load(ext/'extraction_freeze.json')
    assert sha(phase/'phaseA_choices_freeze.json')==ss['phase_a_freeze_sha256'] and sha(ext/'extraction_freeze.json')==pf['extraction_freeze_sha256']
    for folder,f in [(phase,pf),(ext,ef)]:assert f['status']=='PASS' and all(sha(folder/name)==digest for name,digest in f['hashes'].items())
    assert all(pf[k] is False for k in ('target_class_labels_used','privileged_utility_loaded','query_outcomes_scored','model_parameters_updated'))
    summary=load(score/'summary.json');assert summary['new_query_forwards']==summary['new_model_forwards']==0 and summary['model_parameters_updated'] is False
    inputs=load(score/'input_hashes.json');assert all(sha(path)==digest for path,digest in inputs.items())
    fs=load(ext/'fold_membership.json');d=dict(np.load(phase/'phaseA_scores.npz'));scr=np.load(phase/'scramble_scores.npz')['scores']
    old=p/'runs/20260914-183139-ttfl-t021-extract-gpu1/artifacts/t021_frozen_representation_observability';new=p/'runs/20260914-193443-ttfl-t022-extract-gpu1/artifacts/t022_state_response_semantics'
    cached={r:{**dict(np.load(old/f'support_{r}.npz')),**dict(np.load(new/f'new_{r}.npz'))} for r in ('H','L')}
    base=Path('/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10');p7=p/'runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
    model=ContextFedAvgCNN(3,10,1600).cuda().eval();model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True));initial=state_hash(model);assert initial==ef['model_hash']
    st=torch.load(p7/'states.pt',map_location='cuda',weights_only=True);states={b:{'clean':new_state(model),**{t:st[f'{b}|{t}|correct_pair'] for t in C[1:]}} for b in B}
    assert {b:{t:state_digest(v) for t,v in bank.items()} for b,bank in states.items()}==ef['state_digests']
    captured=[];hook=model.fc.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()))
    rotations={(r['client'],r['context'],r['rotation']):r for r in gzload(ext/'rotation_arithmetic.json.gz')};y=np.tile(np.eye(4),(20,1)).reshape(20,4,4)
    hchecks=rotchecks=cachechecks=arithmetic=probechecks=scramblechecks=choicechecks=0;maxh=maxprobe=maxscramble=0.
    for i in range(100):
        ids=fs[str(i)]['support_ids'];order=sorted(range(20),key=lambda k:hashlib.sha256(f'T023|fold|{i}|{ids[k]}'.encode()).hexdigest());split=np.array(order[:10]),np.array(order[10:])
        assert split[0].tolist()==fs[str(i)]['A_positions'] and split[1].tolist()==fs[str(i)]['B_positions'] and not set(split[0])&set(split[1])
        hd=np.load(ext/f'H_{i:03d}.npz');pixels=np.load(ext/f'pixels_{i:03d}.npz')
        for ti,t in enumerate(C):
            tx=pixels[t]
            for r in range(4):
                record=rotations[i,t,r];assert rawsha(tx)==record['input_sha256'] and rawsha(np.rot90(tx,r,axes=(-2,-1)))==record['view_sha256'];arithmetic+=20
            for bi,b in enumerate(B):
                grid=hd[f'{b}|{t}'];refs=[]
                for si,s in enumerate(C):
                    key=f'{i}|{b}|{t}|{s}';np.testing.assert_array_equal(grid[si,:,0],cached['H'][key]);cachechecks+=20
                    if i in (0,24,49,74,99):
                        for r in ([0,1,2,3] if si==0 else [0]):
                            view=torch.as_tensor(np.rot90(tx,r,axes=(-2,-1)).copy()).cuda()
                            with torch.no_grad():z=model(view,states[b][s]).cpu().numpy()
                            h=captured.pop();delta=float(np.max(np.abs(h-grid[si,:,r])));assert delta<=1e-6;maxh=max(maxh,delta)
                            if r==0:assert np.array_equal(z.argmax(1),cached['L'][key].argmax(1));hchecks+=20
                            else:rotchecks+=20
                    if i<40:
                        v,err,rank=probe(grid[si],split,y);delta=max(abs(v-d['scores'][i,bi,ti,si]),float(np.max(np.abs(np.array(err)-d['errors'][i,bi,ti,si]))));assert delta<=1e-10
                        np.testing.assert_array_equal(rank,d['ranks'][i,bi,ti,si]);maxprobe=max(maxprobe,delta);refs.append(v);probechecks+=1
                saved=d['scores'][i,bi,ti];mask=sum(1<<k for k in range(5) if saved[k]==min(saved));assert np.argmin(saved)==d['current_choices'][i,bi,ti] and mask==d['current_masks'][i,bi,ti];choicechecks+=1
                if refs:assert np.argmin(refs)==d['current_choices'][i,bi,ti]
                assert d['clean_choices'][i,bi,ti]==d['current_choices'][i,bi,0] and d['clean_masks'][i,bi,ti]==d['current_masks'][i,bi,0]
                if i<20:
                    for r in range(16):
                        lab=[]
                        for sid in ids:
                            seed=int(hashlib.sha256(f'T023|scramble|{i}|{sid}|{r}'.encode()).hexdigest(),16);perm=np.random.Generator(np.random.PCG64(seed)).permutation(4);lab.append(np.eye(4)[perm])
                        v,_,_=probe(grid[0],split,np.array(lab));delta=abs(v-scr[i,bi,ti,r]);assert delta<=1e-10;maxscramble=max(maxscramble,delta);scramblechecks+=1
        if i%20==19:print('T023_INDEPENDENT_CLIENT',i+1,flush=True)
    hook.remove();assert hchecks>=5000 and rotchecks>=2000 and probechecks>=2000 and scramblechecks>=3200 and state_hash(model)==initial
    assert {b:{t:state_digest(v) for t,v in bank.items()} for b,bank in states.items()}==ef['state_digests']
    for r in rows(phase/'scramble_cells.csv'):
        i=int(r['client']);bi=B.index(r['bank']);ti=C.index(r['context']);v=scr[i,bi,ti];true=d['scores'][i,bi,ti,0]
        assert float(r['true_score'])==true and (r['true_beats_scramble_median']=='True')==bool(true<np.median(v))
        for key,q in [('p05',.05),('median',.5),('p95',.95)]:assert float(r['scramble_'+key])==np.quantile(v,q)
    for r in rows(phase/'scramble_sanity.csv'):
        bi=B.index(r['bank']);ti=C.index(r['context']);means=scr[:,bi,ti].mean(0)
        assert float(r['true_score_mean'])==d['scores'][:,bi,ti,0].mean()
        for key,q in [('p05',.05),('median',.5),('p95',.95)]:assert float(r['scramble_client_mean_'+key])==np.quantile(means,q)
        assert float(r['fraction_clients_true_beats_own_scramble_median'])==np.mean(d['scores'][:,bi,ti,0]<np.median(scr[:,bi,ti],axis=1))
    # Privileged verification starts only after the independent support audit above.
    p13=p/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';p14=p/'results/t014_class_conditional_factorization';p15=p/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture';p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics';p18=p/'runs/20260914-132634-ttfl-t018r2-science/artifacts/t018r2_constrained_prevalence'
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];totals=hc['total'][0].sum((0,1));truth={r['client']:r for r in load(p16/'support_truth.json')}
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    point={(r['client'],r['bank'],r['context'],r['salt'],r['train_half']):r['state'] for r in gzload(p18/'actual_oracle_context_episodes.json.gz')}
    counts={};regs={};utils={};episodes=gzload(score/'actual_regret_episodes.json.gz')
    for r in episodes:
        i=r['client'];bi=B.index(r['bank']);ti=C.index(r['context']);si=S.index(r['salt']);h=r['train_half'];state=r['state'];key=r['policy'],r['bank'],r['context'],r['salt']
        expected=point[i,r['bank'],r['context'],r['salt'],h] if r['policy']=='P' else d['current_choices' if r['policy']=='ROT-CURRENT' else 'clean_choices'][i,bi,ti];assert state==expected
        if key not in counts:counts[key]=np.zeros(10,dtype=np.int64);regs[key]=[]
        counts[key]+=cc[si,bi,i,ti,1-h,state];v,arg=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],templates[r['salt'],r['bank'],h,i][ti]);reg=max(v)-v[state]
        assert str(reg)==r['true_template_regret_exact'] and (state in arg)==r['optimal_set'] and (state==arg[0])==r['canonical'];regs[key].append((i,h,reg));utils[i,r['bank'],r['context'],r['salt'],h]=v
    zero={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    p00={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    bbse={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p16/'integer_count_receipts.json') if r['policy']=='BBSE-S-01'}
    for r in load(score/'integer_count_receipts.json'):
        key=r['policy'],r['bank'],r['context'],r['salt'];np.testing.assert_array_equal(counts[key],r['class_correct']);np.testing.assert_array_equal(totals,r['class_total'])
    aa=rows(score/'actual_aggregate.csv');pa=rows(score/'paired_regret_context.csv')
    expected={(pol,b,t,s) for pol in ('ROT-CURRENT','ROT-CLEAN-SURROGATE','P') for b in B for t in C for s in S}
    assert len(aa)==120 and {(r['policy'],r['bank'],r['context'],r['salt']) for r in aa}==expected
    assert set(counts)==expected and len(episodes)==24000 and all(len(v)==200 and len({(i,h) for i,h,_ in v})==200 for v in regs.values())
    assert len(load(score/'integer_count_receipts.json'))==120
    expected_paired={(pol,comp,b,t,s) for pol,comp in [('ROT-CURRENT','vs_P'),('ROT-CLEAN-SURROGATE','vs_P'),('ROT-CURRENT','current_vs_clean')] for b in B for t in C for s in S}
    assert len(pa)==120 and {(r['policy'],r['comparison'],r['bank'],r['context'],r['salt']) for r in pa}==expected_paired
    historical={(r['bank'],r['context'],r['salt']):r['class_correct'] for r in load(p18/'integer_count_receipts.json') if r['mode']=='01'}
    for (b,t,s),v in historical.items():np.testing.assert_array_equal(counts['P',b,t,s],v)
    for r in aa:
        key=r['policy'],r['bank'],r['context'],r['salt'];bk=r['salt'],r['bank'],r['context'];acc=macro(counts[key],totals);rv=[v for _,_,v in regs[key]]
        assert str(acc)==r['macro_class_exact'] and str((acc-zero[bk])/(p00[bk]-zero[bk]))==r['capture_exact'] and float(acc-bbse[bk])==float(r['delta_vs_BBSE_pp'])
        assert str(sum(rv,Fraction())/200)==r['true_template_regret_exact'] and np.quantile([float(v) for v in rv],.9)==float(r['regret_p90'])
    for r in pa:
        base='P' if r['comparison']=='vs_P' else 'ROT-CLEAN-SURROGATE';tail=r['bank'],r['context'],r['salt'];nv=regs[(r['policy'],)+tail];ov=regs[(base,)+tail]
        assert str(sum((v for _,_,v in nv),Fraction())/200)==r['new_mean_exact'] and str(sum((v for _,_,v in ov),Fraction())/200)==r['baseline_mean_exact']
        assert np.quantile([float(v) for _,_,v in nv],.9)==float(r['new_p90']) and np.quantile([float(v) for _,_,v in ov],.9)==float(r['baseline_p90'])
        delta=[sum((v for j,h,v in nv if j==i),Fraction())-sum((v for j,h,v in ov if j==i),Fraction()) for i in range(100)]
        assert sum(v<0 for v in delta)==int(r['clients_better']) and sum(v==0 for v in delta)==int(r['clients_equal']) and sum(v>0 for v in delta)==int(r['clients_worse'])
    rr=gzload(score/'rank_correlations.json.gz')
    assert len(rr)==16000 and len(rows(score/'rank_summary.csv'))==20 and len(load(score/'state_choice_histograms.json'))==20
    for r in rr:
        i=r['client'];bi=B.index(r['bank']);ti=C.index(r['context']) if r['policy']=='ROT-CURRENT' else 0
        x=rankdata(-d['scores'][i,bi,ti]);y=rankdata(utils[i,r['bank'],r['context'],r['salt'],r['train_half']]);rho=None if np.ptp(x)==0 or np.ptp(y)==0 else float(np.corrcoef(x,y)[0,1])
        assert rho==r['spearman']
    for r in rows(score/'rank_summary.csv'):
        group=[z['spearman'] for z in rr if all(z[k]==r[k] for k in ('policy','bank','context'))];v=[x for x in group if x is not None];assert len(v)==int(r['defined']) and len(group)-len(v)==int(r['undefined'])
        if v:
            for key,q in [('p25',.25),('median',.5),('p75',.75)]:assert np.quantile(v,q)==float(r[key])
            assert np.mean(np.array(v)>0)==float(r['fraction_positive'])
    for r in load(score/'state_choice_histograms.json'):
        bi=B.index(r['bank']);ti=C.index(r['context']);prefix='current' if r['policy']=='ROT-CURRENT' else 'clean';choice=d[prefix+'_choices'][:,bi,ti];mask=d[prefix+'_masks'][:,bi,ti]
        np.testing.assert_array_equal(np.bincount(choice,minlength=5),r['client_state_counts']);assert sum(int(v).bit_count()>1 for v in mask)==r['exact_tie_cells']
    g=load(score/'diagnostic_gates.json');current=[r for r in aa if r['policy']=='ROT-CURRENT'];passes={t:all(float(r['capture'])>=.8 for r in current if r['context']==t) for t in C[1:]};safe=all(float(r['delta_vs_BBSE_pp'])>=-.5 for r in current if r['context']!='clean');task=sum(passes.values())>=3 and safe
    assert passes==g['task_context_pass'] and safe==g['shifted_regression_safety'] and task==g['ROT_TASK_A']
    for comp,name in [('vs_P','ROT_REGRET_A'),('current_vs_clean','ROT_CTX_A')]:
        group=[r for r in pa if r['policy']=='ROT-CURRENT' and r['comparison']==comp];passed={t:all(float(r['baseline_mean'])>0 and float(r['new_mean'])<=.85*float(r['baseline_mean']) and float(r['new_p90'])<=1.05*float(r['baseline_p90']) for r in group if r['context']==t) for t in C[1:]}
        assert passed==g[name]['context_pass'] and (sum(passed.values())>=3)==g[name]['passed']
    regret=g['ROT_REGRET_A']['passed'];diagnosis='T023-R' if task and regret else 'T023-C' if task else 'T023-X' if regret else 'T023-F';assert diagnosis==g['diagnosis'] and g['CTX']==('CTX+' if g['ROT_CTX_A']['passed'] else 'CTX-')
    assert all(sha(path)==digest for path,digest in inputs.items())
    save(out/'remote_artifact_manifest.json',[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for folder in (ext,phase,score) for path in folder.iterdir() if path.is_file()])
    save(out/'independent_verification.json',dict(status='PASS',runtime=a.commit,scoring=str(score),unrotated_original_model_samples=hchecks,rotated_original_model_samples=rotchecks,all_unrotated_cache_samples=cachechecks,exact_quarter_turn_sample_tensors=arithmetic,max_H_difference=maxh,cross_fitted_probe_scores=probechecks,individual_probe_fits=2*probechecks,max_probe_score_difference=maxprobe,scramble_cell_replicas=scramblechecks,scramble_cells=scramblechecks//16,max_scramble_score_difference=maxscramble,all_scramble_summaries_verified=True,choice_cells=choicechecks,exact_regret_episodes=len(episodes),integer_rows=len(counts),rank_correlations=len(rr),all_phase_a_hashes_verified=True,all_gates_and_paired_counts_verified=True,model_and_state_hashes_unchanged=True,new_query_forwards=0,seconds=time.time()-start))
    print('T023_INDEPENDENT_PASS',load(out/'independent_verification.json'),flush=True)

if __name__=='__main__':main()
