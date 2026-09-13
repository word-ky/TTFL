"""Frozen T006 semantic decomposition, centered correlations and count audit."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_prior_receipts import ranks
from scripts.eval_pfllib_covariate_context import paired,write_csv
from src.data.covariate import TARGETS


def correlation(x,y):
    x,y=np.asarray(x),np.asarray(y)
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None


def distribution(x):
    return dict(mean=float(np.mean(x)),sd=float(np.std(x)),min=float(np.min(x)),max=float(np.max(x)))


def weighted(rows,key):
    return sum(float(r[key])*int(r['n_query']) for r in rows)/sum(int(r['n_query']) for r in rows)


def decision_flags(clean,none,correct,low,high,random_mean):
    random_mean=float(random_mean)
    headroom=clean-none;tau=max(.5,.25*headroom)
    recovery=(correct-none)/headroom if headroom>0 else None
    instance=correct-high;semantic=high-low;marginal=correct-low
    return dict(headroom_pp=headroom,tau_pp=tau,recovery_correct=recovery,
        instance_advantage_pp=instance,semantic_advantage_pp=semantic,marginal_gap_pp=marginal,
        correct_minus_random_mean_pp=correct-random_mean,
        S_A=recovery is not None and recovery>=.5 and instance>=tau and correct>random_mean,
        S_B=semantic>=tau and instance<tau,
        S_C=abs(marginal)<tau and abs(semantic)<tau)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True);p.add_argument('--identity-predictions',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();raw,out=Path(a.raw),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((raw/'raw_records.json').read_text());diag=json.loads((raw/'raw_restoration.json').read_text())
    metadata=json.loads((raw/'metadata.json').read_text());verify=json.loads((raw/'verification.json').read_text())
    assert metadata['clients']==100
    semantics=list(csv.DictReader((raw/'pairing_semantics.csv').open()))
    semantic_index={(int(r['client']),r['context']):float(r['same_label_fraction']) for r in semantics}
    index={(r['client'],r['target'],r['context']):r for r in rows}
    contexts=list(dict.fromkeys(r['context'] for r in rows))
    random_contexts=sorted(c for c in contexts if c.startswith('random_derangement_'))
    summary=[]
    for target in TARGETS:
        for ctx in contexts:
            group=[r for r in rows if r['target']==target and r['context']==ctx]
            labels=[dict(n_query=r['n_query'],score=semantic_index[(r['client'],ctx)]) for r in group] if (0,ctx) in semantic_index else []
            summary.append(dict(target=target,context=ctx,accuracy=weighted(group,'accuracy'),
                macro_client_accuracy=float(np.mean([r['accuracy'] for r in group])),
                same_label_weighted=weighted(labels,'score') if labels else None,
                same_label_macro=float(np.mean([r['score'] for r in labels])) if labels else None))
    by_summary={(r['target'],r['context']):r for r in summary}
    effects,random_stats,per_client,centered=[],[],[],[]
    for target in TARGETS:
        random_accuracy=[by_summary[target,ctx]['accuracy'] for ctx in random_contexts]
        random_semantic=[by_summary[target,ctx]['same_label_weighted'] for ctx in random_contexts]
        acc={ctx:by_summary[target,ctx]['accuracy'] for ctx in contexts}
        flags=decision_flags(metadata['clean_acc'],acc['none'],acc['correct_pair'],
                             acc['low_semantic_derangement'],acc['high_semantic_derangement'],np.mean(random_accuracy))
        deltas={name:[] for name in ('correct_low','correct_high','high_low','correct_random_mean')}
        for cid in range(100):
            randoms=[ctx for ctx in random_contexts if (cid,target,ctx) in index]
            xa=np.array([index[cid,target,ctx]['accuracy'] for ctx in randoms])
            xs=np.array([semantic_index[cid,ctx] for ctx in randoms])
            for k,ctx in enumerate(randoms):
                centered.append(dict(client=cid,target=target,context=ctx,centered_acc=float(xa[k]-xa.mean()),
                                     centered_same_label=float(xs[k]-xs.mean())))
            correct=index[cid,target,'correct_pair']['accuracy'];low=index[cid,target,'low_semantic_derangement']['accuracy'];high=index[cid,target,'high_semantic_derangement']['accuracy']
            values=dict(correct_low=correct-low,correct_high=correct-high,high_low=high-low,correct_random_mean=correct-float(xa.mean()))
            for name,value in values.items():deltas[name].append(value)
            per_client.append(dict(client=cid,target=target,n_query=index[cid,target,'none']['n_query'],
                correct=correct,low=low,high=high,random_mean=float(xa.mean()),
                same_label_low=semantic_index[cid,'low_semantic_derangement'],
                same_label_high=semantic_index[cid,'high_semantic_derangement'],
                same_label_random_mean=float(xs.mean()),**values))
        c=[r for r in centered if r['target']==target]
        x=[r['centered_same_label'] for r in c];y=[r['centered_acc'] for r in c]
        random_stats.append(dict(target=target,accuracy=distribution(random_accuracy),
            same_label_fraction_weighted=distribution(random_semantic),
            same_label_fraction_macro=distribution([by_summary[target,ctx]['same_label_macro'] for ctx in random_contexts]),
            correct_minus_random_mean=acc['correct_pair']-float(np.mean(random_accuracy)),
            fraction_random_below_correct=float(np.mean(np.array(random_accuracy)<acc['correct_pair'])),
            centered_pearson=correlation(x,y),centered_spearman=correlation(ranks(x),ranks(y)),
            centered_observations=len(x)))
        effects.append(dict(target=target,**flags,paired={name:paired(values) for name,values in deltas.items()}))
    semantic_summary=[]
    for ctx in dict.fromkeys(r['context'] for r in semantics):
        group=[r for r in semantics if r['context']==ctx]
        semantic_summary.append(dict(context=ctx,query_weighted_mean=weighted(group,'same_label_fraction'),
            client_distribution=distribution([float(r['same_label_fraction']) for r in group])))
    spread=[semantic_index[cid,'high_semantic_derangement']-semantic_index[cid,'low_semantic_derangement'] for cid in range(100)]
    restoration=[]
    for target,ctx,layer in dict.fromkeys((r['target'],r['context'],r['layer']) for r in diag):
        group=[r for r in diag if (r['target'],r['context'],r['layer'])==(target,ctx,layer)]
        valid=[r['restoration_ratio'] for r in group if r['restoration_ratio'] is not None]
        d=dict(target=target,context=ctx,layer=layer,restoration_ratio=float(np.mean(valid)) if valid else None)
        for k in ('mse_before','mse_after','gamma_abs_mean','beta_abs_mean','negative_scale_fraction','cap_fraction'):
            d[k]=float(np.mean([r[k] for r in group]))
        for k in ('gamma_abs_max','beta_abs_max','cap_fraction'):d[k+'_max']=max(r[k] for r in group)
        d['clients_cap_gt10pct']=sum(r['cap_fraction']>.1 for r in group)
        restoration.append(d)
    pred=np.load(raw/'predictions.npz');clean=np.load(a.identity_predictions)
    total=sum(len(pred[f'c{cid}_labels']) for cid in range(100))
    clean_count=sum(int((clean[f'c{cid}_none']==pred[f'c{cid}_labels']).sum()) for cid in range(100))
    exact_clean=100*clean_count/total;assert abs(exact_clean-metadata['clean_acc'])<1e-4
    audit=[]
    for effect in effects:
        target=effect['target'];counts={}
        for r in rows:
            if r['target']==target:
                counts[r['context']]=counts.get(r['context'],0)+int((pred[r['prediction_key']]==pred[f"c{r['client']}_labels"]).sum())
        exact={ctx:100*n/total for ctx,n in counts.items()}
        f=decision_flags(exact_clean,exact['none'],exact['correct_pair'],exact['low_semantic_derangement'],
                         exact['high_semantic_derangement'],np.mean([exact[c] for c in random_contexts]))
        assert all(f[k]==effect[k] for k in ('S_A','S_B','S_C'))
        audit.append(dict(target=target,query_count=total,clean_correct=clean_count,correct_counts=counts,flags=f))
    pairings=json.loads((raw/'pairings.json').read_text())
    for cid,r in pairings.items():
        labels=np.array(r['support_labels']);n=len(labels)
        scores={o:int(np.sum(labels==np.roll(labels,o))) for o in range(1,n)}
        choices=r['choices'];assert choices[0]['offset']==min(scores,key=lambda o:(scores[o],o))
        assert choices[1]['offset']==min(scores,key=lambda o:(-scores[o],o))
        random_offsets=[c['offset'] for c in choices[2:]];assert len(set(random_offsets))==min(8,n-1)
        for choice in choices:
            perm=np.array(choice['permutation']);assert np.all(perm!=np.arange(n)) and sorted(perm)==list(range(n))
            assert scores[choice['offset']]==choice['same_label_count']
    counts={key:sum(e[key] for e in effects) for key in ('S_A','S_B','S_C')}
    candidates=[key.replace('_','-') for key,count in counts.items() if count>=2]
    payload=dict(metadata=metadata,rows=summary,effects=effects,random=random_stats,
        semantic_summary=semantic_summary,semantic_spread=distribution(spread),
        numeric_criteria_counts=counts,numeric_candidates=candidates,
        interpretation_pending='Review semantic variation breadth and correlation-only S-B alternative; numerical conditions are not causal identification.',
        restoration=restoration)
    verify.update(integer_prediction_criteria_match=True,independent_clean_accuracy=exact_clean,
        independent_label_extrema_and_counts=True,random_count_per_client=8)
    for name,value in [('summary.json',payload),('verification.json',verify),('integer_prediction_audit.json',audit)]:
        (out/name).write_text(json.dumps(value,indent=2))
    for name,value in [('summary.csv',summary),('per_client.csv',per_client),('centered_random.csv',centered),('restoration_diagnostics.csv',restoration)]:write_csv(out/name,value)
    shutil.copyfile(raw/'pairing_semantics.csv',out/'pairing_semantics.csv')
    print(json.dumps(dict(numeric_candidates=candidates,counts=counts,effects=effects,random=random_stats,
                         semantics=semantic_summary,spread=distribution(spread)),indent=2))


if __name__=='__main__':main()
