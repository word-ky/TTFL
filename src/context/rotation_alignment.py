"""Frozen T023 alignment diagnostics and inherited task/regret thresholds."""
import numpy as np
from src.context.representation_gates import regret_gate,C

def average_ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);rank=np.empty(len(values));start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        rank[order[start:end]]=(start+end-1)/2+1;start=end
    return rank

def spearman(scores,utilities):
    x=average_ranks([-v for v in scores]);y=average_ranks(utilities)
    if np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])

def gates(actual,regret,context):
    passes={t:all(r['capture']>=.8 for r in actual if r['context']==t) for t in C[1:]}
    safe=all(r['delta_vs_BBSE_pp']>=-.5 for r in actual if r['context']!='clean')
    task=sum(passes.values())>=3 and safe;rg=regret_gate(regret);cg=regret_gate(context)
    return dict(ROT_TASK_A=task,task_context_pass=passes,shifted_regression_safety=safe,ROT_REGRET_A=rg,ROT_CTX_A=cg,
        diagnosis='T023-R' if task and rg['passed'] else 'T023-C' if task else 'T023-X' if rg['passed'] else 'T023-F',CTX='CTX+' if cg['passed'] else 'CTX-')
