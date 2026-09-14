"""T024 exact semantic compositions and oracle-bank geometry; no model code."""
import hashlib,math
from fractions import Fraction as F
import numpy as np
LAMBDAS=(F(1,4),F(1,2),F(3,4),F(1))
SALTS=tuple(f'T013-S{i}' for i in range(4))

def tv(p,q):return sum((abs(x-y) for x,y in zip(p,q)),F())/2
def js_distance(p,q):
    m=[(x+y)/2 for x,y in zip(p,q)]
    divergence=sum(float(x)*math.log2(float(x/z)) for a in (p,q) for x,z in zip(a,m) if x)/2
    return math.sqrt(max(0.,divergence))
def counter_state(p):
    candidates=[([p[(c-k)%10] for c in range(10)],k) for k in range(1,10)]
    q,k=max(candidates,key=lambda x:(tv(p,x[0]),-x[1]))
    return q,k,tv(p,q)==0
def interpolate(p,q,lam):return [(1-lam)*x+lam*y for x,y in zip(p,q)]
def pairing(salt,half,clients):
    order=sorted(clients,key=lambda i:hashlib.sha256(f'T024|pair|{salt}|{half}|{i}'.encode()).digest())
    return {i:order[(k+1)%len(order)] for k,i in enumerate(order)}
def utility(pi,matrix):return [sum((pi[c]*matrix[c][s] for c in range(10)),F()) for s in range(5)]
def argmax(v):return [i for i,x in enumerate(v) if x==max(v)]
def mean(v):return sum(v,F())/len(v)

def evaluate(pi0,current,mismatch,matrix):
    h=utility(pi0,matrix);u=utility(current,matrix);m=utility(mismatch,matrix)
    ha,ua,ma=argmax(h),argmax(u),argmax(m);hs,us,ms=ha[0],ua[0],ma[0]
    ordered=sorted(u,reverse=True)
    return dict(U_hist=h,U_current=u,U_mismatch=m,s_hist=hs,s_current=us,s_mismatch=ms,s_zero=0,hist_argmax=ha,current_argmax=ua,mismatch_argmax=ma,
        oracle_current=u[us],historical=u[hs],mismatched=u[ms],zero=u[0],stale_regret=u[us]-u[hs],mismatch_regret=u[us]-u[ms],oracle_vs_zero=u[us]-u[0],historical_minus_zero=u[hs]-u[0],
        state_switch=hs!=us,argmax_overlap=bool(set(ha)&set(ua)),top_two_margin=ordered[0]-ordered[1])

def gate_summary(rows):
    banks={}
    for b in ('A','B'):
        active=[r for r in rows if r['bank']==b and not r['degenerate']];severe=[r for r in active if r['lambda']==F(1)]
        assert len(severe)>0
        avg=mean([r['stale_regret'] for r in severe]);gt=F(sum(r['stale_regret']>=F(1,100) for r in severe),len(severe));switch=F(sum(r['state_switch'] for r in severe),len(severe))
        head=avg>=F(1,100) and gt>=F(3,10) and switch>=F(3,10)
        salt_means={s:mean([r['mismatch_regret'] for r in severe if r['salt']==s]) for s in SALTS};salt_pass={s:v>=F(1,200) for s,v in salt_means.items()}
        mismatch_fraction=F(sum(r['mismatch_regret']>=F(1,100) for r in severe),len(severe));specific=sum(salt_pass.values())>=3 and mismatch_fraction>=F(1,4)
        trend=[mean([r['stale_regret'] for r in active if r['lambda']==lam]) for lam in LAMBDAS];monotonic=all(y>=x-F(1,10**12) for x,y in zip(trend,trend[1:]));gap=trend[-1]-trend[0]
        banks[b]=dict(non_degenerate_severe_cells=len(severe),headroom=head,stale_mean=avg,stale_ge_1pp_fraction=gt,switch_rate=switch,specificity=specific,mismatch_salt_means=salt_means,mismatch_salt_pass=salt_pass,mismatch_ge_1pp_fraction=mismatch_fraction,trend=monotonic and gap>=F(1,200),trend_means=trend,trend_nondecreasing=monotonic,severe_minus_mild=gap)
    h=all(v['headroom'] for v in banks.values());s=all(v['specificity'] for v in banks.values());t=all(v['trend'] for v in banks.values())
    return dict(SEM_HEADROOM_A=h,SEM_SPEC_A=s,SEM_TREND_A=t,banks=banks,diagnosis='T024-S' if h and s and t else 'T024-H' if h else 'T024-N')

def serial(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
