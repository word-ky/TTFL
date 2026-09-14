"""T018 fixed projected-gradient simplex least squares, no statistical tuning."""
import numpy as np
from src.context.confusion_prevalence import project_simplex


def cls_s(C,q):
    C=np.asarray(C,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    L=float(np.linalg.norm(C,2)**2);eta=1/L
    initial=project_simplex(np.linalg.pinv(C)@q);pi=initial.copy()
    converged=False
    for iteration in range(1,20001):
        gradient=C.T@(C@pi-q)
        next_pi=project_simplex(pi-eta*gradient)
        delta=float(np.max(np.abs(next_pi-pi)));pi=next_pi
        gradient=C.T@(C@pi-q)
        residual=float(np.max(np.abs(pi-project_simplex(pi-eta*gradient)))/eta)
        if delta<=1e-12 and residual<=1e-10:
            converged=True;break
    objective=float(.5*np.sum((C@pi-q)**2));initial_objective=float(.5*np.sum((C@initial-q)**2))
    return pi,dict(iterations=iteration,converged=converged,cap_hit=not converged,L=L,eta=eta,max_step=delta,projected_gradient_residual=residual,
        objective=objective,initial_objective=initial_objective,active_classes=int(np.sum(pi>1e-12)),L1_change=float(np.abs(pi-initial).sum()),differs_from_BBSE=bool(np.abs(pi-initial).sum()>1e-8))


def direct_kkt(C,q,pi):
    gradient=np.asarray(C).T@(np.asarray(C)@pi-q);active=pi>1e-12
    level=float(np.mean(gradient[active]));stationarity=float(np.max(np.abs(gradient[active]-level)))
    dual_violation=float(np.max(np.maximum(level-gradient[~active],0))) if np.any(~active) else 0.
    return max(stationarity,dual_violation,abs(float(pi.sum())-1),float(np.max(np.maximum(-pi,0))))


def face_reference(C,q):
    """Independent exhaustive equality-constrained solve on every nonempty face."""
    C=np.asarray(C,dtype=np.float64);q=np.asarray(q,dtype=np.float64);n=C.shape[1]
    H=C.T@C;f=C.T@q;best=None;best_objective=np.inf
    for mask in range(1,1<<n):
        indices=[i for i in range(n) if mask&(1<<i)];m=len(indices)
        K=np.zeros((m+1,m+1));K[:m,:m]=H[np.ix_(indices,indices)];K[:m,m]=1;K[m,:m]=1
        rhs=np.append(f[indices],1.);solution=np.linalg.solve(K,rhs)[:m]
        if np.any(solution<0):continue
        pi=np.zeros(n);pi[indices]=solution;objective=float(.5*np.sum((C@pi-q)**2))
        if objective<best_objective:best=pi;best_objective=objective
    assert best is not None
    return best,best_objective


class ActiveSetCLS:
    """T018R fixed face-removal/insertion algorithm; cache maps per channel."""
    def __init__(self, C):
        self.C = np.array(C, dtype=np.float64, copy=True)
        self.H = self.C.T @ self.C
        self.pinv = np.linalg.pinv(self.C)
        self.face_systems = {}

    def solve(self, q):
        q = np.asarray(q, dtype=np.float64)
        initial = project_simplex(self.pinv @ q)
        f = self.C.T @ q
        active = tuple(np.flatnonzero(initial > 1e-12).tolist())
        seen = set(); trace = []; updates = 0; clipped = 0
        while True:
            if active in seen:
                raise RuntimeError(f'working-set cycle: {trace}, repeated={active}')
            seen.add(active)
            m = len(active); idx = list(active)
            if active not in self.face_systems:
                K = np.zeros((m+1, m+1))
                K[:m, :m] = self.H[np.ix_(idx, idx)]
                K[:m, m] = 1.; K[m, :m] = 1.
                self.face_systems[active] = K
            x = np.linalg.solve(self.face_systems[active], np.append(f[idx], 1.))[:m]
            if not np.all(np.isfinite(x)):
                raise RuntimeError(f'nonfinite face solve: {active}')
            entry = dict(active=list(active), minimum=float(x.min()))
            if x.min() < -1e-12:
                remove = idx[int(np.argmin(x))]
                entry.update(action='remove', coordinate=remove)
                active = tuple(j for j in active if j != remove)
            else:
                pi = np.zeros(len(f)); pi[idx] = x
                negative = pi < 0
                if np.any(negative):
                    clipped += int(negative.sum()); pi[negative] = 0.; pi /= pi.sum()
                g = self.H @ pi - f; positive = pi > 1e-12
                level = float(np.mean(g[positive]))
                inactive = np.flatnonzero(~positive)
                violation = level-g[inactive]
                if len(inactive) and violation.max() > 1e-10:
                    add = int(inactive[int(np.argmax(violation))])
                    entry.update(action='add', coordinate=add)
                    active = tuple(sorted(set(active) | {add}))
                else:
                    kkt = direct_kkt(self.C, q, pi)
                    if kkt > 1e-10 or abs(pi.sum()-1.) > 1e-12 or pi.min() < -1e-12:
                        raise RuntimeError(f'acceptance failure: KKT={kkt}, active={active}')
                    entry['action'] = 'accept'; trace.append(entry)
                    obj = float(.5*np.sum((self.C@pi-q)**2))
                    initial_obj = float(.5*np.sum((self.C@initial-q)**2))
                    return pi, dict(iterations=updates, updates=updates, trace=trace,
                        converged=True, cap_hit=False, direct_KKT=kkt, sum_error=float(abs(pi.sum()-1.)),
                        objective=obj, initial_objective=initial_obj,
                        active_classes=int(positive.sum()), clipped_coordinates=clipped,
                        L1_change=float(np.abs(pi-initial).sum()),
                        differs_from_BBSE=bool(np.abs(pi-initial).sum()>1e-8))
            trace.append(entry); updates += 1
            if updates > 100:
                raise RuntimeError(f'working-set update cap: {trace}')
