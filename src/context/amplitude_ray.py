"""T012 fixed scaling and the two predeclared true-direction ray selectors."""
ALPHAS=(0.,.25,.5,.75,1.,1.25,1.5)

def scale_state(state,alpha):
    return [v*alpha for v in state]

def choose_ray(scores):
    j=min(ALPHAS,key=lambda a:(scores[a]['J'],a))
    zero=scores[0.]
    allowed=[a for a in ALPHAS if a>0 and scores[a]['consistency_js']<zero['consistency_js'] and scores[a]['flip_top1_agreement']>=zero['flip_top1_agreement']]
    pc=min(allowed,key=lambda a:(scores[a]['consistency_js'],a)) if allowed else 0.
    return dict(J_ray=j,PC_ray_safe=pc)
