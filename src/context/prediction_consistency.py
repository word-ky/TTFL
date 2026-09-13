"""T011 fixed horizontal-flip prediction score and predeclared PC-safe policy."""
import torch

CONTEXTS=('clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur')

def js_divergence(p,q):
    m=(p+q)/2
    return ((torch.special.xlogy(p,p)-torch.special.xlogy(p,m)).sum(-1)+
            (torch.special.xlogy(q,q)-torch.special.xlogy(q,m)).sum(-1))/2

@torch.no_grad()
def score_state(model,pixels,state):
    logits=model(pixels,state);flipped_logits=model(torch.flip(pixels,dims=[-1]),state)
    p=logits.double().softmax(-1);q=flipped_logits.double().softmax(-1)
    top=p.topk(2,dim=-1).values
    metrics=dict(consistency_js=float(js_divergence(p,q).mean()),
        flip_top1_agreement=float((p.argmax(-1)==q.argmax(-1)).double().mean()),
        entropy_original=float(-torch.special.xlogy(p,p).sum(-1).mean()),
        entropy_flip=float(-torch.special.xlogy(q,q).sum(-1).mean()),
        prob_margin_original=float((top[:,0]-top[:,1]).mean()))
    return metrics,dict(logits=logits,flipped_logits=flipped_logits,probabilities=p,flipped_probabilities=q)

def select_pc_safe(scores):
    zero=scores['clean']
    allowed=[c for c in CONTEXTS[1:] if scores[c]['consistency_js']<zero['consistency_js'] and
             scores[c]['flip_top1_agreement']>=zero['flip_top1_agreement']]
    selected=min(allowed,key=lambda c:(scores[c]['consistency_js'],CONTEXTS.index(c))) if allowed else 'clean'
    s=scores[selected]
    return dict(selected=selected,zero_consistency=zero['consistency_js'],selected_consistency=s['consistency_js'],
        consistency_improvement=zero['consistency_js']-s['consistency_js'],selected_agreement=s['flip_top1_agreement'],
        zero_agreement=zero['flip_top1_agreement'])
