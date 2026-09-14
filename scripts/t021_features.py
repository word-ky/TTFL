"""Experiment-local canonical frozen feature extraction; no architecture edit."""
import torch


def extract(model,x,state):
    for i,block in enumerate((model.conv1,model.conv2)):
        x=block(x)
        if state is not None:
            gamma,beta=state[2*i:2*i+2]
            x=x*(1+gamma[None,:,None,None])+beta[None,:,None,None]
    h=model.fc1(torch.flatten(x,1))
    return h,model.fc(h)
