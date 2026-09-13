from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/PFLlib/system'))
from flcore.trainmodel.models import FedAvgCNN


class ContextFedAvgCNN(FedAvgCNN):
    channels = (32,64)

    def forward(self,x,state=None):
        for i,block in enumerate((self.conv1,self.conv2)):
            x = block(x)
            if state is not None:
                gamma,beta = state[2*i:2*i+2]
                x = x*(1+gamma[None,:,None,None])+beta[None,:,None,None]
        return self.fc(self.fc1(torch.flatten(x,1)))
