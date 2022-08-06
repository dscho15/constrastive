import torch
import torch.nn as nn
import torchvision.models as models

from collections import OrderedDict

class Encoder(nn.Module):
    
    def __init__(self, verbose=False) -> None:
        super().__init__()
        
        self.resnet_18_pretrained = models.resnet18(pretrained=True)
        self.resnet_18_pretrained.fc = nn.Identity()
        self.resnet_18_pretrained.train()
        
        if verbose:
            print("Encoder module")
            print(self.resnet_18_pretrained.modules)
    
    def forward(self, x):
        return self.resnet_18_pretrained(x)
    
class Projector(nn.Module):
    
    def __init__(self, f1=512, f2=256, f3=256, verbose=False):
        super().__init__()
        
        self.net = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(f1, f2)),
            ('LeakyRELU', nn.LeakyReLU()),
            ('batcnorm', nn.BatchNorm1d(f2)),
            ('lin2', nn.Linear(f2, f3))
        ]))
        
        if verbose:
            print("Projector module")
            print(self.net.modules)
    
    def forward(self, x):
        return self.net(x)
    

if __name__ == "__main__":
    
    enc = Encoder(verbose=True)
    proj = Projector(verbose=True)
    
    x = torch.rand(100, 3, 92, 92)
    y = enc(x)
    z = proj(y)
    
    pass