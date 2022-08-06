import torch
import torch.optim as opt
import torchvision.datasets as datasets

from net import Encoder as ConstrastEncoder
from net import Projector as ConstrastProjector
from loss import SimCLR_Loss
from augmentations import TransformsSimCLR

from train import train

if __name__ == "__main__":
    
    encoder = ConstrastEncoder()
    projector = ConstrastProjector()
    
    # docs for potential improvement - https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/9
    # params
    lr = 1e-3
    gamma = 0.9
    weight_decay = 1e-6
    img_size = (92, 92)
    
    optimizer = opt.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    train_args = dict(encoder=encoder, 
                      projector=projector,
                      batch_size=128,
                      epochs=1
                      )

    criterion = SimCLR_Loss(train_args["batch_size"], 0.1)
    
    train_data = datasets.STL10('data', 
                                split="unlabeled", 
                                transform=TransformsSimCLR(img_size))
    
    pass