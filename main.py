import torch
import torch.optim as opt
import torchvision.datasets as datasets
import torch.utils.data as tdata

from net import Encoder as ConstrastEncoder
from net import Projector as ConstrastProjector
from loss import SimCLR_Loss
from augmentations import TransformsSimCLR

from train import train

import numpy as np
import os

if __name__ == "__main__":
    
    encoder = ConstrastEncoder()
    projector = ConstrastProjector()
    
    models = dict(encoder=encoder,
                  projector=projector)
    
    # docs for potential improvement - https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/9
    # params
    lr = 1e-3
    gamma = 0.9
    weight_decay = 1e-6
    img_size = (92, 92)
    
    #
    optimizer = opt.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    #
    train_args = dict(batch_size=128, epochs=20)
    
    #
    criterion = SimCLR_Loss(train_args["batch_size"], 0.1)
    
    #
    train_dataset = datasets.STL10('data', 
                                split="train", 
                                transform=TransformsSimCLR(img_size))

    #
    train_dataloader = tdata.DataLoader(train_dataset, 
                                        batch_size=train_args["batch_size"], 
                                        shuffle=True, 
                                        num_workers=os.cpu_count(),
                                        drop_last=True)
    
    #
    model, loss_epoch = train(train_args, train_dataloader, models, criterion, optimizer, scheduler)
    
    # 
    torch.save(model.state_dict(), "models/encoder")
    np.save("models/loss_epoch.npy", np.r_[loss_epoch])
    