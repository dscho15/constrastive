import torch.nn.functional as functional

def train(epochs, dataloader, encoder, projector, augmentations, loss, **kwargs):
    
    for epoch in range(epochs):
        
        for batch, (x, y) in enumerate(dataloader):

            x_1 = augmentations(x)
            x_2 = augmentations(x)

            y_1 = encoder(x_1)
            y_2 = encoder(x_2)

            z_1 = projector(y_1)
            z_2 = projector(y_2)

            z_1 = functional.normalize(z_1)
            z_2 = functional.normalize(z_2)

            s = z_1 @ z_2.T
            
            