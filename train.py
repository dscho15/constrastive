import tqdm as tqdm
import torch.nn.functional as functional

def train(args, dataloader, models, loss_function, optimizer):
    
    criterion = loss_function
    optimizer = optimizer
    
    encoder = models["encoder"]
    projector = models["projector"]
    
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    
    loss_epoch = []
    
    for epoch in tqdm(range(epochs)):
        
        loss_item = 0
        
        for step, ((x_1, x_2), _) in enumerate(dataloader):
            
            optimizer.zero_grad()

            y_1 = encoder(x_1)
            y_2 = encoder(x_2)

            z_1 = projector(y_1)
            z_2 = projector(y_2)
            
            loss = criterion(z_1, z_2)
            loss.backward()
            
            optimizer.step()
            
            loss_item += loss.item()
        
        loss_epoch.append(loss_item)
    
    return encoder, loss_epoch

            
            
            
            