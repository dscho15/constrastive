from tqdm import tqdm
import torch.nn.functional as functional
import torch

def train(args, dataloader, models, loss_function, optimizer, scheduler):
    
    criterion = loss_function
    optimizer = optimizer
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    encoder = models["encoder"].to(device)
    projector = models["projector"].to(device)
    
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    
    loss_epoch = []
    
    for epoch in tqdm(range(epochs)):
        
        loss_item = 0
        
        for step, ((x_1, x_2), _) in enumerate(dataloader):
            
            optimizer.zero_grad()

            x_1 = x_1.to(device)
            x_2 = x_2.to(device)
            
            y_1 = encoder(x_1)
            y_2 = encoder(x_2)

            z_1 = projector(y_1)
            z_2 = projector(y_2)
            
            loss = criterion(z_1, z_2)
            loss.backward()
            
            optimizer.step()
            
            loss_item += loss.item()
        
            break
        
        scheduler.step()
        
        loss_epoch.append(loss_item)
    
    return encoder, loss_epoch

            
            
            
            