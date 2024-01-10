import torch
from hformer_model import HformerModel
from dataloader import get_train_and_validation_dataloader
import numpy as np
from torchinfo import summary

def train_model(model, epochs):
    print('model summary : ')
    summary(model, input_size=(64, 64, 64, 1))
    
    training_dataloader, validation_dataloader = get_train_and_validation_dataloader("../../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data", shuffle=True)
   
    loss_fn = torch.nn.MSELoss()
   
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=0.001)
   
    for epoch in range(epochs):
        model.train(True)
       
        running_loss = 0
       
        for i, data in enumerate(training_dataloader):
            noisy, clean = data
           
            optimizer.zero_grad()
           
            output = model(noisy)
           
            loss = loss_fn(output, clean)
            loss.backward()
           
            optimizer.step()
           
            running_loss += loss.item()
       
        avg_loss = running_loss / len(training_dataloader)
       
        running_vloss = 0
       
        with torch.no_grad():
            for i, vdata in enumerate(validation_dataloader):
                vnoisy, vclean = vdata
               
                voutput = model(vnoisy)
                vloss = loss_fn(voutput, vclean)
               
                running_vloss += vloss
        
        avg_vloss = running_vloss / len(validation_dataloader)
        
        model_path = 'model_{}'.format(epoch)
        torch.save(model.state_dict(), model_path)
        
        print('training and validation loss : ', avg_loss, avg_vloss)
        
             
train_model(HformerModel(num_channels=64, width=64, height=64), epochs=1)
