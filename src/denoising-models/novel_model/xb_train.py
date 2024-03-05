import torch
from xb_denoiser import XModel 
from dataloader import get_train_and_validation_dataloader
import numpy as np
from torchinfo import summary
import os
from piq import ssim, SSIMLoss, psnr

   
def train_model(model, epochs):
 
    number_training_images = 20 
    number_validation_images = 1
    
    training_dataset_path = "../../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data"
    if not os.path.exists(training_dataset_path):
        training_dataset_path = "../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data"

    training_dataloader, validation_dataloader = get_train_and_validation_dataloader(training_dataset_path, shuffle=True)
   
    loss_fn = torch.nn.MSELoss() 
   
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=0.0001)
   
    for epoch in range(epochs):
        model.train(True)
       
        running_loss = 0
       
        for i, data in enumerate(training_dataloader):
            noisy, clean = data

            noisy = noisy.to('cuda') 
            clean = clean.to('cuda')
            
            optimizer.zero_grad()
           
            output = model(noisy)
           
            loss = loss_fn(output, clean)
            loss.backward()
           
            optimizer.step()
           
            running_loss += loss.item()
      
            if i == number_training_images:
                break
             
        print('train image index : ', i)
            
        avg_loss = running_loss / len(training_dataloader)
       
        running_vloss = 0
       
        with torch.no_grad():
            for i, vdata in enumerate(validation_dataloader):
                vnoisy, vclean = vdata
           
                vnoisy = vnoisy.to('cuda')
                vclean = vclean.to('cuda')
                    
                voutput = model(vnoisy)
                vloss = loss_fn(voutput, vclean)
               
                running_vloss += vloss
            
                if i == number_validation_images:
                    break 

                print('validation image index : ', i) 
        
            avg_vloss = running_vloss / len(validation_dataloader)
        
            model_path = 'weights/model_{}.pth'.format(epoch)
            torch.save(model.state_dict(), model_path)
        
            print('training and validation loss : ', avg_loss, avg_vloss)
        
             
train_model(XModel(num_channels=1).to('cuda'), epochs=2)
