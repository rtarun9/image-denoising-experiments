import torch
from x_denoiser import XModel 
from dataloader import get_train_and_validation_dataloader
import numpy as np
from torchinfo import summary
import os
from piq import ssim, SSIMLoss, psnr

   
def ssim_psnr_loss(output, target, alpha=0.5, beta=0.5):
    output = torch.squeeze(output)
    output = torch.unsqueeze(output, 0)
    output = torch.unsqueeze(output, 0)

    target = torch.squeeze(target)
    target = torch.unsqueeze(target, 0)
    target = torch.unsqueeze(target, 0)

    output = torch.clamp(output, 0, 1)
    target = torch.clamp(target, 0, 1)

    ssim_value = (1 - ssim(output, target)) * 5.0

    psnr_value = psnr(output, target) * 2.0

    loss = alpha * ssim_value + beta * psnr_value
    return loss 

def train_model(model, epochs):
 
    number_training_images = 20 
    number_validation_images = 1
    
    training_dataset_path = "../../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data"
    if not os.path.exists(training_dataset_path):
        training_dataset_path = "../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data"

    training_dataloader, validation_dataloader = get_train_and_validation_dataloader(training_dataset_path, shuffle=True)
   
    loss_fn = ssim_psnr_loss 
   
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=0.001)
   
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
        
             
train_model(XModel(num_channels=64).to('cuda'), epochs=2)
