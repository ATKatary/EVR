import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .srcnn import SRCNN
import torch.optim as optim
from .loss_functions import psnr
from .data_utils import SRCNNDataset
from torchvision.utils import save_image

### Functions ###
def train_and_validate(device, val_inputs, val_labels, train_inputs, train_labels, batch_size, epochs, lr, home_dir):
    """
    Tests out the network against an inout image

    Inputs
        :device: the computation device CPU or GPU
        :val_inputs: <SRCNNDataset> the inputs for validation
        :val_labels: <SRCNNDataset> the labels for validation
        :train_inputs: <SRCNNDataset> the inputs for training
        :train_labels: <SRCNNDataset> the labels for training
        :batch_size: <int> size of batches to load data in
        :epochs: <int> number of times to train the network
        :lr: <float> rate at which we update the network weights
        :home_dir: <str> the home directory containing subdirectories to read from and write to
    
    Outputs
        :returns: the traind SRCNN model
    """
    model = torch.nn.DataParallel(SRCNN(k=1)).to(device)

    val_data = SRCNNDataset(val_inputs, val_labels)
    train_data = SRCNNDataset(train_inputs, train_labels)

    val_loader = val_data.load(batch_size)
    train_loader = train_data.load(batch_size)

    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr = _train(model, train_loader, len(train_data), device, lr)
        val_epoch_loss, val_epoch_psnr = _validate(model, val_loader, epoch, len(val_data), device, home_dir)

        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes\nSaving model ...")

    model_name = input("Model name:")
    torch.save(model.state_dict(), f"{home_dir}/pretrained/{model_name}.pth")
    return model

### Helper Functions ###
def _train(model, dataloader, n, device, lr, optimizer = None, criterion = nn.MSELoss()):
    """
    Trains the SRCNN

    Inputs
        :model: <SRCNN> to train 
        :dataloader: <DataLoader> loading the training data 
        :n: <int> length of the training data
        :lr: <float> learning rate
        :optimizer: the optimization function for backward propogation, by defualt it is Adam
        :criterion: the loss function, by default MSE
    
    Outputs
        :returns: the final loss and psnr loss of the model
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr = lr)

    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    batch_size = dataloader.batch_size

    for _, data in tqdm(enumerate(dataloader), total = int(n / batch_size)):
        image_data = data[0].to(device)
        label = data[1].to(device)
        optimizer.zero_grad()

        outputs = model(image_data)
        loss = criterion(outputs, label)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_psnr += psnr(label, outputs)

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(n / batch_size)
    return final_loss, final_psnr

def _validate(model, dataloader, epoch, n, device, home_dir, criterion = nn.MSELoss()):
    """
    Tests out the network against an inout image

    Inputs
        :model: <SRCNN> to train
        :dataloader: <DataLoader> loading the training data  
        :epoch: <int> epoch this network is currently being trained for
        :n: <int> length of the training data
        :device: the computation device CPU or GPU
        :home_dir: <str> the home directory containing subdirectories to read from and write to
        :criterion: the loss function, by default MSE
    
    Outputs
        :returns: a tuple (loss, psnr) of the final loss and final psnr
    """
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    batch_size = dataloader.batch_size

    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total = int(n / batch_size)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            
            outputs = model(image_data)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            running_psnr += psnr(label, outputs)

        outputs = outputs.cpu()
        save_image(outputs, f"{home_dir}/outputs/training/val_sr{epoch}.png")

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(n / batch_size)
    return final_loss, final_psnr
