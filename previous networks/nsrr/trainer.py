import sys
import os
import math

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision



import time

from factories import DataFactory, LossFactory
from bfmodels import NSRR

# allow maximum memory per GPU
torch.cuda.set_per_process_memory_fraction(1.0, 1)
torch.cuda.set_per_process_memory_fraction(1.0, 0)

import sys
import os
import math

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import time

from utils import *

# allow maximum memory per GPU
torch.cuda.set_per_process_memory_fraction(1.0, 1)
torch.cuda.set_per_process_memory_fraction(1.0, 0)

torch.cuda.set_device(1)

print('Using CUDA', torch.version.cuda)
print('dev0 -', torch.cuda.get_device_name(0))
print('dev1 -', torch.cuda.get_device_name(1))

## Hyperparameters
batch_size = 4
workspace = '.'
epochs = 100

train = True

# Video Loading Parameters

# command to convert obs recording to hevc for dali: 
# ffmpeg -i bf4_rawdata.avi -c:v libx265 -x265-params lossless=1 -vtag hvc1 -c:a copy -an output.mp4
sequence_length = 5                       # Number of frames per data sequence
initial_prefetch_size = 2                 # look at DALI docs. Probably number of samples to prefetch
original_video_size = (1920//3, 1080//3)        # The original size of the recording
hires_sf = 1                              # 1/Scalefactor applied to original resolution image. hires_img = (1920x1080) / hires_sf
lores_sf = 2                              # 1/Scalefactor applied to original resolution image. lores_img = (1920x1080) / lores_sf
frame_stride = 2                          # Space btwn sampling frames. stride=1 --> 60fps, stride=2 --> 30fps, stride=3 --> 20fps ...
lr = 0.0001

print("Reference Video Size:", np.array(original_video_size)/hires_sf)
print("Low Resolution Video Size:", np.array(original_video_size)/lores_sf)
print("Upscaling by", lores_sf/hires_sf)

dataf = DataFactory(
            flow_model_type='FlowNet2', 
            depth_model_type='DPT_Large', 
            batch_size=batch_size, 
            frame_stride=frame_stride, 
            lores_sf=lores_sf, 
            hires_sf=hires_sf, 
            seq_len=sequence_length
        )

traindev = 0
print('Buildling model')
nsrr = NSRR(dev=0, debug=False)
print('Moving model to GPU', traindev)
nsrr.cuda(traindev)


def run_test_np(model, frameset):
    ref = torch.tensor(np.array(frameset)/255.0).transpose(-1, 1).transpose(-1, 2).unsqueeze(0)
    ref, rgbd, flow = dataf.prep_frameset(ref.float().cuda(1))
    with torch.no_grad():
        rgbd = rgbd.cuda(0) * 255
        x = model(rgbd, flow.cuda(0))

    x = torch.nn.functional.interpolate(x, size=ref.shape[-2:], mode='nearest')
    loss = LossFactory(dev=0).total_loss(x, ref.cuda(0))

    x, ref = x.cpu().detach().numpy(), ref.cpu().detach().numpy()
    x, ref = x.squeeze(0), ref.squeeze(0)
    x = np.transpose(x, (1, 2, 0))
    
    print('max', np.max(x), 'min', np.min(x), 'mean', np.mean(x))
    x = np.clip(x, 0, 255) / 255
    #x = (x - np.min(x)) / np.max(x)
        
    ref = np.transpose(ref, (1, 2, 0))
    y = rgbd.cpu().detach()[0, 4, 0:3, ...]
    y = np.transpose(y.numpy(), (1, 2, 0))
    return x, ref, y, loss

if train:

    def validate():
        cap = cv2.VideoCapture('./videos/bf4_lossless_640x360.mp4')
        succ, img = cap.read()
        cap.set(1, 400)
        frames = []
        i = 0
        while succ and i < 140:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            frames.append(cv2.resize(img, original_video_size))
            succ, img = cap.read()
            i += 1

        print('validating...')
        x, ref, y, loss = run_test_np(nsrr, frames[0:0+5*frame_stride:frame_stride])
        print('validation total loss:', loss)
        print('writing results to ./results folder...')
        plt.imsave('./results/val_ref.png', ref)
        plt.imsave('./results/val_restored.png', x)
        plt.imsave('./results/val_bilin.png', cv2.resize(y/255.0, (y.shape[1]*2, y.shape[0]*2)))

    print('getting data iteroator...')
    dataf_iter = dataf.get_dali_iterator()

    print('Building Loss')
    criterion = LossFactory()    
    optimizer = torch.optim.Adam(nsrr.parameters(), lr=lr)

    train_start_t = time.time()
    print('started training at', train_start_t)
    print('epoch_size=', dataf.get_epoch_size())
    hires = None
    rgbd = None
    flows = None
    last_t = time.time()
    avg_losses = []
    for e in range(epochs):
        iters = 0
        running_loss = 0.0
        epoch_start = time.time()
        for i, data in enumerate(dataf_iter):
            start_frame, _hires, _rgbd, _flows = dataf.pre_compute_data_from_iter(data)
            
            hires  = _hires.to(device=traindev)
            rgbd   = _rgbd.to(device=traindev)
            flows  = _flows.to(device=traindev)

            if rgbd.shape[1] != 5:
                print('WARNING: less than 5 frames per batch received. skipping...')
                continue
            
            # at this point, we have a batch of: 
            # --------------
            # hires  = reference high resolution goal
            # rgbd   = low resolution RGB-Depth tensor for 5 frames
            # flows  = 4 frames of optica flow calculated as delta between the 5 frames
            # start_frame = frame when this sequence begins in the original data
            
            optimizer.zero_grad()
            rgbd *= 255
            outputs = nsrr(rgbd, flows)
            
            #print('reshaping outputs from,', outputs.shape, 'to', hires.shape)
            outputs = torch.nn.functional.interpolate(outputs, size=hires.shape[-2:], mode='nearest')
            loss = criterion.total_loss(outputs, hires*255)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            print("epoch: {}, loss={}, iter {}" .format(e, round(loss.item(), 6), i), end='\r')
            iters += 1

        torch.save(nsrr.state_dict(), './results/weights/weights_{}.torch'.format(e))
        avg_loss = running_loss / iters
        avg_losses.append(avg_loss)
        print('[epoch done] delta time:', time.time() - epoch_start, 's, avg_loss', avg_loss)
        with open('./results/losses.txt', 'w') as f: 
            f.write(str(avg_losses))

        validate()
        
        dataf_iter.reset()
    
    torch.save(nsrr.state_dict(), 'trained_weights.torch')
    print('losses:')
    print(avg_losses)
else:
    nsrr.load_state_dict(torch.load('trained_weights.torch'))
    nsrr.eval()
    # started at 2:12
    cap = cv2.VideoCapture('./videos/bf4_lossless_640x360.mp4')
    succ, img = cap.read()
    frames = []
    i = 0
    while succ and i < 140:
        frames.append(cv2.resize(img, original_video_size))
        succ, img = cap.read()
        i += 1

    print('running inference...')
    x, ref, y, loss = run_test_np(nsrr, frames[60:70:2])
    print('total loss:', loss)
    print('writing results to ./results folder...')
    cv2.imwrite('./results/ref.png', ref*255)
    cv2.imwrite('./results/restored.png', x*255)
    cv2.imwrite('./results/bilin.png', cv2.resize(y*255, (y.shape[1]*2, y.shape[0]*2)))
