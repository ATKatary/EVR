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


import time

from factories import DataFactory, LossFactory
from bfmodels import NSRR

# allow maximum memory per GPU
torch.cuda.set_per_process_memory_fraction(1.0, 1)
torch.cuda.set_per_process_memory_fraction(1.0, 0)


torch.cuda.set_device(1)

dataf = DataFactory(
            flow_model_type='FlowNet2', 
            depth_model_type='DPT_Large', 
            batch_size=0, 
            frame_stride=0, 
            lores_sf=0, 
            hires_sf=0, 
            seq_len=5
        )


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
    
    x = np.clip(x, 0, 255) / 255
        
    ref = np.transpose(ref, (1, 2, 0))
    y = rgbd.cpu().detach()[0, 4, 0:3, ...]
    y = np.transpose(y.numpy(), (1, 2, 0))
    return x, ref, y, loss


cap = cv2.VideoCapture('./videos/bf4_lossless_640x360.mp4')

nsrr = NSRR(dev=0, debug=False).cuda(0)
nsrr.load_state_dict(torch.load('./results/run_5/weights/weights_99.torch'))
nsrr.eval()

succ, img = cap.read()
frames = []
i = 0
while succ and i < 2500:
    if i % 2 == 0: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        frames.append(img)
        succ, img = cap.read()

    if i % 100 == 0:
        print('read ', i, 'frames...', end='\r')
    i += 1
cap.release()

print('')

out_frames = []
start_t = time.time()
for i in range(5, len(frames)):
    seq = frames[i-5:i]
    x, ref, y, loss = run_test_np(nsrr, seq)
    out_frames.append(x)
    if i % 10 == 0:
        dt = time.time() - start_t 
        frames_per_second = len(out_frames) / dt
        frames_left = len(frames) - 5 - len(out_frames)
        seconds_left = frames_left / frames_per_second
        print('upscaling frame', i, ' done: ', round(i/len(frames)*100), '%, fps:', frames_per_second, ' time left:', round(seconds_left/60, 1), 'minutes', end='\r')

print('')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output2.mp4', fourcc, 30.0, (640,360))

for frame in out_frames:
    print(frame.shape)
    print('upscaling frame', i, ' done: ', i/len(frames)*100, '%', end='\r')
    out.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    del frame

out.release()
cv2.destroyAllWindows()
