import sys
import os
import math

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.append(os.path.abspath('flownet2'))
sys.path.append(os.path.abspath('flownet2/utils'))
from models import FlowNet2, FlowNet2SD
import flow_utils

def display_image_in_actual_size(im_path):

    dpi = mpl.rcParams['figure.dpi']
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()
    
    
def viz_datachunk(rgbd, flows, hires=None, batch_idx=0, roi=None, imname='test.png', imshow=False):
    """ 
    Draws the high resolution and low resolution samples side by side.
    As time passes, draws vertically downard.
    
    Args:
        hires, rgbd, flows: output from DataFactory
        batch_idx (int):   which batch to visualize
        roi (list-like):   region of interest. [x, y, height, width]
    """

    rgbd = rgbd.cpu().detach().numpy()
    flows = flows.cpu().detach().numpy()
    
    lo = rgbd[:, :, 0:3, ...]
    dpt = rgbd[:, :, 3, ...]

    if hires is not None:
        hires = hires.cpu().detach().numpy()
        hi = hires
    else:
        hi = lo
    
    vstack = []
    for i in range(lo.shape[1]):
        h = hi[batch_idx][i].transpose(1, 2, 0)
        l = lo[batch_idx][i].transpose(1, 2, 0)
        d = dpt[batch_idx][i]
        
        if i < 4:
            f = flows[batch_idx][i].transpose(1, 2, 0)
            f = flow_utils.flow2img(f) / 255
        else: 
            f = np.ones(h.shape)
        
        upscaled = cv2.resize(l, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_NEAREST)
        dscaled = cv2.resize(d, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_NEAREST)
        f = cv2.resize(f, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if roi is not None:
            x, y, height, width = roi
            h = h[y:y+height, x:x+width, :]
            upscaled = upscaled[y:y+height, x:x+width, :]
            dscaled = dscaled[y:y+height, x:x+width, :]
            f = f[y:y+height, x:x+width, :]
            
        dscaled = np.stack((dscaled, dscaled, dscaled), axis=-1)
        if hires is not None:
            img = np.hstack([h, upscaled, dscaled, f])
        else:
            img = np.hstack([upscaled, dscaled, f])
    
        vstack.append(img)

    vid = np.vstack(vstack)
    # save resolution accurate image
    plt.imsave(fname=imname, arr=vid,  format='png')

    if imshow:
        # display image in jupyter
        fig = plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.axis('off')
        plt.imshow(plt.imread('test.png'))
    
#viz_datachunk(hires, rgbd, flows, 0)


def _run_optic_flow(flownet, image_1, image_2, dev=0):
    images = np.array([image_1, image_2]).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda(dev)

    print('flonwet input:', im.shape)
    result = flownet(im).squeeze()

    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("test.flo", data)
    flow_utils.visulize_flow_file("test.flo", '.')
    
    del im, result
    
    return data, cv2.imread('test-vis.png')

def run_optic_flow(flownet, image_1, image_2, dev=0):
    x, y = _run_optic_flow(flownet, image_1, image_2, dev)
    torch.cuda.empty_cache()
    return x, y


def zero_upscale(im, sf=2):
    out = np.zeros((im.shape[0]*sf, im.shape[1]*sf, im.shape[2]), dtype=np.uint8)
    for r in range(im.shape[0]):
        for c in range(im.shape[1]):
            out[r*sf][c*sf] = im[r][c]
    return out

def lin_upscale(im, sf=2, interp=True):
    return cv2.resize(im, (im.shape[1]*sf, im.shape[0]*sf), interpolation=cv2.INTER_NEAREST if not interp else None)


def motion_compensate2(im1, im2, upscale_fn=lin_upscale, sf=4):
    data, out = run_optic_flow(im1, im2)
    flow = lin_upscale(data, sf)
    
    im1 = upscale_fn(im1, sf)
    im2 = upscale_fn(im2, sf)
    
    out = np.ones(im2.shape, dtype=np.uint8) * 255
    #out = im1.copy()
    for r in range(im2.shape[0]):
        for c in range(im2.shape[1]):
            one2two = flow[r][c]
            dest = (int(round(r+one2two[1]))), int(round(c+one2two[0]))
            destpix = 0
            if 0 <= dest[0] < im2.shape[0] and 0 <= dest[1] < im2.shape[1]:
                out[dest[0]][dest[1]] = im1[r][c]
            
    return out


def motion_compensate3(im1, im2, upscale_fn=lin_upscale, sf=4):
    data, out = run_optic_flow(im1, im2)
    hisf = 2
    flow = lin_upscale(data, sf*hisf)
    
    im1 = upscale_fn(im1, sf)
    im2 = upscale_fn(im2, sf*hisf)
    
    if True:
        # fully vectorized approach is slower as columwise 
        cols = np.array([[i for i in range(im2.shape[1])]]).T
        cols = np.expand_dims(np.tile(cols, im2.shape[0]).T, 2)
        rows = np.array([[i for i in range(im2.shape[0])]]).T
        rows = np.expand_dims(np.tile(rows, im2.shape[1]), 2)
        dirs = np.concatenate((cols, rows), axis=2)
        
        out = np.ones(im1.shape, dtype=np.float32)
        dests = ((dirs + flow + 0.5) / hisf - 0.5).astype(np.uint16)
        np.clip(dests[:, :, 1], 0, im1.shape[0]-1, out=dests[:, :, 1])
        np.clip(dests[:, :, 0], 0, im1.shape[1]-1, out=dests[:, :, 0])
        out[dests[:, :, 1], dests[:, :, 0], :] = lin_upscale(im1, 2, False)
    else:
        # vectorized approach. Compute columnwise motion compensatoin
        out = np.ones(im1.shape, dtype=np.float32)
        cols = np.array([i for i in range(im2.shape[1])])
        for r in range(im2.shape[0]):
            # dests[r, :, 0] --> cdest, dests[r, :, 1] --> rdest
            cdest = ( (np.array(cols+flow[r][:, 0])+0.5) /hisf-0.5 ).astype(np.uint16).T
            rdest = ( (np.array(r+flow[r][:, 1])+0.5) /hisf-0.5 ).astype(np.uint16).T
            rdest[rdest >= im1.shape[0]] = im1.shape[0] - 1
            cdest[cdest >= im1.shape[1]] = im1.shape[1] - 1
            out[rdest, cdest, :] = im1[r//hisf, cols//hisf]
        
    return out


def flownet_test():
    class args: pass
    args.rgb_max = 1. 
    print("Using FlowNet2...")
    args.fp16 = False
    flownet = FlowNet2(args)
    flownet.load_state_dict(torch.load('./pretrained/FlowNet2_checkpoint.pth.tar')['state_dict'])
    flownet.cuda(1)
    flownet.eval()

    _im1 = plt.imread('test/img0.ppm')
    _im2 = plt.imread('test/img1.ppm')

    def viz_diff(_im1, _im2):
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.imshow(np.hstack([_im1, _im2]))
        plt.show()
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.imshow(cv2.addWeighted(_im1,0.8,_im2,0.2,0))
        plt.show()
        data, out = run_optic_flow(flownet, _im1/255, _im2/255, dev=1)
        data, out = run_optic_flow(flownet, _im1/255, _im2/255, dev=1)
        data, out = run_optic_flow(flownet, _im1/255, _im2/255, dev=1)
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.imshow(out)
        plt.show()
        return data, out

    viz_diff(_im1, _im2)
    
if __name__=='__main__':
    #cap = cv2.VideoCapture('./videos/bf4_rawdata.avi')
    torch.cuda.set_device(0)
    cap = cv2.VideoCapture('./bf_short.avi')
    succ, img = cap.read()
    frames = []
    i = 0
    while succ and i < 140:
        frames.append(img)
        succ, img = cap.read()
        i += 1
        
        
    # choose codec according to format needed
    sf = 4
    #im1 = resize_to_nearest_pow2(frames[50], sf)
    #im2 = resize_to_nearest_pow2(frames[115], sf)
    #im12 = motion_compensate3(im1/255, im2/255, lin_upscale, sf)
    flownet_test()
    flownet_test()
    print('passed')
    
    pass

def resize_to_nearest_pow2(im, sf=1):
    height = int(2**round(math.log(im.shape[0], 2)))
    width = int(2**round(math.log(im.shape[1], 2)))
    return cv2.resize(im, (width//sf, height//sf))

def imshow(im):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(im)
    plt.show()
    
    
def imshow_overlay(im1, im2):
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(cv2.addWeighted(im1,0.5,im2,0.5,0))
    plt.savefig('out.png')
    plt.show()
    
    
def test_of():
    
    data, out = run_optic_flow(im1/255, im2/255)
    t1 = np.transpose(im1, [2, 0, 1])
    t2 = np.transpose(im2, [2, 0, 1])
    t1 = torch.from_numpy(t1/255)
    t2 = torch.from_numpy(t2/255)
    data = np.transpose(data, [2, 0, 1])
    flow = torch.from_numpy(data)

    use_cuda = True 

    # batch size of 4
    test_batch_size = 2
    flow = torch.stack((flow,)*test_batch_size)
    t1 = torch.stack((t2,)*test_batch_size)
    if use_cuda:
        flow = flow.cuda()
        t1 = t1.cuda()

    x = np.ones((3,3))
    print("Checkerboard pattern:")
    index_batch, _, height, width = t1.size()
    x = np.zeros((height,width),dtype=np.float64)
    x[1::1,::2] = 1
    x[::2,1::1] = 1
    x = cv2.resize(x, (width, height), interpolation=cv2.INTER_NEAREST)
    test_img = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).cuda()
    #t1 = test_img
    st = time.time()
    mine = torch_compensate(t1, flow, use_cuda=use_cuda).cpu()
    et = time.time()
    print('dt', et - st)
    warper = BackwardWarper()
    print(t1.shape)
    st = time.time()
    res = warper(t1.double(), flow.double()/255).cpu()
    et = time.time()
    print('dt', et - st)
    print(flow.shape)
    print(t1.shape)

    x = res[1, ...]
    x = x.transpose(0, 1).transpose(1, 2)
    print(x.mean())
    imshow(x)
    
    
def loss_study():
    loss_maker = LossMaker()    
    f1 = resize_to_nearest_pow2(frames[110], 8)
    f2 = resize_to_nearest_pow2(frames[0], 8)
    plt.imshow(f1)
    plt.show()
    plt.imshow(f2)
    plt.show()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    f1 = transform(f1).cuda(1)
    f2 = transform(f2).cuda(1)
    fout = loss_maker.perceptual_loss(f1, f2)
    ssim_loss = loss_maker.ssim_loss(f1, f2)
    total_loss = loss_maker.total_loss(f1, f2)
    print('perceptual_loss', fout)
    print('ssim_loss', ssim_loss)
    print('total_loss', total_loss)
    
    
def test_warping():
    frame_minus_1 = x[3].cpu()
    frame_minus_2 = x[2].cpu()
    frame_minus_3 = x[1].cpu()
    frame_minus_4 = x[0].cpu()

    frame_i = 3             # 0 --> frame_minus_4,  1 --> frame_minus_3,   2 --> frame_minus_2,   3 --> frame_minus_1

    ups = UpsampleZero()
    warp = BackwardWarper()
    fm1 = fext.cpu()(rgbd[:, frame_i, ...])
    warped = ups(fm1, [4, 4])

    for i in range(frame_i, 4):
        warped = warp(warped, torch.nn.functional.interpolate(flows[:, i, ...], scale_factor=4, mode='bilinear'))

    out = warped

    torch.max(abs(out - frame_minus_1))
    
    
def torch_compensate(img, flow, sf=4, use_cuda=False):
    """ 
    Warps and upscales a batch of images, given a batch of flows. 
    
    Args:
        img:  (N, C, H, W) image to be warped
        flow: (N, 2, H, W) flow to be used in warping
        sf:   scale-factor to upscale the warped result by
    """
    
    # Local parameter used to hyper-upscale flow for better precision
    hisf = 2
    
    flow = nn.functional.interpolate(flow, scale_factor=sf*hisf, mode='bilinear')
    img = nn.functional.interpolate(img, scale_factor=sf, mode='bilinear')
    img_2 = nn.functional.interpolate(img, scale_factor=2, mode='nearest')
    
    
    # create unwarped index matrix
    ncols = flow.shape[3]
    nrows = flow.shape[2]
    rows = np.array([[i for i in range(nrows)]]).T
    rows = np.expand_dims(np.tile(rows, ncols).T, 2)
    cols = np.array([[i for i in range(ncols)]]).T
    cols = np.expand_dims(np.tile(cols, nrows), 2)
    dirs = np.concatenate((cols, rows), axis=2)
    dirs = torch.tensor(dirs).unsqueeze(0).transpose(1, -1)
    
    # create output tensor or re-use img
    out = torch.ones(img.shape).double()
    # out = img
    
    if use_cuda:
        out = out.cuda()
        dirs = dirs.cuda()
        
    # compute warped indices for batch
    dests = ((dirs + flow + 0.5) / hisf - 0.5).long()
    torch.clip(dests[:, 1, :, :], 0, out.shape[2]-1, out=dests[:, 1, :, :])
    torch.clip(dests[:, 0, :, :], 0, out.shape[3]-1, out=dests[:, 0, :, :])
    
    # This could maybe be optimized
    for b in range(dests.shape[0]):
        out[b, :, dests[b, 1, :, :], dests[b, 0, :, :]] = img_2[b, ...]
    
    return out
