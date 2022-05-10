import sys
import os

import torch
import torch.nn as nn

import nvidia
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

import pytorch_msssim 
import torchvision

sys.path.append(os.path.abspath('./flownet2'))
sys.path.append(os.path.abspath('./flownet2/utils'))
from models import FlowNet2, FlowNet2SD
import flow_utils
import numpy as np

class DataFactory():
    def __init__(self, 
            flow_model_type='FlowNet2-SD', depth_model_type='DPT_Hybrid', 
            lores_sf=2, hires_sf=1, frame_stride=1, batch_size=1, seq_len=5):

        self.lores_sf = lores_sf
        self.hires_sf = hires_sf
        self.batch_size = batch_size
        self.frame_stride = frame_stride
        self.seq_len = seq_len
        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        with torch.no_grad():
            self.datadev = 1
            self.traindev = 0

            self.pipe = None

            print('Using depth', depth_model_type)
            self.midas = torch.hub.load("intel-isl/MiDaS", depth_model_type)
            self.midas.to(device=self.datadev)

            class args: pass
            args.rgb_max = 1. 

            if flow_model_type == 'FlowNet2':
                print("Using FlowNet2...")
                args.fp16 = False
                self.flownet = FlowNet2(args)
                print('Loading pretrained weights...')
                self.flownet.load_state_dict(torch.load('./pretrained/FlowNet2_checkpoint.pth.tar')['state_dict'])
            elif flow_model_type == 'FlowNet2-SD':
                print("Using FlowNet2-SD...")
                args.fp16 = True
                self.flownet = FlowNet2SD(args)
                print('Loading pretrained weights...')
                self.flownet.load_state_dict(torch.load('./pretrained/FlowNet2-SD_checkpoint.pth.tar')['state_dict'])
            else:
                print("invalid flownet")
                assert "Invalid flownet"


            print('moving flownet to device') 
            self.flownet.cuda(1)#cuda(self.datadev)
            print('evaling flownet')
            self.flownet.eval()
            print('ready')
            
    def depth(self, input_batch, savemem=True): 
        """ Computes Depth on a batch of images using MiDaS """
        with torch.no_grad():
            if savemem:
                p1 = self.midas(input_batch[:input_batch.shape[0]//2,...])
                #torch.cuda.empty_cache()
                p2 = self.midas(input_batch[input_batch.shape[0]//2:,...])
                prediction = torch.cat([p1, p2], dim=0)
            else:
                prediction = self.midas(input_batch)
                
        return prediction / torch.max(prediction)
    
    def flow(self, x):
        """ Computes flow on a batch of images using FlowNet2 """
        with torch.no_grad():
            prediction = self.flownet(x)
        return prediction
    
    def get_dali_iterator(self):
        """ 
        Creates a DALI pipeline for loading video data and doing 
        basic pre-processing on it. Returns a PyTorch friendly iterator
        """
        @pipeline_def
        def video_pipe(seq_len, stride):
            out = fn.readers.video(
                device="gpu", 
                sequence_length=seq_len, 
                random_shuffle=True, 
                file_list="filelist.txt",
                file_list_frame_num=True,
                file_list_include_preceding_frame=False,
                enable_frame_num=True, 
                initial_fill=2,
                stride=stride,
                name="Reader")
            video = out[0]
            
            # Switch to (N, Frames, C, H, W)
            video = fn.transpose(video, perm=[0, 3, 1, 2]) 

            # Convert to 0-1 range
            video /= 255
            
            return out[2], video

        self.pipe = video_pipe(
            self.seq_len, self.frame_stride, 
            batch_size=self.batch_size, num_threads=1, device_id=self.datadev, seed=12345)
        print('Building datapipe...' )
        self.pipe.build()
        pii = PyTorchIterator(self.pipe, ['frame', 'video'], 
                              reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
        
        return pii

    def get_epoch_size(self):
        return self.pipe.epoch_size()

    @torch.no_grad()
    def pre_compute_data_from_iter(self, data):
        """ 
        Computes Optic Flow and Depth for a batch of video sequences returned 
        from the DALI pipeline and its iterator. 
        """
        
        '''
        lores = fn.resize(video, resize_x=original_video_size[0]//lores_sf)

        if hires_sf != 1: 
            hires = fn.resize(video, resize_x=original_video_size[0]//hires_sf)
        else:
            hires = video

        # pre-transform depth for MiDas network
        depth = (hires - 0.5) * 2
        depth = fn.resize(depth, resize_y=192, resize_x=384, 
                          interp_type=nvidia.dali.types.DALIInterpType.INTERP_CUBIC)

        square = fn.resize(hires, 
                           resize_y=square_shape[1]//lores_sf, 
                           resize_x=square_shape[0]//lores_sf, 
                           interp_type=nvidia.dali.types.DALIInterpType.INTERP_CUBIC)

        return out[2], lores, hires, depth, square
        '''
        video = data[0]['video']
        start_frame = data[0]['frame']
        return start_frame, *self.prep_frameset(video)
    
    def prep_frameset(self, video, inference=False):
        _batch_size = video.shape[0]
        with torch.no_grad():
            nearest_square_shape = (192, 320)
            square = nn.functional.interpolate(video.flatten(0, 1), size=nearest_square_shape)
            square = square.unflatten(0, (_batch_size, self.seq_len))
 
            #depth = nn.functional.interpolate(square.flatten(0, 1), size=(192, 384))
            #depth = nn.functional.interpolate(square.flatten(0, 1), size=(160, 320))
            depth = (square - 0.5) * 2
            
            # compute depth over batch * frame dimension: (N, frames, 1, H, W)
            depth = self.depth(depth.flatten(0, 1), savemem=not inference)
            depth = depth.unflatten(0, (_batch_size, self.seq_len)).unsqueeze(2)
            #depth = nn.functional.interpolate(depth, [1] + list(square.shape[3:]))

            # get RGBD for each frame
            rgbd = torch.cat((square, depth), axis=2)

            del square, depth

            #torch.cuda.empty_cache()
            #torch.cuda.synchronize()

            # pre-allocate the flow tensor (N, Frames-1, 2, H, W)
            flows = torch.empty(*([_batch_size, 4, 2] + list(rgbd.shape)[3:]), device=self.datadev)
            for i in range(_batch_size): 
                batch = rgbd[i, :, 0:3, ...]
                # pre-allocate the temporally continuous frames for this batch 
                inp = torch.empty(*([4, 3, 2] + list(batch.shape)[2:]), device=self.datadev)
                # concatenate adjacent frames for input into flownet 
                for p in range(4): 
                    pair = batch[p:p+2, ...]
                    inp[p, ...] = pair.transpose(1, 0)
                # execute flownet on the frame batch and add it to the main flow tensor
                flows[i, ...] = self.flow(inp)

            return video[:, 4, ...], rgbd, flows

class LossFactory():
    def __init__(self, dev=0):
        self.features = list(torchvision.models.vgg16(pretrained=True).features)[:23]
        self.features = torch.nn.ModuleList(self.features).cuda(dev).eval()
        self.ssim_module = pytorch_msssim.SSIM(data_range=255, size_average=True, channel=3)

    def perceptual_loss(self, out, ref):
        layers = {15}
        layers = {3, 8, 15, 22}
        
        # make sure we have gradient on output for gradient of loss calculation
        if not out.requires_grad:
            out = torch.autograd.Variable(out, requires_grad=True)
            
        fout = self.get_perceptual_features(out, layers_to_pick_from=layers)
        fref = self.get_perceptual_features(ref, layers_to_pick_from=layers)
        loss = 0
        for i in range(len(fout)):
            loss += fout[i].dist(fref[i], p=2) / torch.numel(fref[i])
        return loss
    
    def get_perceptual_features(self, x, layers_to_pick_from={3,8,15,22}):
        results = []
        for ii,model in enumerate(self.features):
            if ii > max(layers_to_pick_from): break
            x = model(x)
            if ii in layers_to_pick_from:
                results.append(x)
        return results
    
    def ssim_loss(self, out, ref):
        
        if len(out.shape) != len(ref.shape): 
            assert "Out and ref must have same dim count"
            
        if len(out.shape) == 3: 
            if type(out) == torch.Tensor:
                out = out.unsqueeze(0)
                ref = ref.unsqueeze(0)
            elif type(out) == np.ndarray:
                out = np.expand_dims(out, 0)
                ref = np.expand_dims(ref, 0)
                
        # make sure we have gradient on output for gradient of loss calculation
        if not out.requires_grad:
            out = torch.autograd.Variable(out, requires_grad=True)
            
        return self.ssim_module(out, ref)
     
    def total_loss(self, out, ref): 
        return 1 - self.ssim_loss(out, ref) + 0.1 * self.perceptual_loss(out, ref)
