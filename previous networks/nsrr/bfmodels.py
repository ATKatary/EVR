import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from bfutils import viz_datachunk
import cv2
import flow_utils
import matplotlib.pyplot as plt

class UpsampleZero(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img, scale_factor):
        """
        IMPORTANT: we only support integer scaling factors for now!!
        """
        # input shape is: batch x channels x height x width
        # output shape is:
        device = img.device
        input_size = torch.tensor(img.size(), dtype=torch.int)
        input_image_size = input_size[2:] # input_image_size[0]-height, input_image_size[1]-width
        data_size = input_size[:2]
        
        # Get the last two dimensions -> height x width
        # compare to given scale factor
        b_ = np.asarray(scale_factor)
        b = torch.tensor(b_)
        # check that the dimensions of the tuples match.
        if len(input_image_size) != len(b):
            raise ValueError("scale_factor should match input size!")
        output_image_size = (input_image_size * b).type(torch.int) # element-wise product

        scale_factor = torch.tensor(np.asarray(scale_factor), dtype=torch.int)
        ##
        output_size = torch.cat((data_size, output_image_size))
        output = torch.zeros(tuple(output_size.tolist()))
        ##
        # todo: use output.view(...) instead.
        output[:, :, ::scale_factor[0], ::scale_factor[1]] = img
        return output.to(device=device)
    
class BackwardWarper(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_image, x_motion, debug_goal=None):
        """ Stolen from github nsrr-reimplementation """
        # this is diff for some reason
        x_motion = -x_motion

        index_batch, _, height, width = x_image.size()
        grid_x = torch.arange(width).view(1, -1).repeat(height, 1)
        grid_y = torch.arange(height).view(-1, 1).repeat(1, width)
        grid_x = grid_x.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
        grid_y = grid_y.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
        ##  
        grid = torch.cat((grid_x, grid_y), 1).float().to(device=x_motion.device)
        # grid is: [batch, channel (2), height, width]
        vgrid = grid + x_motion
        # Grid values must be normalised positions in [-1, 1]
        vgrid_x = vgrid[:, 0, :, :]
        vgrid_y = vgrid[:, 1, :, :]
        vgrid[:, 0, :, :] = (vgrid_x / width) * 2.0 - 1.0
        vgrid[:, 1, :, :] = (vgrid_y / height) * 2.0 - 1.0

        # swapping grid dimensions in order to match the input of grid_sample.
        # that is: [batch, output_height, output_width, grid_pos (2)]
        vgrid = vgrid.permute((0, 2, 3, 1)).to(device=x_image.device)
        output = torch.nn.functional.grid_sample(x_image, vgrid, mode='nearest', align_corners=False)
        return output

class FeatureExtractionNet(nn.Module):
    def __init__(self, kernel_size=3, padding='same'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        
    def forward(self, rgbd_tensor):
        # rgbd_tensor: (N, C, H, W) where C = R, G, B, D channels
        full_features = self.net(rgbd_tensor)
        channel_dim = len(rgbd_tensor.shape)-1-2
        return torch.concat((rgbd_tensor, full_features), dim=channel_dim)

class FeatureReweightingNet(nn.Module):
    '''Adopted from nsrr github'''
    def __init__(self, kernel_size=3, padding='same', scale=10):
        super().__init__()

        self.scale = scale
        self.net = nn.Sequential(
            # We think of the input as the concatanation of RGB-Ds of current frame, which has 4 channles
            # and full features of previous frames, each of which has 12 channels
            # so `in_channels=20`, which is 4+4*12 = 52
            # To save memory, we have to feed the upsampled RGB-D to compute a weight and calculate the weighted sum of the 12 channels
            nn.Conv2d(in_channels=4*5, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=kernel_size, padding=padding),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        
    def forward(self, current_rgbd_scaled, previous_features):
        stacked_reweighted_rgbds = torch.cat([
            current_rgbd_scaled.unsqueeze(1)[:, :, 0:4, ...], 
            previous_features[:, :, 0:4, ...] 
        ], dim=1)
        stacked_reweighted_rgbds = stacked_reweighted_rgbds.flatten(1, 2)
        
        w = self.net(stacked_reweighted_rgbds)
        w = (w-(-1))/2*self.scale # Scale
        
        weighted_previous_features = [w[:,i,:,:].unsqueeze(1)*previous_features[:, i, ...] for i in range(4)] # Reweighting
        return weighted_previous_features

class ReconstructionNet(nn.Module):
    '''Adopted from NSRR github repo'''
    def __init__(self, kernel_size=3, padding='same'):
        super().__init__()
        self.pooling = nn.MaxPool2d(2)

        # Split the network into 5 groups of 2 layers to apply concat operation at each stage
        # todo: the first layer of the model would take
        # the concatenated features of all previous frames,
        # so the input number of channels of the first 2D convolution
        # would be 12 * self.number_previous_frames
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(12*4+12, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.center = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),

        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(32+64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)


    def forward(self, current_features, reweighted_previous_features):
        # Features of the current frame and the reweighted features
        # of previous frames are concatenated
        
        reconstruction_inp = torch.cat([
            current_features.unsqueeze(1), 
            reweighted_previous_features
        ], dim=1)
        reconstruction_inp = reconstruction_inp.flatten(1, 2)

        channel_dim = 1
   
        out_encoder_1 = self.pooling(self.encoder_1(reconstruction_inp))
        
        out_encoder_2 = self.pooling(self.encoder_2(out_encoder_1))
        
        out_center = self.center(out_encoder_2)
        
        out_decoder_1 = self.decoder_1(torch.concat((out_center, out_encoder_2), dim=channel_dim))

        out_decoder_2 = self.decoder_2(torch.concat((out_encoder_1, out_decoder_1), dim=channel_dim))

        return out_decoder_2

class NSRR(nn.Module):
    def __init__(self, dev=0, debug=False):
        super().__init__()
        self.past_fextractor = FeatureExtractionNet()
        self.current_fextractor = FeatureExtractionNet()
        self.upsample_zero = UpsampleZero()
        self.upsample_factor = 2 
        self.backward_warp = BackwardWarper()
        self.reweighter = FeatureReweightingNet()
        self.reconstructer = ReconstructionNet()
        
        self.past_fextractor.cuda(dev)
        self.reweighter.cuda(dev)
        self.dev = dev
        self.debug = debug

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
    
    def forward(self, rgbds, flows): 
        _batch_size = rgbds.shape[0]
        zero_upsample = False
        upsample_mode = 'bilinear'

        if self.debug:
            print('visualizing input batch to ./debug/batch_d_input.png')
            for batch_i in range(rgbds.shape[0]):
                viz_datachunk(rgbds/255.0, flows, batch_idx=batch_i, \
                    imname=('./debug/a_batch_%d_input.png' % batch_i))
        
        current_rgbd = rgbds[:, -1, ...]
        current_features = self.current_fextractor(current_rgbd)

        if zero_upsample:
            current_features = self.upsample_zero(current_features,  [self.upsample_factor]*2)
        else:
            current_features = torch.nn.functional.interpolate(current_features,
                                                    scale_factor=self.upsample_factor, 
                                                    mode=upsample_mode)
        
        input_shape = torch.tensor(current_rgbd.shape[-2:])
        
        past_rgbds = rgbds[:, 0:-1, ...]
        past_features = torch.empty([_batch_size, 4, 12] + (input_shape*self.upsample_factor).tolist())
    
        for i in range(4): 
            past = past_rgbds[:, i, ...]
            past_feature = self.past_fextractor(past)
            if zero_upsample:
                past_feature = self.upsample_zero(past_feature, [self.upsample_factor]*2)
            else:
                past_feature = torch.nn.functional.interpolate(past_feature,
                                                        scale_factor=self.upsample_factor, 
                                                        mode=upsample_mode)
            past_features[:, i, ...] = past_feature
        
        # past_features[0] -> frame i-4
        # past_features[1] -> frame i-3
        # past_features[2] -> frame i-2
        # past_features[3] -> frame i-1
        # current_features -> frame i 

        if self.debug:
            inp = rgbds[:, 3, 0:3, ...]
            goal = rgbds[:, 4, 0:3, ...]
            res = self.backward_warp(inp, flows[:, 3, ...])
            res = res[0, ...].transpose(0, 1).transpose(1, 2)
            plt.imsave('./debug/x_warped1.png', res.cpu().detach().numpy()/255.0)
            x = inp[0, ...].transpose(0, 1).transpose(1, 2)
            plt.imsave('./debug/y_input_warp.png', x.cpu().detach().numpy()/255.0)
            x = goal[0, ...].transpose(0, 1).transpose(1, 2)
            plt.imsave('./debug/z_goal_warp.png', x.cpu().detach().numpy()/255.0)
            f = flow_utils.flow2img(flows[0, -1, ...].cpu().detach().numpy().transpose(1, 2, 0)) 
            plt.imsave('./debug/w_flow.png', f)
        
        flows = torch.nn.functional.interpolate(flows.flatten(0, 1), 
                                                scale_factor=self.upsample_factor, 
                                                mode='bilinear')
        flows = flows.unflatten(0, (_batch_size, 4))
        
        
        # Do Accumulative backward warping in batches
        # Frame i-1 gets warped into frame i using flow(i-1 --> i)
        # Frame i-2 gets warped into frame i using flow(i-2 --> i-1) and then flow(i-1 --> i)
        # Frame i-3 gets warped into frame i using flow(i-3 --> i-2), flow(i-2 --> i-1), flow(i-1 --> i)
        # Frame i-4 gets warped into frame i using flow(i-4 --> i-3), ..., flow(i-1 --> i)

        done_warped = [None] * 4
        warped = past_features
        for i in range(4):
            to_warp = warped[:, 0:4-i, ...].flatten(0, 1)
            warp_flows = flows[:, i:4, ...].flatten(0, 1)
            warped = self.backward_warp(to_warp, warp_flows)
            warped = warped.unflatten(0, (_batch_size, 4-i))
            done_warped[3-i] = warped[:, 3-i, ...]
 
        # done_warped[0] --> zero upscaled, acc. warped frame_minus_4
        # done_warped[1] --> zero upscaled, acc. warped frame_minus_3
        # done_warped[2] --> zero upscaled, acc. warped frame_minus_2
        # done_warped[3] --> zero upscaled, warped frame_minus_1
        
        # Create input to Feature Reweighting network
        done_warped = torch.stack(done_warped, dim=1).cuda(self.dev)
        if self.debug:
            print('visualizing warped batch to ./debug/batch_d_warped.png')
            for batch_i in range(rgbds.shape[0]):
                viz_datachunk(done_warped/255.0, flows, batch_idx=batch_i, \
                    imname=('./debug/b_batch_%d_warped.png' % batch_i))
        current_rgbd_scaled = current_features[:, 0:4, ...]
        
        reweighted = self.reweighter(current_rgbd_scaled, done_warped)
        reweighted = torch.stack(reweighted, dim=1).cuda(self.dev) 
        
        reconstructed = self.reconstructer(current_features, reweighted)
        reconstructed 
        
        return reconstructed 
        
        
