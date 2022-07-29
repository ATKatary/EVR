import re
import math
import json
import torch
import numpy as np
from regex import F
import torch.nn as nn
from torch.nn import init

### Global Constants ###
_padding_size = lambda kernel_size: (kernel_size - 1) // 2
_upsampled_flow = lambda: nn.ConvTranspose2d(2, 2, 4, 2, 1)
_deconv = lambda in_c, out_c: _conv2d(in_c, out_c, k=4, s=2, batch_norm=False, deconv=True)
_inter_conv = lambda in_c, out_c: _conv2d(in_c, out_c, activation=False)
_predict_flow = lambda in_c: _conv2d(in_c, 2, batch_norm=False, activation=False, deconv=False)

'Parameter count = 45,371,666'

### Classes ###
class FlowNet2SD(nn.Module):
    """
    AF(args, div_flow) = a FlowNetSD model with architecture specified in the flownet2sd.json file

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self, args, div_flow = 20):
        super(FlowNet2SD,self).__init__()
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

        with open("implementation/optical_flow/flownet2sd.json", 'rb') as layers:
            print("layers", layers)
            layers = json.loads(layers.read())
            self.conv = [_conv2d(in_c, out_c, s=s) for in_c, out_c, s in layers['conv']]
            self.deconv = [_deconv(in_c, out_c) for in_c, out_c in layers['deconv']]
            self.inter_conv = [_inter_conv(in_c, out_c) for in_c, out_c in layers['inter_conv']]
            self.predict_flow = [_predict_flow(in_c) for in_c in layers['predict_flow']]
            self.upsampled_flow = [_upsampled_flow() for _ in range(4)]
         
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

        self.upsample = [nn.Upsample(scale_factor=4, mode='bilinear')]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward porpagation of input through the net """
        print("Analyzing ...")
        rgb_mean = input.contiguous().view(input.size()[:2] + (-1,)).mean(dim=-1).view(input.size()[:2] + (1, 1, 1,))
        input = (input - rgb_mean) / self.rgb_max
        input = torch.cat((input[:,:,0,:,:], input[:,:,1,:,:]), 1)
        print(f"Preforming optical flow on input of size {input.shape} ...")

        n_conv = len(self.conv)
        n_predict_flow = len(self.predict_flow)
        out_convs = []
        out_conv = input
        for i in range(n_conv):
            out_conv = self.conv[i](out_conv)
            if i % 2 == 0: out_convs.append(out_conv)

        out_flows = []
        out_flow = self.predict_flow[0](out_convs[n_conv // 2])
        for i in range(n_predict_flow - 1):
            out_deconv = self.deconv[i](out_conv)
            out_conv = out_convs[n_conv // 2 - (i + 1)]
            out_upsampled_flow = self.upsampled_flow[i](out_flow)

            print(f"Concatinating: {out_conv.shape} {out_deconv.shape} {out_upsampled_flow.shape}")

            out_conv = torch.cat((out_conv, out_deconv, out_upsampled_flow), 1)
            out_interconv = self.inter_conv[i](out_conv)
            out_flow = self.predict_flow[i + 1](out_interconv)

            out_flows.append(out_flow)

        if self.training: return out_flows[::-1]
        else: return self.upsample[0](out_flows[-1] * self.div_flow)
    
    def initalize(self):
        """ Initialized the network using the pretrained FlowNet2SD model from https://github.com/NVIDIA/flownet2-pytorch/tree/master """
        parameter_dict = torch.load('implementation/optical_flow/pretrained_models/FlowNet2-SD.pth.tar')['state_dict']
        print("Initializing ...")

        for layer_str, parameters in parameter_dict.items():
            if not isinstance(parameters, nn.Parameter): parameters = nn.Parameter(parameters)

            layer_info = layer_str.split(".")
            layer_name = layer_info[0]
            parameter_type = layer_info[-1]

            if "upsampled_flow" in layer_name:
                layer = self.upsampled_flow
                layer_number = 5 - int(layer_name.split("_")[-1])
            elif "predict_flow" in layer_name:
                layer = self.predict_flow
                layer_number = 6 - int(layer_name.split("_")[-1][-1])
            elif "inter_conv" in layer_name:
                layer = self.inter_conv
                layer_number = 5 - int(layer_name.split("_")[-1][-1])
            elif "deconv" in layer_name:
                layer = self.deconv
                layer_number = 5 - int(layer_name[-1])
            elif "conv" in layer_name:
                layer = self.conv
                layer_number = 2 * int(layer_name[4]) - 1
                if len(layer_name) > 5: layer_number += 1
            elif "upsample" in layer_name:
                layer = self.upsample
                layer_number = 0
            else: continue

            try:
                if parameter_type == "weight": layer[layer_number].weight = parameters
                elif parameter_type == "bias": layer[layer_number].bias = parameters
            except Exception as e:
                print(f"Failed to initalizing layer {layer_str}'s {parameter_type}\n{parameters}")
                raise e

### Helper Functions ###
def _conv2d(in_c, out_c, k = 3, s = 1, bias = True, batch_norm = True, activation = True, deconv = False):
    """
    """
    batch_norm_layer = nn.BatchNorm2d(out_c)
    activation_layer = nn.LeakyReLU(0.1, inplace=True)
    conv_layer = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=_padding_size(k), bias=bias)
    deconv_layer = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=_padding_size(k), bias=bias)

    if batch_norm and activation: return nn.Sequential(conv_layer, batch_norm_layer, activation_layer)
    if batch_norm: return nn.Sequential(conv_layer, batch_norm_layer)
    if deconv: return nn.Sequential(deconv_layer, activation_layer)
    if activation: return nn.Sequential(conv_layer, activation_layer)
    return conv_layer