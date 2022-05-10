import torch
import torch.nn as nn
from .nsrr_utils import *
from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb

### Classes ###
class NSRR(nn.Module):
    """
    AF(net) = the composition of 3 neural networks: feature extraction, feature rewieghting, and reconstruction

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self) -> None:
        ### Representation ###
        super(NSRR, self).__init__()
        
        self.R = Reconstruction()
        self.FR = FeatureReweight()
        self.FE = FeatureExtraction()

    def forward(self, input: tuple, k: int) -> torch.Tensor:
        """ 
        Forward porpagation of input through the net 

        Inputs
            :input: (<Tensor>, tuple<Tensor>) of the frame to super resolved and the previous 4 frames, (current, (prev1, prev2, prev3, prev4))
                    each frame is of size D x C x H x W or C x H x W
            :k: <int> super resolution factor
        
        Outputs
            :returns: <Tensor> of the resolved image
        """
        current_frame, past_frames = input
        current_frame_output = zero_upsampling(self.FE(rgb_to_ycbcr(current_frame)))
        past_frame_outputs = torch.Tensor([zero_upsampling(current_frame)] + [backward_wrapping(zero_upsampling(self.FE(past_frame))) for past_frame in past_frames])
        all_frames_reweight = self.FR(torch.cat(past_frame_outputs, 1))
        output = self.R(torch.cat((current_frame_output, all_frames_reweight), 1))

        return ycbcr_to_rgb(output)

### Helper Classes ###
class FeatureExtraction(nn.Module):
    """
    AF(net) = a 3 layer CNN with ReLU activations on 4 channel, RGB + D, images

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self) -> None:
        ### Representation ###
        super(FeatureExtraction, self).__init__()
        padding = 1
        kernel_size = 3

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward porpagation of input through the net """
        return torch.cat((input, self.net(input)), 1)

class FeatureReweight(nn.Module):
    """
    AF(net) = a 3 layer CNN with ReLU, ReLU, Tanh + Scale activations on 4 channel, RGB + D, images

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self) -> None:
        ### Representation ###
        super(FeatureReweight, self).__init__()
        padding = 1
        kernel_size = 3

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=kernel_size, padding=padding),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward porpagation of input through the net """
        return self.net(input)

class Reconstruction(nn.Module):
    """
    AF(net) = a 10 layer CNN with (ReLU, ReLU + Pooling), (ReLU, ReLU + Pooling), 
                                  (ReLU, ReLU + Upsize), (Concat + ReLU, ReLU + Upsize),
                                  (Concat + ReLU, ReLU) activations on 4 channel, RGB + D, images

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self) -> None:
        ### Representation ###
        super(Reconstruction, self).__init__()
        padding = 1
        kernel_size = 3
        self.pooling = nn.MaxPool2d(2)

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(4 * 12 + 12, 64, kernel_size=kernel_size, padding=padding),
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
        self.center    = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward porpagation of input through the net """
        out_encoder_1 = self.pooling(self.encoder_1(input))
        out_encoder_2 = self.pooling(self.encoder_2(out_encoder_1))
        out_center = self.center(out_encoder_2)
        out_decoder_1 = self.decoder_1(torch.concat((out_center, out_encoder_2), 1))
        out_decoder_2 = self.decoder_2(torch.concat((out_encoder_1, out_decoder_1), 1))

        return out_decoder_2
