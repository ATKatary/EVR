import math 
import numpy as np

def psnr(label, outputs, max_val = 1.0):
    """
    Computes Peak Signal to Noise Ratio 
    PSNR = 20 * log_10(max_val / sqrt(MSE))

    Definitions
        PSNR
            Peak Signal to Noise Ratio (the higher the better)
            https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
        
        MSE
            Mean Square Error

    Inputs
        :label: <torch.tensor> the ground truth image
        :outputs: <torch.tensor> the resulting super resoluted image
        :max_val: <float> the highest pixel value
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR