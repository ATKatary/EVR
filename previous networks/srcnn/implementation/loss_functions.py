import math 
import numpy as np
from skimage.metrics import structural_similarity

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
    
    Outputs
        :returns: psnr(label, outputs)
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
    
def ssim(label, outputs):
    """
    Calculates the Structural Similairty Index Measure 
    SSIM = L(x, y)^a C(x, y)^b S(x, y)^c

    Definitions
        SSIM
            Structural Similairty Index Measure (the higher the better)
            https://en.wikipedia.org/wiki/Structural_similarity
    
    Inputs
        :label: <torch.tensor> the ground truth image
        :outputs: <torch.tensor> the resulting super resoluted image
    
    Outputs
        :returns: ssim(label, outputs)
    """
    score, loss = structural_similarity(label, outputs, full=True, multichannel=True)
    return score