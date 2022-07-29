import torch
import requests
import numpy as np
from turtle import home

### Functions ###
def download(url: str, home_dir: str, fn = None) -> str:
    """
    Downloads the content of a url to the specified home_dir

    Inputs
        :url: <str> to the location contianing the content
        :home_dir: <str> the home directory containing subdirectories to write to
        :fn: the name to give the file when saved
    
    Outputs
        :returns: the path to the saved file containing the content
    """
    if fn is None:
        fn = url.split('/')[-1]

    r = requests.get(url)
    if r.status_code == 200:
        open(f"{home_dir}/outputs/{fn}", 'wb').write(r.content)
        print("{} downloaded: {:.2f} KB".format(fn, len(r.content) / 1024.0))
        return f"{home_dir}/outputs/{fn}"
    else:
        raise ValueError(f"url not found: {url}")
    
def zero_upsampling(img: torch.Tensor, factors: int) -> torch.Tensor:
    """
    Upsamples an image by assigning each input pixel to its corresponding pixel at high resolution 
    and leaving all the missing pixels around it as zeros. The location of each input pixel falls equally 
    in between factor pixels in the high resolution, where factor is on of the upsampling factors

    For a 3D image and factors = (k1, k2)
        original index (n, i, j, ...) -> new index (n, i', j', ...) where (n, i', j', ...) = (n, i*k1, j*k2, ...)
    For a 4D image and factors = (k1, k2)
        original index (n, m, i, j, ...) -> new index (n, m, i', j', ...) where (n, m, i', j', ...) = (n, m, i*k1, j*k2, ...)

    Currently supports only 3D and 4D images

    Example
        upsampling by a factors = (2, 2) and 3D image
                         [[[1, 0, 2, 0, 3, 0],        
        [[[1, 2, 3]],      [0, 0, 0, 0, 0, 0]],
         [[4, 5, 6]],  -> [[4, 0, 5, 0, 6, 0],    
         [[7, 8, 9]]]      [0, 0, 0, 0, 0, 0]],    
                          [[7, 0, 8, 0, 9, 0],
                           [0, 0, 0, 0, 0, 0]]]

    Inputs
        :img: <Tensor> representing the image of size C x H x W or D x C x H x W
        :factors: <tuple<int>> of length 2 containing the factors, (n, m), of how much to upscale each dimension of the image by 

    Outputs
        :returns: <Tensor> of the upsampled image with the same dimensions as the input img, i.e C x H x W or D x C x H x W
    """
    x = len(factors)
    img_size = list(img.shape)
    dim = len(img_size)
    upscaled_img_size = img_size[0: dim - x] + [img_size[dim - x + i] * factors[i] for i in range(x)]
    upscaled_img = np.zeros(upscaled_img_size)
    print(f"upscaling img from {img_size} -> {upscaled_img_size}")

    n, m = factors
    if dim == 3: upscaled_img[:, ::n, ::m] = img
    if dim == 4: upscaled_img[:, :, ::n, ::m] = img


    return torch.Tensor(upscaled_img)

def backward_wrapping(img: torch.Tensor) -> torch.Tensor:
    """
    Preforming backWrapping on an image using the x_motion 
    """
    raise NotImplementedError

### Functions ###
    