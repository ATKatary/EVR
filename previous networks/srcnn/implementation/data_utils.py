import os
import cv2
import h5py
import torch
import requests
import numpy as np
from torch.utils.data import DataLoader, Dataset

### Classes ###
class SRCNNDataset(Dataset):
    """
    AF(image_data, labels) = a datset and corresponding labels for supervised network training

    Representation Invariant:
        - true
    Representation Exposure:
        - safe
    """
    def __init__(self, image_data, labels):
        ### Representation ###
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        """ Override Object.__len__ """
        return (len(self.image_data))

    def __getitem__(self, index):
        """ Override Object.__getitem__ """
        image = self.image_data[index]
        label = self.labels[index]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )

    def load(self, batch_size): 
        """
        Load our dataset and return the loader
        """
        return DataLoader(self, batch_size=batch_size)

### Functions ###
def create_dataset(src_path, home_dir, stream = False, max_iters = None, k = 2):
    """
    Creates a dataset of images and labels from a source by downsampling the images in the source 

    Inputs
        :src_path: <str> of where the source file is, must be a video or a directory of images
        :home_dir: <str> the home directory containing subdirectories to read from and write to
        :stream: <boolean> True if the source is a video, False otherwise
        :max_iters: <int> number of images to use in dataset (None by default means use all available)
        :k: <int> factor to scale by
    
    Outpts
        :returns: path to the h5 file containing the generated dataset
    """
    hf_path = f"{home_dir}/data/train.h5"
    try: os.remove(hf_path)
    except Exception: pass

    images = []
    hf = h5py.File(hf_path, 'a')
    data, low_res_data = [], []

    if stream: images = extract_frames(src_path, max_iters=max_iters)
    else: images = read_images(src_path)

    for image_name, image in images.items():
        h, w, _ = image.shape 
        low_res_image = cv2.resize(image, (w // k, h // k))
        # low_res_image = cv2.resize(image, (256 // k, 256 // k))
        # high_res_image = cv2.resize(image, (256, 256))
       
        # splitting frame into 100 tiles of size m x n
        n = 26
        data += _split(image, n)
        low_res_data += _split(low_res_image, n)
        # data.append(np.transpose(high_res_image, (2, 0, 1)).astype(np.float32))
        # low_res_data.append(np.transpose(low_res_image, (2, 0, 1)).astype(np.float32))
    
    hf.create_dataset(name="label", data=np.asarray(data))
    hf.create_dataset(name="data", data=np.asarray(low_res_data))
    hf.close()

    return hf_path

def download(url, home_dir, stream = False, fn = None):
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

    r = requests.get(url, stream=stream)
    if r.status_code == 200:
        with open(f"{home_dir}/inputs/images/{fn}.png", 'wb') as output_file:
            if stream:
                for chunk in r.iter_content(chunk_size=1024**2): 
                    if chunk: output_file.write(chunk)
            else:
                output_file.write(r.content)
                print("{} downloaded: {:.2f} KB".format(fn, len(r.content) / 1024.0))
            return f"{home_dir}/inputs/images/{fn}.png"
    else:
        raise ValueError(f"url not found: {url}")
    
def extract_frames(src_path, max_iters = None):
    """
    Extracts the frames of a video

    Inputs
        :src_path: <str> the path to the video source file
        :max_iters: <int> | None how many frames to get (default is None => get all frames)
    
    Outputs
        :returns: a list of the video frames (represented as ndarrays)
    """
    frames = []
    iter_num = 0
    video = cv2.VideoCapture(src_path)
    while (video.isOpened()):
        if iter_num % 100 == 0: print(f"Super resolving video frame {iter_num} ...")
        ret, frame = video.read()
        iter_num += 1
        if not ret: break
        if max_iters is not None:
            if iter_num >= max_iters: break

        frames.append(frame)
    video.release()

    return frames

def read_images(dir_path):
    """
    Reads images from a directory

    Inputs
        :dir_path: <str> the path to the image directory
    
    Outputs
        :returns: a list of the images (represented as ndarrays)
    """
    images = []
    for image_name in os.listdir(dir_path):
        image = cv2.imread(os.path.join(dir_path, image_name))
        if image is not None: images.append(image)

    return images

### Helper Functions ###
def _split(image, k):
    """
    Splits an image into h // k x w // k smaller images

    Inputs
        :image: <np.ndarray> to be split
        :k: <int> the factor to split the image by 
    
    Outputs
        :returns: a list of smaller images that together form the original image
    """
    h, w, _ = image.shape
    m, n = h // k, w // k
    return [np.transpose(image[x : x + m, y : y + n, ::], (2, 0, 1)).astype(np.float32) for x in range(0, h, m) for y in range(0, w, n)]
