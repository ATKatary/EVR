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
    hf_path = f"{home_dir}/inputs/train.h5"
    try: os.remove(hf_path)
    except Exception: pass

    iter_num = 0
    hf = h5py.File(hf_path, 'a')
    images, low_res_images = [], []

    if stream:
        video = cv2.VideoCapture(src_path)
        while (video.isOpened()):
            if iter_num % 100 == 0: print(f"Super resolving video frame {iter_num} ...")
            ret, frame = video.read()
            iter_num += 1
            if not ret: break
            if max_iters is not None:
                if iter_num >= max_iters: break

            images.append(frame)

        video.release()
    else:
        for image_name in os.listdir(src_path):
            image = cv2.imread(os.path.join(src_path, image_name))
            if image is not None: images.append(image)

    for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape 
            # splitting frame into 600 tiles of size m x n
            m, n = h // 20, w // 30
            tiles = [image[x : x + m, y : y + n] for x in range(0, h, m) for y in range(0, w, n)]
            
            for tile in tiles:
                h, w, _ = tile.shape 
                low_res_tile = cv2.resize(tile,  (w // k, h // k))
                low_res_tile = cv2.resize(low_res_tile,  (w, h))
                
                images.append(np.transpose(tile, (2, 0, 1)).astype(np.float32))
                low_res_images.append(np.transpose(low_res_tile, (2, 0, 1)).astype(np.float32))
            
    hf.create_dataset(name="label", data=np.asarray(images))
    hf.create_dataset(name="data", data=np.asarray(low_res_images))
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
        with open(f"{home_dir}/outputs/{fn}", 'wb') as output_file:
            if stream:
                for chunk in r.iter_content(chunk_size=1024**2): 
                    if chunk: output_file.write(chunk)
            else:
                output_file.write(r.content)
                print("{} downloaded: {:.2f} KB".format(fn, len(r.content) / 1024.0))
            return f"{home_dir}/outputs/{fn}"
    else:
        raise ValueError(f"url not found: {url}")