import cv2
import torch
import numpy as np
from .data_utils import download
from .srcnn_utils import create_video
from torchvision.utils import save_image

def test_image(model, device, home_dir, image_path = None):
    """
    Tests out the network against an input image

    Inputs
        :model: <SRCNN> to test 
        :device: the computation device CPU or GPU
        :home_dir: <str> the home directory containing subdirectories to read from and write to
        :image_path: <str> path to the image to test the network on, None by default
     
    Outputs
        :returns: a tuple (before, after) of the path to the before and after images
    """
    image_name = input("Image name: ")
    if image_path is None:
        image_url = input("Image url: ")
        image_path = download(image_url, home_dir, fn=image_name)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(image.shape[0], image.shape[1], 3)
    before_image_path = f"{home_dir}/inputs/images/before_{image_name}.png"

    cv2.imwrite(before_image_path, image)
    image = image / 255.0
    model.eval()

    with torch.no_grad():
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(device)
        image = image.unsqueeze(0)
        outputs = model(image).squeeze(0)
        outputs_path = f"{home_dir}/outputs/images/after_{image_name}.png"

        outputs = outputs.cpu()
        save_image(outputs, outputs_path)
        outputs = outputs.detach().numpy()
        outputs = outputs.reshape(outputs.shape[1], outputs.shape[2], outputs.shape[0])

    return before_image_path, outputs_path

def test_video(model, device, home_dir, video_path = None, max_iters = None):
    """
    Tests out the network against an input video

    Inputs
        :model: <SRCNN> to test 
        :device: the computation device CPU or GPU
        :home_dir: <str> the home directory containing subdirectories to read from and write to
        :video_path: <str> path to the video to test the network on, None by default
        :max_iters: <int> maximum number of frames to concatenate
    
    Outputs
        :returns: a tuple (before, after) of the path to the before and after last video frame
    """
    video_name = input("Video name: ")
    if video_path is None:
        video_url = input("Video url: ")
        video_path = download(video_url, home_dir, stream=True, fn=video_name)
    
    video = cv2.VideoCapture(video_path)

    iter_num = 0
    resolved_frames, low_res_frames = [], []
    while (video.isOpened()):
        if iter_num % 10 == 0: print(f"Super resolving video frame {iter_num} ...")
        ret, frame = video.read()
        iter_num += 1
        if not ret: break
        if max_iters is not None:
            if iter_num > max_iters: break

        input_path = f"{home_dir}/inputs/images/before_image1.png"
         
        _, after_frame = test_image(model, device, home_dir, image_path=input_path)
        after_frame = cv2.imread(after_frame)

        resolved_frames.append(after_frame)
    video.release()
    
    resolved_video_path = create_video(resolved_frames, home_dir)

    return video_path, resolved_video_path
