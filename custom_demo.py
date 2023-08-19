import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar

torch.set_grad_enabled(False)

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

device = 'cuda:0'
network = XMem(config, './saves/XMem.pth').eval().to(device)

cap = cv2.VideoCapture('/home/xuanlin/Downloads/20230815_205733_cse_2_pink_mugs.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
mask = np.load('/home/xuanlin/Downloads/20230815_205733_cse_2_pink_mugs_frame0_masks.npy')
mask = mask * np.arange(1, mask.shape[0] + 1)[:, None, None]
mask = mask.sum(axis=0)
num_objects = len(np.unique(mask)) - 1
print("num objects", num_objects)

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) # consecutive labels

# You can change these two numbers
visualize_every = 1

current_frame_index = 0

visualizations = []
with torch.cuda.amp.autocast(enabled=True):
  for i in range(frameCount):
    print(i)
    # load frame-by-frame
    frame = buf[i]
    if frame is None:
      break

    # convert numpy array to pytorch tensor format
    frame_torch, _ = image_to_torch(frame, device=device)
    if current_frame_index == 0:
      # initialize with the mask
      mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
      # the background mask is not fed into the model
      prediction = processor.step(frame_torch, mask_torch[1:])
    else:
      # propagate only
      prediction = processor.step(frame_torch)

    # argmax, convert to numpy
    prediction = torch_prob_to_numpy_mask(prediction)

    if current_frame_index % visualize_every == 0:
      visualization = overlay_davis(frame, prediction)
      visualizations.append(visualization)

    current_frame_index += 1
    
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('/home/xuanlin/Downloads/20230815_205733_cse_2_pink_mugs_xmem_visualization.mp4', fourcc, 20.0, visualizations[0].shape[:2][::-1])
for i in range(len(visualizations)):
  out.write(visualizations[i])
out.release()