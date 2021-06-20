#! /usr/bin/python

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    Activations,
)
from monai.metrics import compute_meandice, DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, NiftiSaver
from monai.config import print_config
from monai.apps import download_and_extract

import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from configparser import ConfigParser

from common import *

device = torch.device("cuda:0")
#device = torch.device("cpu")

model = model_unet.to(device)

#dice_metric = DiceMetric(include_background=True, reduction="mean")

post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = AsDiscrete(to_onehot=True, n_classes=2)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()

with torch.no_grad():

    saver = NiftiSaver(output_dir="./output")
    metric_sum = 0.0
    metric_count = 0
    
    for i, val_data in enumerate(val_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4

        val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        val_outputs = post_pred(val_outputs)
        #val_outputs = post_trans(val_outputs)        
        val_labels = post_label(val_labels)
        value = compute_meandice(
            y_pred=val_outputs,
            y=val_labels,
            include_background=False,
        )
        print(value)
        metric_count += len(value)
        metric_sum += value.sum().item()        
        #metric_sum += value.item() * len(value)
        
        # # plot the slice [:, :, 80]
        # sl = 15
        # plt.figure("check", (18, 6))
        # plt.subplot(1, 3, 1)
        # plt.title(f"image {i}")
        # plt.imshow(val_data["image"][0, 0, :, :, sl], cmap="gray")
        # plt.subplot(1, 3, 2)
        # plt.title(f"label {i}")
        # plt.imshow(val_data["label"][0, 0, :, :, sl])
        # plt.subplot(1, 3, 3)
        # plt.title(f"output {i}")
        # plt.imshow(torch.argmax(
        #     val_outputs, dim=1).detach().cpu()[0, :, :, sl])
        # plt.show()
        val_output_label = torch.argmax(val_outputs, dim=1, keepdim=True)
        saver.save_batch(val_output_label, val_data['image_meta_dict'])
        
    metric = metric_sum / metric_count
    print("evaluation metric:", metric)

