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
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import compute_meandice, DiceMetric
from monai.losses import DiceLoss
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

device = torch.device("cuda:0")
#device = torch.device("cpu")


from configparser import ConfigParser
config = ConfigParser()


### Following parameters must come from training.py ###

# val_ratio = 0.1
# data_dir = './sorted_nii'
# 
# root_dir = '.'
# 
# pd = (0.9375*2, 0.9375*2, 3.6)
# 
# intensity_max=250

val_ratio = config.getfloat('main', 'val_ratio'))
data_dir = config.get('main', 'data_dir')
root_dir = config.get('main', 'root_dir')
pixel_dim = config.get('main', 'pixel_dim')
if pixel_dim:
    pixel_dim = pixel_dim.split(',')
    pixel_dim = [float(s) for s in pixel_dim]
    pixel_dim = tuple(pixel_dim)
else:
    pixel_dim = (1.0,1.0,1.0)
pixel_intensity_min = config.getfloat('main', 'pixel_intensity_min')
pixel_intensity_max = config.getfloat('main', 'pixel_intensity_max')


model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

#######################################################

print('Reading data from: ' + data_dir)

train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

n_total = len(data_dicts)
n_val = round(n_total * val_ratio)

print('Total data size:      ' + str(n_total))
print('Validation data size: ' + str(n_val))

train_files, val_files = data_dicts[:-n_val], data_dicts[-n_val:]
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixel_dim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=pixel_intensity_min, a_max=pixel_intensity_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
#dice_metric = DiceMetric(include_background=True, reduction="mean")

post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = AsDiscrete(to_onehot=True, n_classes=2)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model.pth")))
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

