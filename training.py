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
    RandAffined,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from configparser import ConfigParser

# Tensorboard
writer = SummaryWriter('runs/segmentation_experiment_1')


### Following parameters will be used for both training and testing ###
config = ConfigParser()
config.read('config.ini')

val_ratio = config.getfloat('main', 'val_ratio')
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

#######################################################################

torch.multiprocessing.set_sharing_strategy('file_system')
print_config()

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

set_determinism(seed=0)



train_transforms = Compose(
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
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            #spatial_size=(96,96,96),
            #spatial_size=(32, 32, 16),
            spatial_size=(80, 80, 16),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        #RandAffined(
        #    keys=['image', 'label'],
        #    mode=('bilinear', 'nearest'),
        #    prob=1.0,
        #    #spatial_size=(96, 96, 96),
        #    spatial_size=(64, 64, 16),
        #    rotate_range=(0, 0, np.pi/15),
        #    scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ]
)

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

check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
print(check_loader)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
sl = 8
# # plot the slice [:, :, sl]
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:, :, sl], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# plt.imshow(label[:, :, sl])
# plt.show()

# Check data sizes
#check_ds = Dataset(data=train_files, transform=train_transforms)
#check_loader = DataLoader(check_ds, batch_size=1)
#for check_data in check_loader:
#    image, label = (check_data["image"][0][0], check_data["label"][0][0])
#    print(f"image shape: {image.shape}, label shape: {label.shape}")
#

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=1.0, num_workers=4)
#train_ds = Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)



# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
#device = torch.device("cpu")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)


max_epochs = 600
#max_epochs = 600
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = AsDiscrete(to_onehot=True, n_classes=2)

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = post_pred(val_outputs)
                val_labels = post_label(val_labels)
                value = compute_meandice(
                    y_pred=val_outputs,
                    y=val_labels,
                    include_background=False,
                )
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
                )
            
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Mean Dice", metric, epoch)

            # write to tensorboard
            #img_grid = torchvision.utils.make_grid(val_labels)
            #writer.add_image('segmentation', img_grid)
            writer.flush()

            


print(f"train completed, best_metric: {best_metric:.4f} "
      f"at epoch: {best_metric_epoch}")

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
