#! /usr/bin/python

from configparser import ConfigParser
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
#    RandCropByPosNegLabeld,
#    RandAffined,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,    
    Spacingd,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm

import glob
import os
import shutil

from monai.data import CacheDataset, DataLoader, Dataset


#--------------------------------------------------------------------------------
# Load configurations
#--------------------------------------------------------------------------------

def getvector(config, section, key):
    value = config.get(section, key)
    if value:
        value = value.split(',')
        value = [float(s) for s in value]
        value = tuple(value)
        return value
    else:
        return None

    
### Following parameters will be used for both training and testing ###
config = ConfigParser()
config.read('config.ini')

val_ratio = config.getfloat('common', 'val_ratio')
data_dir = config.get('common', 'data_dir')
root_dir = config.get('common', 'root_dir')

pixel_dim = getvector(config, 'common', 'pixel_dim')
if pixel_dim == None:
    pixel_dim = (1.0,1.0,1.0)

window_size = getvector(config, 'common', 'window_size')
if window_size:
    window_size = [int(s) for s in window_size]
    window_size = tuple(window_size)
else:
    window_size = (160,160,160)

pixel_intensity_min = config.getfloat('common', 'pixel_intensity_min')
pixel_intensity_max = config.getfloat('common', 'pixel_intensity_max')
pixel_intensity_percentile_min = config.getfloat('common', 'pixel_intensity_percentile_min')
pixel_intensity_percentile_max = config.getfloat('common', 'pixel_intensity_percentile_max')

model_file = config.get('common', 'model_file')

use_tensorboard = int(config.get('training', 'use_tensorboard'))
use_matplotlib = int(config.get('training', 'use_matplotlib'))
max_epochs = int(config.get('training', 'max_epochs'))

training_device_name = config.get('training', 'training_device_name')
inference_device_name = config.get('inference', 'inference_device_name')


#--------------------------------------------------------------------------------
# Validation transform
#--------------------------------------------------------------------------------

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
        # ScaleIntensityRangePercentilesd(
        #     keys=["image"], lower=pixel_intensity_percentile_min, upper=pixel_intensity_percentile_max,
        #     b_min=0.0, b_max=1.0, clip=True,
        # ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)


#--------------------------------------------------------------------------------
# Validation transform
#--------------------------------------------------------------------------------

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

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)


#--------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------

model_unet = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
)

post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = AsDiscrete(to_onehot=True, n_classes=2)
