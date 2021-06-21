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
from monai.utils import first, set_determinism
from monai.networks.nets import UNet
from monai.networks.layers import Norm

import glob
import os
import shutil

from monai.data import CacheDataset, DataLoader, Dataset


#--------------------------------------------------------------------------------
# Load configurations
#--------------------------------------------------------------------------------

class Param():

    def __init__(self, filename='config.ini'):
        self.config = ConfigParser()
        self.config.read(filename)
        self.readParameters()

    def getvector(self, config, section, key):
        value = config.get(section, key)
        if value:
            value = value.split(',')
            value = [float(s) for s in value]
            value = tuple(value)
            return value
        else:
            return None

    def readParameters(self):
 
        self.val_ratio = self.config.getfloat('common', 'val_ratio')
        self.data_dir = self.config.get('common', 'data_dir')
        self.root_dir = self.config.get('common', 'root_dir')
        
        self.pixel_dim = self.getvector(self.config, 'common', 'pixel_dim')
        if self.pixel_dim == None:
            self.pixel_dim = (1.0,1.0,1.0)
        
        self.window_size = self.getvector(self.config, 'common', 'window_size')
        if self.window_size:
            self.window_size = [int(s) for s in self.window_size]
            self.window_size = tuple(self.window_size)
        else:
            self.window_size = (160,160,160)
        
        self.pixel_intensity_min = self.config.getfloat('common', 'pixel_intensity_min')
        self.pixel_intensity_max = self.config.getfloat('common', 'pixel_intensity_max')
        self.pixel_intensity_percentile_min = self.config.getfloat('common', 'pixel_intensity_percentile_min')
        self.pixel_intensity_percentile_max = self.config.getfloat('common', 'pixel_intensity_percentile_max')
        
        self.model_file = self.config.get('common', 'model_file')


class TrainingParam(Param):
    
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()

        self.use_tensorboard = int(self.config.get('training', 'use_tensorboard'))
        self.use_matplotlib = int(self.config.get('training', 'use_matplotlib'))
        self.max_epochs = int(self.config.get('training', 'max_epochs'))
        self.training_device_name = self.config.get('training', 'training_device_name')

        
class InferenceParam(Param):
    
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()

        self.inference_device_name = self.config.get('inference', 'inference_device_name')


#--------------------------------------------------------------------------------
# Validation transform
#--------------------------------------------------------------------------------

def loadValidationTransforms(param):
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="LPS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=param.pixel_intensity_min, a_max=param.pixel_intensity_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            # ScaleIntensityRangePercentilesd(
            #     keys=["image"], lower=param.pixel_intensity_percentile_min, upper=param.pixel_intensity_percentile_max,
            #     b_min=0.0, b_max=1.0, clip=True,
            # ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return val_transforms



# def getLoader(param, data_dicts, transforms):
#     
#     ds = CacheDataset(
#         data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)
#     # val_ds = Dataset(data=val_files, transform=val_transforms)
#     loader = DataLoader(ds, batch_size=1, num_workers=4)
# 
#     # # TODO: THe following lines are not needed.
#     # check_ds = Dataset(data=data_dicts, transform=transforms)
#     # check_loader = DataLoader(check_ds, batch_size=1)
#     # print(check_loader)
#     # check_data = first(check_loader)
#     # image, label = (check_data["image"][0][0], check_data["label"][0][0])
#     # print(f"image shape: {image.shape}, label shape: {label.shape}")
#     
#     return loader


def generateFileList(param, prefix):
    
    print('Reading data from: ' + param.data_dir)
    images = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_labels", "*.nii.gz")))
    
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]

    return data_dicts



#--------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------

def setupModel():

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

    return (model_unet, post_pred, post_label)
