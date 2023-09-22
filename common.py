#! /usr/bin/python

import torch
from configparser import ConfigParser
from monai.transforms import (
    Activationsd,
    AdjustContrastd,
    AsDiscrete,
    AsDiscreted,
    # AddChanneld, # Mariana: deprecated. Replace with EnsureChannelFirst
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandAdjustContrastd,
    RandFlipd,
    RandZoomd,
    RandScaleIntensityd,
    RandStdShiftIntensityd,
    RemoveSmallObjectsd,
    SaveImaged,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,    
    Spacingd,
    ToTensord,
)

from monai.utils import first, set_determinism
from monai.networks.nets import UNet
from monai.networks.layers import Norm

import numpy as np
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

    def getvector(self, config, section, key, default=None):
        value = None
        if default:
            value = config.get(section, key, fallback=default)
        else:
            value = config.get(section, key)
        if value:
            value = value.split(',')
            value = [float(s) for s in value]
            value = tuple(value)
            return value
        else:
            return None

    def readParameters(self):
 
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

        self.axcodes = self.config.get('common', 'orientation', fallback ='RAS')
        self.in_channels = int(self.config.get('common', 'in_channels'))
        self.out_channels = int(self.config.get('common', 'out_channels'))
        self.input_type = self.config.get('common', 'input_type')
        self.label_type = self.config.get('common', 'label_type')
        self.model_file = self.config.get('common', 'model_file')
        
        
class TrainingParam(Param):
    
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()
        self.use_tensorboard = int(self.config.get('training', 'use_tensorboard'))
        self.use_matplotlib = int(self.config.get('training', 'use_matplotlib'))
        self.training_name = self.config.get('training', 'training_name')
        self.max_epochs = int(self.config.get('training', 'max_epochs', fallback='200'))
        self.training_device_name = self.config.get('training', 'training_device_name')
        self.training_rand_flip = int(self.config.get('training', 'random_flip', fallback='0'))
        self.training_rand_zoom = float(self.config.get('training', 'random_zoom', fallback='0.0'))

class TestParam(Param):

    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()

        self.test_device_name = self.config.get('test', 'test_device_name')

        
class TransferParam(TrainingParam):
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()
        self.tl_model_file = self.config.get('transfer', 'tl_model_file')
        self.tl_name = self.config.get('transfer', 'tl_name', fallback='transfer_learning_1')
        self.tl_data_dir = self.config.get('transfer', 'tl_data_dir')
    
        
class InferenceParam(Param):
    
    def __init__(self, filename='config.ini'):
        super().__init__(filename)

    def readParameters(self):
        super().readParameters()
        self.inference_device_name = self.config.get('inference', 'inference_device_name')
        self.min_size_object = self.config.get('inference', 'min_size_object')



#--------------------------------------------------------------------------------
# Load Transforms
#--------------------------------------------------------------------------------

def loadTrainingTransforms(param):
    # Load images
    if param.in_channels==2:
        # Two channels input
        transform_array = [
            LoadImaged(keys=["image_1", "image_2", "label"]),
            EnsureChannelFirstd(keys=["image_1", "image_2", "label"]), # Mariana: AddChanneld(keys=["image", "label"]) deprecated, use EnsureChannelFirst instead
            ConcatItemsd(keys=["image_1", "image_2"], name="image")
        ]
    else:
        # One channel input
        transform_array = [            
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'), # Mariana: AddChanneld(keys=["image", "label"]) deprecated, use EnsureChannelFirst instead
        ]

    # Intensity adjustment
    if (param.input_type == 'R') or (param.input_type == 'I'):
        transform_array.append(AdjustContrastd(keys=["image"], gamma=2.5))
    transform_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))

    # Spatial adjustments
    transform_array.append(Orientationd(keys=["image", "label"], axcodes=param.axcodes))
    transform_array.append(Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")))

    # Data augmentation
    if param.training_rand_flip == 1:
        transform_array.append(RandZoomd(
                keys=['image', 'label'],
                prob=0.5,
                min_zoom=1.02,
                max_zoom=1.20,
                mode=['area', 'nearest'],
            ))
    if param.training_rand_zoom == 1:
        transform_array.append(RandFlipd(
            keys=['image', 'label'],
            prob=0.5,
            spatial_axis=2,
        ))

    # Balance background/foreground
    transform_array.append(RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=param.window_size,
        pos=5, 
        neg=1,
        num_samples=5,
        image_key="image",
        image_threshold=0, 
    ))

    train_transforms = Compose(transform_array)
    return train_transforms

def loadValidationTransforms(param):    
    # Load images
    if param.in_channels==2:
        # Two channels input
        val_array = [
            LoadImaged(keys=["image_1", "image_2", "label"]),
            EnsureChannelFirstd(keys=["image_1", "image_2", "label"]), # Mariana: AddChanneld(keys=["image", "label"]) deprecated, use EnsureChannelFirst instead
            ConcatItemsd(keys=["image_1", "image_2"], name="image")
        ]
    else:
        # One channel input
        val_array = [            
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'), # Mariana: AddChanneld(keys=["image", "label"]) deprecated, use EnsureChannelFirst instead
        ]

    # Intensity adjustment
    if (param.input_type == 'R') or (param.input_type == 'I'):
        val_array.append(AdjustContrastd(keys=["image"], gamma=2.5))
    val_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))

    # Spatial adjustments
    val_array.append(Orientationd(keys=["image", "label"], axcodes=param.axcodes))
    val_array.append(Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")))
    # val_array.append(CropForegroundd(keys=["image", "label"], source_key="image"))
    # val_array.append(ToTensord(keys=["image", "label"]))
    # val_array.append(EnsureTyped(keys=["image", "label"]))
    val_transforms = Compose(val_array)
    return val_transforms

def loadInferenceTransforms(param, output_path):
    if param.in_channels==2:
        pre_array = [
            # 2-channel input
            LoadImaged(keys=["image_1", "image_2"]),
            EnsureChannelFirstd(keys=["image_1", "image_2"]), 
            ConcatItemsd(keys=["image_1", "image_2"], name="image"),
        ]        
    else:
        pre_array = [
            # 1-channel input
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        ]
    pre_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))
    pre_array.append(Orientationd(keys=["image"], axcodes=param.axcodes))
    pre_array.append(Spacingd(keys=["image"], pixdim=param.pixel_dim, mode=("bilinear")))
    # pre_array.append(CropForegroundd(keys=["image"], source_key="image"))
        #ToTensord(keys=["image"]),
    # pre_array.append(EnsureTyped(keys=["image"]))
    pre_transforms = Compose(pre_array)

    # define post transforms
    post_transforms = Compose([
        # EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=pre_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                                # then invert `pred` based on this information. we can use same info
                                # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
                                             # for example, may need the `affine` to invert `Spacingd` transform,
                                             # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                                           # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                                           # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                   # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        Activationsd(keys="pred", sigmoid=True),
        #AsDiscreted(keys="pred", threshold_values=True),
        AsDiscreted(keys="pred", argmax=True, num_classes=param.out_channels),
        RemoveSmallObjectsd(keys="pred", min_size=int(param.min_size_object), connectivity=1, independent_channels=False),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False),
    ])
    
    return (pre_transforms, post_transforms)


#--------------------------------------------------------------------------------
# Generate a file list
#--------------------------------------------------------------------------------

def generateLabeledFileList(param, prefix):
    print('Reading labeled images from: ' + param.data_dir)
    images_m = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_images", "*_M.nii.gz")))
    images_p = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_images", "*_P.nii.gz")))
    images_r = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_images", "*_R.nii.gz")))
    images_i = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_images", "*_I.nii.gz")))
    labels = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_labels", "*_"+param.label_type+"_label.nii.gz")))
    
    # Use two types of images combined
    if param.in_channels==2:
        # Use real and imaginary images
        if param.input_type=='R' or param.input_type=='I':
            data_dicts = [
                {"image_1": image_r_name, "image_2": image_i_name, "label":label_name}
                for image_r_name, image_i_name, label_name in zip(images_r, images_i, labels)
            ]
        # Use magnitude and phase images
        else:
            data_dicts = [
                {"image_1": image_m_name, "image_2": image_p_name, "label":label_name}
                for image_m_name, image_p_name, label_name in zip(images_m, images_p, labels)
            ]
    # Use only one type of image        
    else:
        # Use real images
        if param.input_type=='R':
            data_dicts = [
                {"image": image_name, "label": label_name}
                for image_name, label_name in zip(images_r, labels)
            ]
        # Use imaginary images
        elif param.input_type=='I':
            data_dicts = [
                {"image": image_name, "label": label_name}
                for image_name, label_name in zip(images_i, labels)
            ]
        # Use phase images
        elif param.input_type=='P':
            data_dicts = [
                {"image": image_name, "label": label_name}
                for image_name, label_name in zip(images_p, labels)
            ]
        # Use magnitude images
        else:
            data_dicts = [
                {"image": image_name, "label": label_name}
                for image_name, label_name in zip(images_m, labels)
            ]
    return data_dicts    

def generateFileList(param, input_path):
    print('Reading images from: ' + param.data_dir)
    images_m = sorted(glob.glob(os.path.join(param.data_dir, input_path, "*_M.nii.gz")))
    images_p = sorted(glob.glob(os.path.join(param.data_dir, input_path, "*_P.nii.gz")))
    images_r = sorted(glob.glob(os.path.join(param.data_dir, input_path, "*_R.nii.gz")))
    images_i = sorted(glob.glob(os.path.join(param.data_dir, input_path, "*_I.nii.gz")))
    
    # Use two types of images combined
    if param.in_channels==2:
        # Use real and imaginary images
        if param.input_type=='R' or param.input_type=='I':
            data_dicts = [
                {"image_1": image_r_name, "image_2": image_i_name}
                for image_r_name, image_i_name in zip(images_r, images_i)
            ]
        # Use magnitude and phase images
        else:
            data_dicts = [
                {"image_1": image_m_name, "image_2": image_p_name}
                for image_m_name, image_p_name in zip(images_m, images_p)
            ]    
    # Use only one type of image        
    else:
        # Use real images
        if param.input_type=='R':
            data_dicts = [
                {"image": image_name} for image_name in images_r
            ]
        # Use imaginary images
        elif param.input_type=='I':
            data_dicts = [
                {"image": image_name} for image_name in images_i
            ]
        # Use phase images
        elif param.input_type=='P':
            data_dicts = [
                {"image": image_name} for image_name in images_p
            ]
        # Use magnitude images
        else:
            data_dicts = [
                {"image": image_name} for image_name in images_m
            ]
    return data_dicts
    

#--------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------

def setupModel(param):

    model_unet = UNet(
        spatial_dims=3, # Mariana: dimensions=3 was deprecated
        in_channels=param.in_channels,
        out_channels=param.out_channels,
        # channels=(16, 32, 64, 128, 256),  # Mariana: Reduce the number of channels to fit the smaller spatial dimensions
        # strides=(2, 2, 2, 2),             # Mariana: Adjust the strides based on the desired downsampling (keep depth untouched)
        channels=[16, 32, 64, 128], 
        strides=[(1, 2, 2), (1, 2, 2), (1, 1, 1)], 
        num_res_units=2,
        norm=Norm.BATCH,
    )
    
    post_pred = AsDiscrete(argmax=True, to_onehot=param.out_channels, n_classes=param.out_channels)
    post_label = AsDiscrete(to_onehot=param.out_channels, n_classes=param.out_channels)
    #post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=param.out_channels)  # Mariana: Deprecated error
    #post_label = AsDiscrete(to_onehot=True, n_classes=param.out_channels)              # Mariana: Deprecated error

    return (model_unet, post_pred, post_label)
