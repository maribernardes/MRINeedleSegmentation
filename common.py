#! /usr/bin/python

import torch
from configparser import ConfigParser
from monai.transforms import (
    Activationsd,
    AdjustContrastd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandBiasFieldd,
    RandAffined,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandRicianNoised,
    RandFlipd,
    RandZoomd,
    RandRotated,
    RandScaleIntensityd,
    RandStdShiftIntensityd,
    RandKSpaceSpikeNoised,
    RemoveSmallObjectsd,
    SaveImage,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,    
    Spacingd,
    ToTensord,
)

from monai.utils import first, set_determinism
from monai.networks.nets import UNet, BasicUNetPlusPlus
from monai.networks.layers import Norm

import numpy as np
import random
import glob
import os
import sys
import shutil

from monai.data import CacheDataset, DataLoader, Dataset

script_directory = os.getcwd()
path = os.path.dirname(script_directory)
sitkTools_path = os.path.join(path, 'SitkTools')
sys.path.append(sitkTools_path)
from sitkMonaiIO import *


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
        self.training_rand_bias = float(self.config.get('training', 'random_bias', fallback='0.0'))
        self.training_rand_noise = float(self.config.get('training', 'random_noise', fallback='0.0'))
        self.training_spike_noise = float(self.config.get('training', 'random_spike', fallback='0.0'))
        self.training_rand_flip = float(self.config.get('training', 'random_flip', fallback='0.0'))
        self.training_rand_rotation = float(self.config.get('training', 'random_rotation', fallback='3.14'))
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


#--------------------------------------------------------------------------------
# Load Transforms
#--------------------------------------------------------------------------------

def loadTrainingTransforms(param):
    # Load images
    transform_array = [
        LoadImaged(keys=["image_1", "image_2", "label"], image_only=False),                 # Load magnitude, phase and labelmap
        EnsureChannelFirstd(keys=["image_1", "image_2", "label"]),                          # Ensure channel first
        ScaleIntensityd(keys=["image_1"], minv=0, maxv=1, channel_wise=True),               # Scale magnitude intensity to 0-1
        ScaleIntensityd(keys=["image_2"], minv=-torch.pi, maxv=torch.pi, channel_wise=True) # Scale phase intensity to -pi +pi
    ]
    # Field bias addition
    if param.training_rand_bias != 0:
        transform_array.append(RandBiasFieldd(keys=["image_1"], prob=param.training_rand_bias, coeff_range=(0.2, 0.3)))     # Add Field Bias to Magnitude
        transform_array.append(ScaleIntensityd(keys=["image_1"], minv=0, maxv=1, channel_wise=True)) # Re-scale intensity after noise addition
    # White noise addition
    if param.training_rand_noise != 0:
        if random.random() < param.training_rand_noise: # Probability of adding noise 
            transform_array.append(RandRicianNoised(keys=["image_1"], prob=param.training_rand_noise, mean=0, std=0.1))     # Add Rician noise to Magnitude -  mean=0, std=0.1
            transform_array.append(RandGaussianNoised(keys=["image_2"], prob=param.training_rand_noise, mean=0, std=0.08))  # Add small Gaussian noise to Phase - mean=0, std=0.08
            transform_array.append(ScaleIntensityd(keys=["image_1"], minv=0, maxv=1, channel_wise=True))                    # Re-scale intensity after noise addition
            transform_array.append(ScaleIntensityd(keys=["image_2"], minv=-torch.pi, maxv=torch.pi, channel_wise=True))     # Re-scale intensity after noise addition
    # Combine images
    transform_array.append(ConcatItemsd(keys=["image_1", "image_2"], name="image"))     # Concatenate Magnitude and Phase to 2-channels   
    # Spike noise addition
    if param.training_spike_noise != 0:
        transform_array.append(RandKSpaceSpikeNoiseMagPhased(keys=["image"], prob=param.training_spike_noise, reorder_axes=True, intensity_range=(0.95*3.5, 1.01*3.5)))
        transform_array.append(ScaleIntensityPerChanneld(keys=["image"], min_max_values=[(0, 1), (-torch.pi, torch.pi)])) # Re-scale intensity after noise addition
    # Convert to real/imaginary
    if (param.input_type == 'R') or (param.input_type == 'I'):
        transform_array.append(MagPhaseToRealImagd(keys=["image"]))  
        print('Use real/imaginary images')
    else:
        print('Use magnitude/phase images') 

    # ONE channel input
    if param.in_channels==1:
        print('Use 1 CHANNEL')
        if (param.input_type == 'M') or (param.input_type == 'R'):
            print('Use first channel')
            transform_array.append(SelectChanneld(keys="image", indices=0))  # Use the first channel only
        else:
            print('Use second channel')
            transform_array.append(SelectChanneld(keys="image", indices=1))  # Use the second channel only
    # TWO channels input
    elif param.in_channels==2:
        print('Use 2 CHANNELS')
    else:
        print('Use MORE CHANNELS not implemented yet')

    # Adjust intensity and scaling of final image
    transform_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))  

    # Spatial adjustments
    if param.axcodes != 'NO':
        transform_array.append(Orientationd(keys=["image", "label"], axcodes=param.axcodes))                            # Adjust image orientation
    else:
        print('No Orientationd')
    transform_array.append(Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")))     # Adjust image spacing

    # Data augmentation
    if param.training_rand_flip != 0:
        transform_array.append(RandZoomd(
            keys=['image', 'label'],
            prob=param.training_rand_flip,
            min_zoom=1.02,
            max_zoom=1.20,
            mode=['area', 'nearest'],
        ))
    if param.training_rand_zoom != 0:
        transform_array.append(RandFlipd(
            keys=['image', 'label'],
            prob=param.training_rand_zoom,
            spatial_axis=2,
        ))

    if param.training_rand_rotation != 0:
        transform_array.append(RandRotated(
            keys=['image', 'label'],
            prob=param.training_rand_rotation,
            range_x = 3.14,
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
    val_array = [
        LoadImaged(keys=["image_1", "image_2", "label"], image_only=False),                 # Load magnitude, phase and labelmap
        EnsureChannelFirstd(keys=["image_1", "image_2", "label"]),                          # Ensure channel first
        ScaleIntensityd(keys=["image_1"], minv=0, maxv=1, channel_wise=True),               # Scale magnitude intensity to 0-1
        ScaleIntensityd(keys=["image_2"], minv=-torch.pi, maxv=torch.pi, channel_wise=True) # Scale phase intensity to -pi +pi
    ]
    # Combine images
    val_array.append(ConcatItemsd(keys=["image_1", "image_2"], name="image"))     # Concatenate Magnitude and Phase to 2-channels   
    # Convert to real/imaginary
    if (param.input_type == 'R') or (param.input_type == 'I'):
        val_array.append(MagPhaseToRealImagd(keys=["image"]))  
        print('Use real/imaginary images')
    else:
        print('Use magnitude/phase images') 

    # ONE channel input
    if param.in_channels==1:
        print('Use 1 CHANNEL')
        if (param.input_type == 'M') or (param.input_type == 'R'):
            print('Use first channel')
            val_array.append(SelectChanneld(keys="image", indices=0))  # Use the first channel only
        else:
            print('Use second channel')
            val_array.append(SelectChanneld(keys="image", indices=1))  # Use the second channel only
    # TWO channels input
    elif param.in_channels==2:
        print('Use 2 CHANNELS')
    else:
        print('Use MORE CHANNELS not implemented yet')

    # Adjust intensity and scaling of final image
    val_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))  

    # Spatial adjustments
    if param.axcodes != 'NO':
        val_array.append(Orientationd(keys=["image", "label"], axcodes=param.axcodes))                            # Adjust image orientation
    else:
        print('No Orientationd')
    val_array.append(Spacingd(keys=["image", "label"], pixdim=param.pixel_dim, mode=("bilinear", "nearest")))     # Adjust image spacing
    val_transforms = Compose(val_array)
    return val_transforms
    
def loadInferenceTransforms(param, output_path):
    # Define pre-inference transforms
    pre_array = [
        LoadImaged(keys=["image_1", "image_2"], image_only=False),                          # Load magnitude and phase
        EnsureChannelFirstd(keys=["image_1", "image_2"]),                                   # Ensure channel first (EnsureChannelFirstd(keys=["image"], channel_dim='no_channel')
        ScaleIntensityd(keys=["image_1"], minv=0, maxv=1, channel_wise=True),               # Scale magnitude intensity to 0-1
        ScaleIntensityd(keys=["image_2"], minv=-torch.pi, maxv=torch.pi, channel_wise=True) # Scale phase intensity to -pi +pi
    ]
    # Combine images
    pre_array.append(ConcatItemsd(keys=["image_1", "image_2"], name="image"))     # Concatenate Magnitude and Phase to 2-channels   
    # Convert to real/imaginary
    if (param.input_type == 'R') or (param.input_type == 'I'):
        pre_array.append(MagPhaseToRealImagd(keys=["image"]))  
        print('Use real/imaginary images')
    else:
        print('Use magnitude/phase images') 

    # ONE channel input
    if param.in_channels==1:
        print('Use 1 CHANNEL')
        if (param.input_type == 'M') or (param.input_type == 'R'):
            print('Use first channel')
            pre_array.append(SelectChanneld(keys="image", indices=0))  # Use the first channel only
        else:
            print('Use second channel')
            pre_array.append(SelectChanneld(keys="image", indices=1))  # Use the second channel only
    # TWO channels input
    elif param.in_channels==2:
        print('Use 2 CHANNELS')
    else:
        print('Use MORE CHANNELS not implemented yet')

    # Adjust intensity and scaling of final image
    pre_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))  

    # Spatial adjustments
    if param.axcodes != 'NO':
        pre_array.append(Orientationd(keys=["image"], axcodes=param.axcodes))                 # Adjust image orientation
    else:
        print('No Orientationd')
    pre_array.append(Spacingd(keys=["image"], pixdim=param.pixel_dim, mode=("bilinear")))     # Adjust image spacing
    pre_transforms = Compose(pre_array)

    
    # Define post-inference transforms
    post_array = [AsDiscrete(argmax=True, n_classes=param.out_channels),
                SaveImage(output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False, print_log=False)
    ]
    post_transforms = Compose(post_array)
    return (pre_transforms, post_transforms)

#--------------------------------------------------------------------------------
# Generate a file list
#--------------------------------------------------------------------------------

def generateLabeledFileList(param, prefix):
    print('Reading labeled images from: ' + param.data_dir)
    images_m = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_images", "*_M.nii.gz")))
    images_p = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_images", "*_P.nii.gz")))
    labels = sorted(glob.glob(os.path.join(param.data_dir, prefix + "_labels", "*_"+param.label_type+"_label.nii.gz")))

    # Use magnitude and phase images
    data_dicts = [
        {"image_1": image_m_name, "image_2": image_p_name, "label":label_name}
        for image_m_name, image_p_name, label_name in zip(images_m, images_p, labels)
    ]
    return data_dicts    

def generateFileList(param, input_path):
    print('Reading images from: ' + input_path)
    images_m = sorted(glob.glob(os.path.join(input_path, "*_M.nii.gz")))
    images_p = sorted(glob.glob(os.path.join(input_path, "*_P.nii.gz")))
    
    data_dicts = [
        {"image_1": image_m_name, "image_2": image_p_name}
        for image_m_name, image_p_name in zip(images_m, images_p)
    ]    
    return data_dicts
    

#--------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------

def setupModel(param):

    if param.axcodes == 'PIL':
        strides = [(1, 2, 2), (1, 2, 2), (1, 1, 1)]   # PIL
    else:    
        strides = [(2, 2, 1), (2, 2, 1), (1, 1, 1)]   # Make other options according to chosen orientation
    model_unet = UNet(
        spatial_dims=3, 
        in_channels=param.in_channels,
        out_channels=param.out_channels,
        channels=[16, 32, 64, 128],                 # This is a Unet with 4 layers
        strides= strides,  
        num_res_units=2,
        norm=Norm.BATCH,
    )
    
    post_pred = AsDiscrete(argmax=True, to_onehot=param.out_channels, n_classes=param.out_channels) # MARIANA
    post_label = AsDiscrete(to_onehot=param.out_channels, n_classes=param.out_channels)             # MARIANA
    
    # post_pred = AsDiscrete(argmax=True, n_classes=param.out_channels)
    # post_label = AsDiscrete(n_classes=param.out_channels)
    
    return (model_unet, post_pred, post_label)
