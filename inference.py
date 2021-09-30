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
    Activationsd,
    EnsureTyped,
    Invertd,
    AsDiscreted,
    SaveImaged,
)
from monai.handlers.utils import from_engine
from monai.metrics import compute_meandice, DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, NiftiSaver, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract

import numpy as np
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import sys
import argparse

from configparser import ConfigParser

from common import *



def run(param, output_path, image_type, val_files):

    device = torch.device(param.inference_device_name)

    val_transforms =  loadInferenceTransforms(param)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    #val_ds = Dataset(data=val_files, transform=val_transforms)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
 
    
    #--------------------------------------------------------------------------------
    # Model
    #--------------------------------------------------------------------------------
    
    (model_unet, post_pred, post_label) = setupModel(param)
    
    model = model_unet.to(device)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    model.load_state_dict(torch.load(os.path.join(param.root_dir, param.model_file), map_location=device))



    #--------------------------------------------------------------------------------
    # Transforms
    #--------------------------------------------------------------------------------

    pre_transforms = val_transforms
    
    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
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
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False),
    ])

    
    #--------------------------------------------------------------------------------
    # Validate
    #--------------------------------------------------------------------------------
    
    model.eval()
    
    with torch.no_grad():
    
        #saver = NiftiSaver(output_dir=output_path, separate_folder=False)
        metric_sum = 0.0
        metric_count = 0
        
        for i, val_data in enumerate(val_loader):
            roi_size = param.window_size
            sw_batch_size = 4
            
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs = from_engine(["pred"])(val_data)
            #val_output_label = torch.argmax(val_outputs, dim=1, keepdim=True)
            #saver.save_batch(val_output_label, val_data['image_meta_dict'])            

            

def main(argv):
  try:
    parser = argparse.ArgumentParser(description="Apply a saved DL model for segmentation.")
    parser.add_argument('cfg', metavar='CONFIG_FILE', type=str, nargs=1,
                        help='Configuration file')
    parser.add_argument('input', metavar='INPUT_PATH', type=str, nargs=1,
                        help='A file or a folder that contains images.')
    parser.add_argument('output', metavar='OUTPUT_PATH', type=str, nargs=1,
                        help='A folder to store the output file(s).')
    parser.add_argument('-t', dest='type', default='folder',
                        help="Image type ('file': a file; 'folder': a folder containing multiple images.)")
            
    args = parser.parse_args(argv)

    config_file = args.cfg[0]
    input_path = args.input[0]
    output_path = args.output[0]
    image_type = args.type

    # Make the destination directory, if it does not exists.
    #os.makedirs(output_path, exist_ok=True)

    print('Loading parameters from: ' + config_file)
    param = InferenceParam(config_file)
    files = generateFileList(input_path)
    n_files = len(files)
    print('# of images: ' + str(n_files))

    run(param, output_path, image_type, files)


  except Exception as e:
    print(e)
  sys.exit()


if __name__ == "__main__":
  main(sys.argv[1:])
