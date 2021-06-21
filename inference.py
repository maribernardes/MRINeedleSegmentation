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
import sys
import argparse

from configparser import ConfigParser

from common import *



def run(param, input_path, output_path, image_type, val_files):

    val_transforms =  loadValidationTransforms(param)

    #val_loader = getValidationLoader(param)
    #val_loader = getLoader(param, val_files, val_transforms)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    

    #--------------------------------------------------------------------------------
    # Model
    #--------------------------------------------------------------------------------

    (model_unet, post_pred, post_label) = setupModel()
    
    device = torch.device(param.inference_device_name)
    model = model_unet.to(device)
    
    #dice_metric = DiceMetric(include_background=True, reduction="mean")
    #post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    
    #model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.load_state_dict(torch.load(os.path.join(param.root_dir, param.model_file)))
    
    
    #--------------------------------------------------------------------------------
    # Validate
    #--------------------------------------------------------------------------------
    
    model.eval()
    
    with torch.no_grad():
    
        saver = NiftiSaver(output_dir=output_path)
        metric_sum = 0.0
        metric_count = 0
        
        for i, val_data in enumerate(val_loader):
            roi_size = param.window_size
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

    val_files = generateFileList(param, 'val')
    n_val = len(val_files)
    
    print('Validation data size: ' + str(n_val))
    
    run(param, input_path, output_path, image_type, val_files)


  except Exception as e:
    print(e)
  sys.exit()


if __name__ == "__main__":
  main(sys.argv[1:])
