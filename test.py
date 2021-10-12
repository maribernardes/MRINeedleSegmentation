#! /usr/bin/python

from monai.utils import first, set_determinism
from monai.metrics import compute_meandice
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import sys
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from common import *


def run(param, test_files):
    
    #--------------------------------------------------------------------------------
    # Test datasets
    #--------------------------------------------------------------------------------

    val_transforms = loadValidationTransforms(param)
    
    test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    
    #--------------------------------------------------------------------------------
    # Model
    #--------------------------------------------------------------------------------

    print("Loading the model...")
    
    (model_unet, post_pred, post_label) = setupModel(param)

    device = torch.device(param.test_device_name)    
    model = model_unet.to(device)
    
    model.load_state_dict(torch.load(os.path.join(param.root_dir, param.model_file), map_location=device))


    #--------------------------------------------------------------------------------
    # Test
    #--------------------------------------------------------------------------------
    
    # Loss function & optimizer
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    model.eval()

    print("Start evaluating...")
    with torch.no_grad():
        
        i = 0
        for test_data in test_loader:
            print("Processing image #" + str(i))
            i = i + 1

            test_inputs, test_labels = (
                test_data["image"].to(device),
                test_data["label"].to(device),
            )
            roi_size = param.window_size
            sw_batch_size = 4
            test_outputs = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model)
            test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]
            dice_metric(y_pred=test_outputs, y=test_labels)
            
            
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        print("Mean Dice : " + str(metric))
        
        # reset the status for next validation round
        dice_metric.reset()
    

def main(argv):
    
  try:
    parser = argparse.ArgumentParser(description="Apply a saved DL model for segmentation.")
    parser.add_argument('cfg', metavar='CONFIG_FILE', type=str, nargs=1,
                        help='Configuration file')
    #parser.add_argument('input', metavar='INPUT_PATH', type=str, nargs=1,
    #help='A file or a folder that contains images.')
            
    args = parser.parse_args(argv)

    config_file = args.cfg[0]
    #input_path = args.input[0]

    print('Loading parameters from: ' + config_file)
    param = TestParam(config_file)

    test_files = generateLabeledFileList(param.data_dir, 'test')
    
    n_test = len(test_files)
    print('Test data size: ' + str(n_test))

    run(param, test_files)
    

  except Exception as e:
    print(e)
  sys.exit()


if __name__ == "__main__":
  main(sys.argv[1:])


