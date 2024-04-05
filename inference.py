#! /usr/bin/python

from monai.utils import first, set_determinism
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
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
from monai.transforms import SaveImage, AsDiscrete, Invertd
from sitkIO import PushSitkImage

def run(param, output_path, val_files, model_file):
  device = torch.device(param.inference_device_name)
  
  print('Loading dataset...')
  (pre_transforms, post_transforms) = loadInferenceTransforms(param, output_path)
  val_ds = CacheDataset(data=val_files, transform=pre_transforms, cache_rate=1.0, num_workers=4)
  val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
  print('Outputs will be saved in ' + output_path)
  
  #--------------------------------------------------------------------------------
  # Model
  #--------------------------------------------------------------------------------
  print('Loading model...')
  (model_unet, post_pred, post_label) = setupModel(param)
  model = model_unet.to(device)
  model.load_state_dict(torch.load(os.path.join(param.root_dir, model_file), map_location=device))
  
  with torch.no_grad():
    for i,val_data in enumerate(val_loader):
      roi_size = param.window_size
      sw_batch_size = 4
      val_inputs = val_data['image'].to(device)
      val_data['pred'] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
      val_data = [post_transforms(i) for i in decollate_batch(val_data['pred'])]
    
  #--------------------------------------------------------------------------------
  # Evaluate
  #--------------------------------------------------------------------------------
  print('Evaluate model')
  model.eval()
      
def main(argv):
  try:
    parser = argparse.ArgumentParser(description="Apply a saved DL model for segmentation.")
    parser.add_argument('cfg', metavar='CONFIG_FILE', type=str, nargs=1,
                        help='Configuration file')
    parser.add_argument('input', metavar='INPUT_PATH', type=str, nargs=1,
                        help='A file or a folder that contains images.')
    parser.add_argument('output', metavar='OUTPUT_PATH', type=str, nargs=1,
                        help='A folder to store the output file(s).')
    parser.add_argument('-T', dest='tl_data', action='store_const',
                        const=True, default=False,
                        help='Use a result of transfer learning.')

    args = parser.parse_args(argv)

    config_file = args.cfg[0]
    input_path = args.input[0]
    output_path = args.output[0]

    # Make the destination directory, if it does not exists.
    os.makedirs(output_path, exist_ok=True)

    print('Loading parameters from: ' + config_file)
    param = InferenceParam(config_file)
    files = generateFileList(param, input_path)
    n_files = len(files)
    print('# of images: ' + str(n_files))
    
    model_file = None
    if args.tl_data:
      param_tl = TransferParam(config_file)
      model_file = param_tl.tl_model_file
    else:
      model_file = param.model_file

    run(param, output_path, files, model_file)
    
  except Exception as e:
    print(e)
    
  sys.exit()


if __name__ == "__main__":
  main(sys.argv[1:])
