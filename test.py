#! /usr/bin/python

from monai.utils import first, set_determinism
# from monai.metrics import compute_meandice
# from monai.metrics import DiceMetric
# from monai.metrics import HausdorffDistanceMetric
# from monai.metrics import SurfaceDistanceMetric

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
from monai.transforms import SaveImage, Activationsd, Invertd, AsDiscreted
from sitkIO import PushSitkImage

def calculate_tip(sitk_output):
  # Separate labels
    sitk_tip = (sitk_output==2)
    sitk_shaft = (sitk_output==1)
    
    center = None
    shaft_tip = None
    
    # Select tip from segmentation
    if sitk.GetArrayFromImage(sitk_tip).sum() > 0:
      # Get labels from segmentation
      stats = sitk.LabelShapeStatisticsImageFilter()
      # TODO: With fewer false positives, check if really necessary to compute the radius to check shaft pairing
      # stats.SetComputeFeretDiameter(True)
      stats.Execute(sitk.ConnectedComponent(sitk_tip))
      # Get labels sizes and centroid physical coordinates
      labels_size = []
      labels_centroid = []
      # labels_max_radius = []
      for l in stats.GetLabels():
        number_pixels = stats.GetNumberOfPixels(l)
        centroid = stats.GetCentroid(l)
        # max_radius = stats.GetFeretDiameter(l)
        print('Tip Label %s: -> Size: %s, Center: %s' %(l, number_pixels, centroid))
        labels_size.append(number_pixels)
        labels_centroid.append(centroid)    
        # labels_max_radius.append(max_radius)
      # Get tip estimate position
      index_largest = labels_size.index(max(labels_size)) # Find index of largest centroid
      print('Selected tip = %s' %str(index_largest+1))
      center = labels_centroid[index_largest]             # Get the largest centroid center
      return center

def run(param, output_path, test_files, model_file):
    
    #--------------------------------------------------------------------------------
    # Test datasets
    #--------------------------------------------------------------------------------

    val_transforms = loadValidationTransforms(param)
    test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)
    
    post_transforms = Compose([
      Invertd(
        keys="pred",
        transform=val_transforms,
        orig_keys="image", 
        meta_keys="pred_meta_dict", 
        orig_meta_keys="image_meta_dict",  
        meta_key_postfix="meta_dict",  
        nearest_interp=False,
        to_tensor=True,
      ),
      Activationsd(keys=["pred"], sigmoid=True),
      AsDiscreted(keys=["pred"], argmax=True, num_classes=param.out_channels)])

    
    #--------------------------------------------------------------------------------
    # Model
    #--------------------------------------------------------------------------------

    print("Loading the model...")
    
    (model_unet, post_pred, post_label) = setupModel(param)

    device = torch.device(param.test_device_name)    
    model = model_unet.to(device)
    
    model.load_state_dict(torch.load(os.path.join(param.root_dir, model_file), map_location=device))
    
    # Save the test images
    if (output_path is not None):
        # Save images in output_path
        arr_pred = [post_transforms, SaveImage(output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False)]
        array_label = [post_label, SaveImage(output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False)]
    else:
        arr_pred = [post_transforms]
        arr_pred = [post_label]
        
    # # Convert to sitk for calculating metric
    # arr_pred.append(PushSitkImage(resample=False, output_dtype=np.uint16, print_log=False))
    # array_label.append(PushSitkImage(resample=False, output_dtype=np.uint16, print_log=False))

    post_pred = Compose(arr_pred)
    post_label = Compose(array_label)


    #--------------------------------------------------------------------------------
    # Test
    #--------------------------------------------------------------------------------
    
    model.eval()

    print("Start evaluating...")
    with torch.no_grad():
        k = 0
        for test_data in test_loader:
            print("Processing image #" + str(k))
            k = k + 1
            test_inputs, test_labels = (
                test_data['image'].to(device),
                test_data['label'].to(device),
            )
            roi_size = param.window_size
            sw_batch_size = 4
            test_data['pred'] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            test_outputs = [post_pred(i) for i in decollate_batch(test_data)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]
        
 
 ### FROM INFERENCE TEST
    #    # Get predictions
    #   val_inputs = val_data['image'].to(device)
    #   val_data['pred'] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
    #   val_data = [post_transforms(i) for i in decollate_batch(val_data)]
    #   val_preds = from_engine(['pred'])(val_data)      
    #   val_labels = from_engine(['label'])(val_data) 
      
    #   # Use image to extract tip data
    #   for j in range(len(val_preds)):
    #     sitk_pred = val_preds[j]
    #     sitk_label = val_labels[j]
    #     tip_pred = calculate_tip(sitk_pred)
    #     tip_real = calculate_tip(sitk_label)
    #     distance = np.linalg.norm(np.array(tip_pred) - np.array(tip_real))  # Calculate distance between tip and shaft
 
            
        #     dice_metric(y_pred=test_outputs, y=test_labels)
        #     hd_metric(y_pred=test_outputs, y=test_labels)
        #     sd_metric(y_pred=test_outputs, y=test_labels)
            
        # # aggregate the final mean dice result
        # metric = dice_metric.aggregate().item()
        # print("Mean Dice : " + str(metric))
        
        # dice_buf = dice_metric.get_buffer()
        # dice_m = torch.mean(dice_buf,0)
        # dice_sd = torch.std(dice_buf,0)

        # hd_buf = hd_metric.get_buffer()
        # hd_m = torch.mean(hd_buf,0)
        # hd_sd = torch.std(hd_buf,0)

        # sd_buf = sd_metric.get_buffer()
        # sd_m = torch.mean(sd_buf,0)
        # sd_sd = torch.std(sd_buf,0)
        
        # n = len(dice_m)
        
        # print ("===== Dice =====")
        # for i in range(n):
        #     print("Structure %d : %.3f +/- %.3f" %(i, dice_m[i].item(), dice_sd[i].item()))
            
        # print ("===== Hausdorff Distance  =====")
        # for i in range(n):
        #     print("Structure %d : %.3f +/- %.3f" %(i, hd_m[i].item(), hd_sd[i].item()))
        
        # print ("===== Surface Distance  =====")
        # for i in range(n):
        #     print("Structure %d : %.3f +/- %.3f" %(i, sd_m[i].item(), sd_sd[i].item()))
        
        # # reset the status for next validation round
        # dice_metric.reset()
    

def main(argv):
    
  try:
    parser = argparse.ArgumentParser(description="Apply a saved DL model for segmentation.")
    parser.add_argument('cfg', metavar='CONFIG_FILE', type=str, nargs=1,
                        help='Configuration file')
    parser.add_argument('output', metavar='OUTPUT_PATH', type=str, nargs=1,
                        help='A folder to store the output file(s).')
    parser.add_argument('-T', dest='tl_data', action='store_const',
                        const=True, default=False,
                        help='Test a result of transfer learning.')

    #parser.add_argument('input', metavar='INPUT_PATH', type=str, nargs=1,
    #help='A file or a folder that contains images.')
            
    args = parser.parse_args(argv)

    config_file = args.cfg[0]
    output_path = args.output[0]
    #input_path = args.input[0]

    print('Loading parameters from: ' + config_file)
    param = TestParam(config_file)
    
    test_files = None
    model_file = None
    if args.tl_data == True:
        param_tl = TransferParam(config_file)
        test_files = generateLabeledFileList(param_tl, 'mytest')
        model_file = param_tl.tl_model_file
    else:
        test_files = generateLabeledFileList(param, 'mytest')
        model_file = param.model_file
    
    n_test = len(test_files)
    print('Test data size: ' + str(n_test))

    run(param, output_path, test_files, model_file)
    
  except Exception as e:
    print(e)
  sys.exit()

if __name__ == "__main__":
  main(sys.argv[1:])


