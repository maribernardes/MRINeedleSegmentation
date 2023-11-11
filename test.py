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
from monai.transforms import SaveImage, AsDiscrete, Invertd
from sitkIO import PushSitkImage

import SimpleITK as sitk 

# Calculate the euclidean distance between two tensors = torch(D, H, W)
# Where D = depth, H = height and W = width
def euclidean_distance_3d(X, Y):
    # Compute squared differences
    squared_diff = (X - Y) ** 2
    # Sum along dimension
    sum_squared_diff = squared_diff.sum(dim=0)
    # Take the square root
    distance = torch.sqrt(sum_squared_diff)
    return distance

# Get the centroid coordinates in RAS coordinates
def get_centroid(sitk_output, label_value):
    # Separate labels
    sitk_label = (sitk_output==int(label_value))
    # Select centroid from segmentation
    if sitk.GetArrayFromImage(sitk_label).sum() > 0:
        # Get labels from segmentation
        stats = sitk.LabelShapeStatisticsImageFilter()
        # TODO: With fewer false positives, check if really necessary to compute the radius to check shaft pairing
        # stats.SetComputeFeretDiameter(True)
        stats.Execute(sitk.ConnectedComponent(sitk_label))
        # Get labels sizes and centroid physical coordinates
        labels_size = []
        labels_centroid = []
        for l in stats.GetLabels():
            number_pixels = stats.GetNumberOfPixels(l)
            centroid = stats.GetCentroid(l)
            labels_size.append(number_pixels)
            labels_centroid.append(centroid)    
        # Get tip estimate position
        index_largest = labels_size.index(max(labels_size)) # Find index of largest centroid
        # print('Selected tip = %s' %str(index_largest+1))
        # print('Tip: -> Size: %s, Center: %s' %(labels_size[index_largest] , labels_centroid[index_largest] ))
        centerLPS = labels_centroid[index_largest]             # Get the largest centroid center
        # Convert to 3D Slicer coordinates (RAS)
        centerRAS = torch.tensor([-centerLPS[0], -centerLPS[1], centerLPS[2]])    
    else:
        centerRAS = None
    return centerRAS

def run(param, output_path, test_files, model_file):
    
    #--------------------------------------------------------------------------------
    # Test datasets
    #--------------------------------------------------------------------------------

    batch_size = 2
    val_transforms = loadValidationTransforms(param)
    test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)
    
    
    #--------------------------------------------------------------------------------
    # Model
    #--------------------------------------------------------------------------------

    print("Loading the model...")
    
    (model_unet, post_pred, post_label) = setupModel(param)
    
    device = torch.device(param.test_device_name)    
    model = model_unet.to(device)
    
    model.load_state_dict(torch.load(os.path.join(param.root_dir, model_file), map_location=device))
    
    #--------------------------------------------------------------------------------
    # Additional transforms to get image format
    #--------------------------------------------------------------------------------

    # Save images in output_path
    save_transform = SaveImage(output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False, print_log=False)
    # Convert to sitk for calculating metric
    sitk_transform = PushSitkImage(resample=False, output_dtype=np.uint16, print_log=False)

    # Compose desired transforms
    post_pred_arr = [post_pred]
    post_label_arr = [post_label]
    if (output_path is not None):
        post_pred_arr.append(save_transform)
        post_label_arr.append(save_transform)
    post_pred_arr.append(sitk_transform)
    post_label_arr.append(sitk_transform)
    post_pred = Compose(post_pred_arr)
    post_label = Compose(post_label_arr)

    #--------------------------------------------------------------------------------
    # Test
    #--------------------------------------------------------------------------------
    
    model.eval()

    print("Start evaluating...")
    with torch.no_grad():
        err_list = []
        false_negatives = 0
        false_positives = 0
        # Batch processing
        N = 0
        for test_data in test_loader:
            test_input, test_label = (
                test_data['image'].to(device),
                test_data['label'].to(device),
            )
            roi_size = param.window_size
            sw_batch_size = 4
            test_pred = sliding_window_inference(test_input, roi_size, sw_batch_size, model)
            sitk_pred = [post_pred(i) for i in decollate_batch(test_pred)]
            sitk_label = [post_label(i) for i in decollate_batch(test_label)]
            # Use image to extract tip data
            for k in range(len(sitk_pred)):
                N += 1
                print()
                tip_pred = get_centroid(sitk_pred[k], 2)
                tip_real = get_centroid(sitk_label[k], 2)
                if (tip_pred is not None) and (tip_real is not None):
                    err = euclidean_distance_3d(tip_pred, tip_real)
                    err_list.append(err)
                    print('Image #%i: Err = %f' %(N, err))
                elif (tip_pred is None):
                    false_negatives += 1
                    print('Image #%i: False negative' %(N))
                else:
                    false_positives += 1
                    print('Image #%i: False positive' %(N))
        
        # Calculate mean and variance
        distances = torch.stack(err_list)
        mean_distance = distances.mean().item()
        variance_distance = distances.var().item()
        
        print ("===== Mean Euclidean Distance =====")
        print("Mean = %f" %(mean_distance))
        print("Var = %f" %(variance_distance))
        print("False Neg = %i/%i" %(false_negatives, N))
        print("False Pos = %i/%i" %(false_positives, N))
        
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
        test_files = generateLabeledFileList(param_tl, 'test')
        model_file = param_tl.tl_model_file
    else:
        test_files = generateLabeledFileList(param, 'test')
        model_file = param.model_file
    
    n_test = len(test_files)
    print('Test data size: ' + str(n_test))

    run(param, output_path, test_files, model_file)
    
  except Exception as e:
    print(e)
  sys.exit()

if __name__ == "__main__":
  main(sys.argv[1:])


