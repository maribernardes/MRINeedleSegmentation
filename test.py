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
import math
import json

from common import *
from monai.transforms import SaveImage, AsDiscrete, Invertd
from sitkIO import PushSitkImage

import SimpleITK as sitk 

def generate_json_list(data_dir):
    # Get dataset list
    dataset_list_filename = os.path.join(data_dir, 'imagelist_test_images.json')
    with open(dataset_list_filename) as file_images:
        list_images = json.load(file_images)
    cols = list_images['columns']
    # Obtain a numpy list of images formatted as col:
    dataset_image_list = np.array(list_images['images'], dtype=object)
    # image_tip = dataset_image_list[:,cols.index('tip')]
    # image_base = dataset_image_list[:,cols.index('base')]
    return (dataset_image_list, cols)

# Get coordinates stored at the json file
def get_physical_coordinates(name, dataset_image_list, cols):
    image_filename = dataset_image_list[:,cols.index('filename')]
    iminfo_needle_label = dataset_image_list[(image_filename==name),:]
    image_tip = iminfo_needle_label[0, cols.index('tip')]
    image_base = iminfo_needle_label[0, cols.index('base')]
    return (torch.tensor(image_tip), torch.tensor(image_base))

# Returns the unit vector of the vector
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# Returns the signed angle between two vectors
# Source: https://stackoverflow.com/a/70789545/19347752
# Based in: https://people.eecs.berkeley.edu/%7Ewkahan/MathH110/Cross.pdf (page 15)
def angle_between_vectors(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    y = v1_u - v2_u
    x = v1_u + v2_u
    a0 = 2 * np.arctan(np.linalg.norm(y) / np.linalg.norm(x))
    if (not np.signbit(a0)) or np.signbit(np.pi - a0):
        return np.rad2deg(a0)
    elif np.signbit(a0):
        return 0.0
    else:
        return np.rad2deg(np.pi)

# Calculate the 3D euclidean distance between two points with coordinates in tensors = torch.tensor(L, P, S)
# Where L = right-left, P = anterior-posterior and S = inferior-superior
# 2D uses only L and S coordinates (coronal images)
def euclidean_distance_2d(X, Y, dir='COR'):
    # Compute squared differences
    squared_diff = (X - Y) ** 2
    # Sum along dimension
    if dir=='AX':
        sum_squared_diff = squared_diff[0]+squared_diff[1]  # AX: L-P
    elif dir == 'SAG':
        sum_squared_diff = squared_diff[1]+squared_diff[2]  # SAG: P-S
    else: 
        sum_squared_diff = squared_diff[0]+squared_diff[2]  # COR: L-S
    # Take the square root
    distance = torch.sqrt(sum_squared_diff)
    return distance

# Calculate the euclidean distance betweentwo points with coordinates in tensors = torch.tensor(L, P, S)
# Where L = right-left, P = anterior-posterior and S = inferior-superior
# 3D uses all coordinates
def euclidean_distance_3d(X, Y):
    # Compute squared differences
    squared_diff = (X - Y) ** 2
    # Sum along dimension
    sum_squared_diff = squared_diff.sum(dim=0)
    # Take the square root
    distance = torch.sqrt(sum_squared_diff)
    return distance

def get_direction(sitk_output, label_value):
    stats = sitk.LabelShapeStatisticsImageFilter()    
    # Separate labels
    sitk_label = (sitk_output==int(label_value))
    sitk_shaft = sitk.ConnectedComponent(sitk_label)
    stats.Execute(sitk_shaft)
    # Select centroid from segmentation
    if sitk.GetArrayFromImage(sitk_label).sum() > 0:
        # Get labels from segmentation
        stats.SetComputeOrientedBoundingBox(True)
        stats.Execute(sitk.ConnectedComponent(sitk_shaft))
        # Get labels sizes and centroid physical coordinates
        labels = stats.GetLabels()
        labels_size = []
        labels_obb_dir = []
        labels_obb_size = []
        for l in stats.GetLabels():
            number_pixels = stats.GetNumberOfPixels(l)
            labels_size.append(number_pixels)
            labels_obb_dir.append(stats.GetOrientedBoundingBoxDirection(l))
            labels_obb_size.append(stats.GetOrientedBoundingBoxSize(l))       
        # Get the main insertion axis from the bounding box
        index_largest = labels_size.index(max(labels_size)) # Find index of largest centroid
        obb_size = labels_obb_size[index_largest]
        i_axis = obb_size.index(max(obb_size))
        obb_vec = unit_vector(labels_obb_dir[index_largest][3*i_axis:(3*i_axis+3)]) # Choose the vector of the longer axis
        obb_vec = obb_vec*math.copysign(1, obb_vec[2]) # Always choose dir that is positive in the direction of S
        return unit_vector(obb_vec)
    else:
        return None

# Get the centroid coordinates in RAS coordinates
def get_centroid(sitk_output, label_value):
    # Separate labels
    sitk_label = (sitk_output==int(label_value))
    # Select centroid from segmentation
    if sitk.GetArrayFromImage(sitk_label).sum() > 0:
        # Get labels from segmentation
        stats = sitk.LabelShapeStatisticsImageFilter()
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
        return torch.tensor([centerLPS[0], centerLPS[1], centerLPS[2]]) 
        ## Convert to 3D Slicer coordinates (RAS)
        # centerRAS = torch.tensor([-centerLPS[0], -centerLPS[1], centerLPS[2]])   
    else:
        return None


def run(param, output_path, test_files, model_file):
    
    #--------------------------------------------------------------------------------
    # Load json file with physical real positions
    #--------------------------------------------------------------------------------
    
    (image_list, cols_list) = generate_json_list(os.path.join(param.data_dir, 'test_labels'))
    label_prefix = param.data_dir.removeprefix('./')+'/test_labels/'

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
        err_2d_list = []
        err_3d_list = []
        err_ang_list = []
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
                # Get tip position from images
                tip_pred = get_centroid(sitk_pred[k], 2)
                tip_label = get_centroid(sitk_label[k], 2)
                # Get needle direction from images
                dir_pred = get_direction(sitk_pred[k], 1)
                dir_label= get_direction(sitk_label[k], 1)
                # Get tip position from stored physical coordinates 
                # filename = test_data['label_meta_dict']['filename_or_obj'][k]
                # name = filename.removeprefix(label_prefix)
                # (tip_real, base_real) = get_physical_coordinates(name, image_list, cols_list)
                if (tip_pred is not None) and (tip_label is not None):
                    err_3d = euclidean_distance_3d(tip_pred, tip_label)
                    err_2d = euclidean_distance_2d(tip_pred, tip_label)
                    err_ang = angle_between_vectors(dir_pred, dir_label)
                    err_3d_list.append(err_3d)
                    err_2d_list.append(err_2d)
                    err_ang_list.append(err_ang)
                    print('Image #%i: Err 3D = %f' %(N, err_3d))
                    print('Image #%i: Err 2D = %f' %(N, err_2d))
                elif (tip_pred is None):
                    false_negatives += 1
                    print('Image #%i: False negative' %(N))
                else:
                    false_positives += 1
                    print('Image #%i: False positive' %(N))
        
        # Calculate mean and variance
        distances_3d = torch.stack(err_3d_list)
        variance_distance_3d = distances_3d.var().item()
        mean_distance_3d = distances_3d.mean().item()
        
        distances_2d = torch.stack(err_2d_list)
        mean_distance_2d = distances_2d.mean().item()
        variance_distance_2d = distances_2d.var().item()
        
        angles = torch.stack(err_ang_list)
        mean_angles = angles.mean().item()
        variance_angles = angles.var().item()

        print ("===== FP/FN =====")
        print("False Neg = %i/%i" %(false_negatives, N))
        print("False Pos = %i/%i" %(false_positives, N))

        print ("===== Angle between needle directions =====")
        print("Mean Angle = %f" %(mean_angles))
        print("Var Angle = %f" %(variance_angles))

        print ("===== Mean Euclidean Distance (from label images) =====")
        print("Mean 3D Err = %f" %(mean_distance_3d))
        print("Var 3D Err = %f" %(variance_distance_3d))
        print("Mean 2D Err = %f" %(mean_distance_2d))
        print("Var 2D Err = %f" %(variance_distance_2d))
                

        
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


