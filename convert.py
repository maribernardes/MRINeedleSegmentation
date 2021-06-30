#! /usr/bin/python

import numpy
import glob
import argparse, sys, shutil, os, logging
import SimpleITK as sitk

def convert(sub_dir, input_dir, output_dir):
    
    image_files = glob.glob(os.path.join(input_dir, sub_dir, "*.nrrd"))
    for input_path in image_files:
        image   = sitk.ReadImage(input_path)
        intput_dir, output_file = os.path.split(input_path)
        output_file_name, output_file_ext =  os.path.splitext(output_file)
        output_file = output_file_name + '.nii.gz'
        
        output_path = os.path.join(output_dir, sub_dir, output_file)
        print(output_path)
        sitk.WriteImage(image, output_path)
    
    
def main(argv):

    input_dir = './sorted'
    output_dir = './sorted_nii'

    os.makedirs(output_dir+'/training_images', exist_ok=True)
    os.makedirs(output_dir+'/training_labels', exist_ok=True)
    convert('training_images', input_dir, output_dir)
    convert('training_labels', input_dir, output_dir)

    os.makedirs(output_dir+'/val_images', exist_ok=True)
    os.makedirs(output_dir+'/val_labels', exist_ok=True)
    convert('val_images', input_dir, output_dir)
    convert('val_labels', input_dir, output_dir)
                            
if __name__ == "__main__":
    main(sys.argv[1:])

