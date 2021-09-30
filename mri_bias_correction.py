#! /usr/bin/python

import numpy
import glob
import argparse, sys, shutil, os, logging
import SimpleITK as sitk

def convert(input_dir, output_dir, param):
    
    image_files = glob.glob(os.path.join(input_dir, "*.nrrd"))
    for input_path in image_files:
        print('Processing ' + input_path + '...')
        image   = sitk.ReadImage(input_path)
        cimage = sitk.Cast(image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetNumberOfControlPoints(param['numberOfControlPoints'])
        corrector.SetMaximumNumberOfIterations(param['numberOfIterations'])
        corrector.SetSplineOrder(param['bsplineOrder'])
        corrector.SetConvergenceThreshold(param['convergenceThreshold'])
        corrected_image = corrector.Execute(cimage)
                                           
        intput_dir, output_file = os.path.split(input_path)
        output_file_name, output_file_ext =  os.path.splitext(output_file)
        output_file = output_file_name + '.nii.gz'
        
        output_path = os.path.join(output_dir, output_file)
        print(output_path)
        sitk.WriteImage(corrected_image, output_path)

        
def strToFloat(s):
    
    return float(s)

def strToInt(s):
    
    return int(s)

def strToIntArray(s):
    strArray = s.split(',')
    intArray = [int(d) for d in strArray]
    return intArray
    
def main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Apply N4 bias field correction.")
        parser.add_argument('src', metavar='SRC_DIR', type=str, nargs=1,
                            help='Source directory.')
        parser.add_argument('dst', metavar='DST_DIR', type=str, nargs=1,
                            help='Destination directory.')
        
        parser.add_argument('-c', dest='numberOfControlPoints', default='4,4,4',
                            help='Number of control points (default: 4,4,4)')
        parser.add_argument('-t', dest='convergenceThreshold', default='0.0001',
                            help='Convergence threshold (default: 0.0001)')
        parser.add_argument('-b', dest='bSplineOrder', default='3',
                            help='B-Spline order (default: 3)')
        parser.add_argument('-i', dest='numberOfIterations', default='50,40,30',
                            help='Number of iteration for each step (default: 50,40,30)')
        
        args = parser.parse_args(argv)
        
    except Exception as e:
        print(e)
        sys.exit()

    input_dir = args.src[0]
    output_dir = args.dst[0]
    
    numberOfControlPoints = strToIntArray(args.numberOfControlPoints)
    convergenceThreshold = strToFloat(args.convergenceThreshold)
    bSplineOrder = strToInt(args.bSplineOrder)
    numberOfIterations = strToIntArray(args.numberOfIterations)
    
    param = {
        'numberOfControlPoints': numberOfControlPoints,
        'convergenceThreshold' : convergenceThreshold,
        'bsplineOrder'         : bSplineOrder,
        'numberOfIterations'   : numberOfIterations
    }

    # Make the destination directory, if it does not exists.
    os.makedirs(output_dir, exist_ok=True)

    convert(input_dir, output_dir, param)

if __name__ == "__main__":
    main(sys.argv[1:])

