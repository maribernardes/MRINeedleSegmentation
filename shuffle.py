#! /usr/bin/python

import shutil
import random
import os
import sys
import argparse

def shuffle_dataset(src_dir, dst_dir, training_split, dir_prefix):

  random.seed()
  
  cases = list(range(0,sum(training_split)))
  random.shuffle(cases)
  
  
  # Clean the destination folders
  for p in dir_prefix:
    dir_name = '%s/%s_images' % (dst_dir, p)
    shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    dir_name = '%s/%s_labels' % (dst_dir, p)
    shutil.rmtree(dir_name)
    os.mkdir(dir_name)
  
  
  training_split = [0] + training_split
    
  # Copy files
  for group in range(3): 
    print('Copying %s dataset...' % dir_prefix[group])
    dst_dir_image = '%s/%s_images' % (dst_dir, dir_prefix[group])
    dst_dir_label = '%s/%s_labels' % (dst_dir, dir_prefix[group])
  
    start_index = training_split[group]
    end_index = training_split[group]+training_split[group+1]
    for i in range(start_index, end_index):
      # images files
      filename1 = 'PhantomSynthetic_%s_M.nii.gz' % str(cases[i]).zfill(3) 
      filename2 = 'PhantomSynthetic_%s_P.nii.gz' % str(cases[i]).zfill(3) 
      src_path1 = '%s/%s' % (src_dir, filename1)
      src_path2 = '%s/%s' % (src_dir, filename2)
      shutil.copy(src_path1, dst_dir_image)
      shutil.copy(src_path2, dst_dir_image)
      
      # label file
      filename = 'PhantomSynthetic_%s_label.nii.gz' % str(cases[i]).zfill(3) 
      src_path = '%s/%s' % (src_dir, filename)
      shutil.copy(src_path, dst_dir_label)


def main(argv):

  #src_dir = 'data_pool'
  dir_prefix = ['train', 'val', 'test']
  training_split = [10, 6, 8]

  args = []
  try:
    parser = argparse.ArgumentParser(description="Randomly assign image/label files to training, validation, and test datasets.")
    parser.add_argument('src', metavar='SRC_DIR', type=str, nargs=1,
                        help='Data pool folder.')
    parser.add_argument('dst', metavar='DST_DIR', type=str, nargs=1,
                        help='Destination directory.')
    parser.add_argument('-t', dest='nTraining', default=training_split[0],
                        help='Number of training images (default: %d)' % training_split[0])
    parser.add_argument('-v', dest='nValidation', default=training_split[1],
                        help='Number of validation images (default: %d)' % training_split[1])
    parser.add_argument('-s', dest='nTesting', default=training_split[2],
                        help='Number of testing images (default: %d)' % training_split[2])
        
    args = parser.parse_args(argv)
        
  except Exception as e:
    print(e)
    sys.exit()

  src_dir = args.src[0]
  dst_dir = args.dst[0]
  training_split[0]  = int(args.nTraining)
  training_split[1]  = int(args.nValidation)
  training_split[2]  = int(args.nTesting)
  
  # Make the destination directory, if it does not exists.
  os.makedirs(dst_dir, exist_ok=True)
  shuffle_dataset(src_dir, dst_dir, training_split, dir_prefix)

  
if __name__ == "__main__":
  main(sys.argv[1:])
    
