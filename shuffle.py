#! /usr/bin/python

import shutil
import random
import os
import sys
import argparse

import os
import shutil

def create_folder(folder_path, clean_files=False):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if clean_files:
            # Clean files inside the folder
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

def shuffle_dataset(src_dir, dst_dir, training_split, dir_prefix, file_prefix):

  # Randomly shuffle the cases
  random.seed()
  cases = list(range(1,sum(training_split)+1))
  random.shuffle(cases)
        
  # Clean the destination folders
  for p in dir_prefix:
    dst_dir_image = '%s/%s_images' % (dst_dir, p)
    create_folder(dst_dir_image, clean_files=True)
    dst_dir_label = '%s/%s_labels' % (dst_dir, p)
    create_folder(dst_dir_label, clean_files=True)
  
  # Initialize previous group final index as 0
  end_index = 0
  
  # Copy files for each group folder
  for group in range(3): 
    print('Copying %s dataset...' % dir_prefix[group])
    dst_dir_image = '%s/%s_images' % (dst_dir, dir_prefix[group])
    dst_dir_label = '%s/%s_labels' % (dst_dir, dir_prefix[group])

    # Get start/end indexes for current group
    start_index = end_index
    end_index = start_index + training_split[group]
    for i in range(start_index, end_index):
      print(i)
      # images files
      filename1 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_M.nii.gz'
      filename2 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_P.nii.gz'
      filename3 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_R.nii.gz'
      filename4 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_I.nii.gz'
      src_path1 = '%s/%s' % (src_dir, filename1)
      src_path2 = '%s/%s' % (src_dir, filename2)
      src_path3 = '%s/%s' % (src_dir, filename3)
      src_path4 = '%s/%s' % (src_dir, filename4)
      shutil.copy(src_path1, dst_dir_image)
      shutil.copy(src_path2, dst_dir_image)
      shutil.copy(src_path3, dst_dir_image)
      shutil.copy(src_path4, dst_dir_image)
      
      # label file
      filename1 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_shaft_label.nii.gz'
      filename2 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_tip_label.nii.gz'
      filename3 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_both_label.nii.gz'
      filename4 = file_prefix + '_'+ str(cases[i]).zfill(3) + '_multi_label.nii.gz'
      src_path1 = '%s/%s' % (src_dir, filename1)
      src_path2 = '%s/%s' % (src_dir, filename2)
      src_path3 = '%s/%s' % (src_dir, filename3)
      src_path4 = '%s/%s' % (src_dir, filename4)
      shutil.copy(src_path1, dst_dir_image)
      shutil.copy(src_path2, dst_dir_image)
      shutil.copy(src_path3, dst_dir_image)
      shutil.copy(src_path4, dst_dir_image)
      

def main(argv):

  dir_prefix = ['train', 'val', 'test']
  training_split = [10, 6, 8]

  args = []
  try:
    parser = argparse.ArgumentParser(description="Randomly assign image/label files to training, validation, and test datasets.")
    parser.add_argument('src', metavar='SRC_DIR', type=str, nargs=1,
                        help='Data pool folder.')
    parser.add_argument('dst', metavar='DST_DIR', type=str, nargs=1,
                        help='Destination directory.')
    parser.add_argument('-p', "--prefix", default='t2', 
                        help="Files common prefix")
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
  file_prefix = args.prefix
  
  # Make the destination directory, if it does not exists.
  os.makedirs(dst_dir, exist_ok=True)
  shuffle_dataset(src_dir, dst_dir, training_split, dir_prefix, file_prefix)

  
if __name__ == "__main__":
  main(sys.argv[1:])