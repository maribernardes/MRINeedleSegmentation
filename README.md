# IceBallSegmentation
AI-based Iceball segmentation for MRI-guided cryoablation

## Installation

Before running the scripts, make sure to install the following python libraries:
 - [MONAI](https://monai.io/)
 - [SimpleITK](https://simpleitk.readthedocs.io/en/v1.1.0/index.html) (for image conversion)
 - (to be added)

In the following instruction, we assume that the workspace is structured as follows:

~~~~
 + <working directory> 
     + IceBallSegmentation
     + config.ini
     + sorted_nii
         + train_images
             + Training image 1.nii.gz
             + Training image 2.nii.gz
                 ...
         + train_labels
             + Training label 1.nii.gz
             + Training label 2.nii.gz
                 ...
         + val_images
             + Validation image 1.nii.gz
             + Validation image 2.nii.gz
                 ...
         + val_labels
             + Validation label 1.nii.gz
             + Validation label 2.nii.gz
                 ...
~~~~

### Getting the code from GitHub

The script can be obtained from the GitHub repository: 

~~~~
$ cd <woking directory>
$ git clone https://github.com/tokjun/IceBallSegmentation
~~~~

### Prepare dataset

If your images are formatted in NRRD, they should be converted to Nii files, as MONAI's
image loader does not seem to handle NRRD's image header information (e.g., dimensions,
position, and orientation) correctly in the current version. The 'convert.py' script can
batch-process multiple images in a folder to convert from NRRD to Nii.

Before running the script, store the files as follows:

~~~~
 + <working directory> 
     + sorted
         + train_images
             + Training image 1.nii.gz
             + Training image 2.nii.gz
                 ...
         + train_labels
             + Training label 1.nii.gz
             + Training label 2.nii.gz
                 ...
         + val_images
             + Validation image 1.nii.gz
             + Validation image 2.nii.gz
                 ...
         + val_labels
             + Validation label 1.nii.gz
             + Validation label 2.nii.gz
                 ...
~~~~

Then, run convert.py. 
~~~~
$ cd <working directory>
$ IceBallSegmentation/convert.py
~~~~

If the script will output the images in the following directory structure:
~~~~
 + <working directory> 
     + sorted_nii
         + train_images
             + Training image 1.nii.gz
             + Training image 2.nii.gz
                 ...
         + train_labels
             + Training label 1.nii.gz
             + Training label 2.nii.gz
                 ...
         + val_images
             + Validation image 1.nii.gz
             + Validation image 2.nii.gz
                 ...
         + val_labels
             + Validation label 1.nii.gz
             + Validation label 2.nii.gz
                 ...
~~~~


### Prepare a configuration file

Example configuration file can be find in the directory cloned from the repository. Copy it to <working directory> and modify as needed.

~~~~
$ cp IceBallSegmentation/config.sample.ini config.ini
~~~~

### Training

To train the model, run the following script:
~~~~
$ IceBallSegmentation/training.py config.ini
~~~~

### Monitoring the training process using TensorBoard

If you have TensorBoard installed on the system, you can monitor the loss function from the web browser.
To activate it, edit the following line in the configuration file:

~~~~
[training]
use_tensorboard = 1
~~~~

Launch TensorBoard using as follows (make sure to change the current directory to where
the training script is running, as TensorBoard reads data from the file under 'runs/'):

~~~~
$ cd <working directory>
$ tensorboard --logdir=runs
~~~~

Then, open http://localhost:6006/ from a web browser.


### Validation

To apply the model to the validation image and store the results under 'output':
~~~~
$ IceBallSegmentation/inference.py config.ini sorted_nii output
~~~~






