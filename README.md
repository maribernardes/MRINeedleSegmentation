# IceBallSegmentation
AI-based Iceball segmentation for MRI-guided cryoablation

## Prerequisite
 - Python 3.8 or later

## Installation

Before running the scripts, make sure to install the following python libraries:
 - [MONAI](https://monai.io/)
 - [SimpleITK](https://simpleitk.readthedocs.io/en/v1.1.0/index.html) (for image conversion)
 - tqdm (for showing a progress bar for loading images)
 - [NiBabel](https://nipy.org/nibabel/)
 - (to be added)

## Training

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
$ cd <working directory>
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
$ IceBallSegmentation/convert_dataset_to_nifty.sh
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

An example configuration file can be found in the directory cloned from the repository. Copy it to <working directory> and modify as needed.

~~~~
$ cp IceBallSegmentation/config.sample.ini config.ini
~~~~

### Training the model

To train the model, run the following script:
~~~~
$ IceBallSegmentation/training.py config.ini
~~~~

The result is stored in a *.pth file. The file name can be specified in config.ini using the 'model_file' parameter.


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



## Inference

The trained model (*.pth) can be used for the segmentation of unseen image data (images of ice balls not used for training). If you have not trained a model, an example model file ('best_metric_model.pth') is available in the repository.

First, copy the trained model to the working directory. Assuming that the model file is named 'best_metric_model.pth':

~~~
$ cd <working directory>
$ cp <model directory>/best_metric_model.pth
~~~

Next, copy unseen images under a folder named 'sample':

~~~
$ cp <images> sample/
~~~

Make sure to have the config.ini in the working directory (see the 'Training' section), and the 'model_file' parameter matches the name of the model file. Then run the following command:

~~~~
$ IceBallSegmentation/inference.py config.ini sample output
~~~~

The results are stored under the 'output' directory.







