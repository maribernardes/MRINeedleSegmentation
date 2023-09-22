import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(os.path.dirname(path),'TestingNotebook')
sys.path.append(utils_path)
from Utils import *

#########################################
## SCRIPT CONFIG                       ##
#########################################
offset_px = [0,0]

#########################################
## Images selection                    ##
#########################################
## You can use an image as reference for the desired slice direction
## Or you can use one of the standard directions bellow
image_list_file = 'imagelist_original_images.json'
output_dir = 'Final'

#########################################
## Desired output                      ##
#########################################
# Spacing
filename_ref_spacing = 'REF_BEAT_NEEDLE.nrrd'  # This image has the spacing we want
ref_spacing= sitk.ReadImage(os.path.join(path, 'Originals', filename_ref_spacing))
# show_image(ref_spacing, title='Reference image')
desired_spacing = ref_spacing.GetSpacing()  # Use spacing from reference image
# Direction
desired_direction = COR_DIR   # Direction vector (Canonic options: COR_DIR, AX_DIR or SAG_DIR)
# Dimensions
desired_size = (192,192,3)    # Volume size
# Position
desired_center = getPhysicalCenterItk(ref_spacing) 

##########################################################
## Main script

with open(os.path.join(path, image_list_file)) as file_images:
    list_images = json.load(file_images)

image_dataset = list_images['description']
dir = os.path.join(path, list_images['directory'])
cols = list_images['columns']
firstCase = list_images['firstCase']
totalCases = list_images['totalCases']
print(firstCase)
# Obtain a numpy list of images formatted as cols:
original_image_list = np.array(list_images['images'])
image_case = original_image_list[:,cols.index('case')]
image_type = original_image_list[:,cols.index('type')]

#########################################
## Load  images                        ##
#########################################
final_list = []
for caseNumber in range(firstCase, totalCases+firstCase):
    print('Case number = %i' %caseNumber)
    
    iminfo_image_m = original_image_list[(image_case==str(caseNumber)) & (image_type=='M'),:]
    iminfo_image_p = original_image_list[(image_case==str(caseNumber)) & (image_type=='P'),:]
    start_slice = int(iminfo_image_m[0, cols.index('startSlice')])
    end_slice = int(iminfo_image_m[0, cols.index('endSlice')])
    print('Original files = ' + iminfo_image_m[0, cols.index('filename')] + ' AND ' + iminfo_image_p[0, cols.index('filename')])

    base_image_m = read_image(iminfo_image_m, cols, dir)
    base_image_p = read_image(iminfo_image_p, cols, dir)

    image_set = (base_image_m, base_image_p)
    type_list = ('M', 'P')
    
    # For all images in the list:
    for i in range(len(image_set)):
        base_image = image_set[i]
        type = type_list[i]
        
        #########################################
        ## Change to reference direction       ##
        #########################################
        # ATTENTION: This is NOT a resampling. 
        # It is a transformation that will rotate and translate the volume in space
        rotated_image = sitk.Image(base_image)
        if (base_image.GetDirection() != desired_direction):    # Does nothing if already at the same orientation
            rotated_image.SetDirection(desired_direction)       # This rotates the image in space, it is not a resampling!
            
        #########################################
        ## Move to desired center              ##
        #########################################    
        # Center image at desired center
        spacing = rotated_image.GetSpacing()
        offset = [spacing[0]*offset_px[0], 0, -spacing[1]*offset_px[1]] # LPS in inverted
        setCenterItk(rotated_image, desired_center, offset)

        #########################################
        ## Resample to spatial resolution      ##
        #########################################
        # Assure same spatial resolution from reference
        resampled_image = adjustSpacingItk(rotated_image, spacing=desired_spacing)

        #########################################
        ## Crop volume to ref image size (2D)  ##
        #########################################
        image_size = resampled_image.GetSize()
        ref_size = (desired_size[0], desired_size[1], image_size[2])
        # Do some padding at the top and/or left if image is smaller than reference
        padding_left = ref_size[0]-image_size[0]
        padding_top = ref_size[1]-image_size[1]
        if type == 'P': # Phase image - pad with nois
            pixel_value = (-4096, 4092)
        else:        # Other images - pad with black
            pixel_value = 0
        if padding_top>0:   
            resampled_image = addPaddingItk(resampled_image, padding_top, side='top', padding_value=pixel_value)
        if padding_left>0:   
            resampled_image = addPaddingItk(resampled_image, padding_left, side='left', padding_value=pixel_value)
        # Do some croppint if image is bigger than reference
        image_size = resampled_image.GetSize()
        if (ref_size[0]< image_size[0]) or (ref_size[1]< image_size[1]):
            center_px = 0.5*(np.array(image_size)) + np.array([offset_px[0],offset_px[1],0])
            delta_px = 0.5*np.array(ref_size)
            crop_image = resampled_image[int(center_px[0]-delta_px[0]):int(center_px[0]+delta_px[0]), int(center_px[1]-delta_px[1]):int(center_px[1]+delta_px[1]),:]
        else:
            crop_image = resampled_image
        
        #########################################
        ## Select desired slices only          ##
        #########################################
        image_slices = crop_image.GetDepth()
        final_image = crop_image[:,:,start_slice:end_slice+1]
        setCenterItk(final_image, desired_center)
        filename_image = str(caseNumber) + '_TestImage_'+type+'.nrrd'
        sitk.WriteImage(final_image, os.path.join(path, output_dir, filename_image))
        final_list.append([caseNumber, type, filename_image])
                  
#########################################
## Save background files in json file  ##
#########################################
dataset_data = {}       # Initialize dataset info for json file
dataset_data['description'] = image_dataset
dataset_data['directory'] = output_dir
dataset_data['firstCase'] = firstCase
dataset_data['totalCases'] = totalCases
dataset_data['columns'] = ["case", "type", "filename"]
dataset_data['images'] = final_list
filename_json = 'imagelist_test_images.json'
with open(os.path.join(path, filename_json), 'w') as write_file:    
    json.dump(dataset_data, write_file)
    print('JSON file saved.')