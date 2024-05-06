import SimpleITK as sitk
import torch
import numpy as np
from monai.data import MetaTensor
from monai.data.utils import orientation_ras_lps, affine_to_spacing, to_affine_nd
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from monai.utils import TraceKeys, MetaKeys, SpaceKeys, GridSampleMode, GridSamplePadMode, convert_to_tensor, convert_data_type, get_equivalent_dtype
from monai.transforms.spatial.array import Resize, SpatialResample
    
# Receives an SimpleITK image and converts is to metatensor format:
# Example usage
# sitk_image = sitk.Image(64, 64, 64, sitk.sitkFloat32)  # Example SimpleITK image
# metatensor = SitkToMetatensor(sitk_image)
# print(metatensor)
def SitkToMetatensor(sitk_image):
    # Convert SimpleITK image to numpy array
    data = sitk.GetArrayFromImage(sitk_image)
    # Depending on the dimensionality of the image, handle it appropriately
    dimension = sitk_image.GetDimension()
    num_components = sitk_image.GetNumberOfComponentsPerPixel() 
    if (dimension == 3) and (num_components == 1):
        # Reorder data array from (Z, Y, X) to (X, Y, Z) for MONAI
        data = np.moveaxis(data, [0, 1, 2], [2, 1, 0])  
    elif (dimension == 2) and (num_components == 1):
        #This is a 3-channel 2D image (e.g., RGB)
        # TODO: Appropriate handling of (Y, X) sitk images
        raise ValueError('One-channel 2D image - Currently unsupported image dimensionality for conversion.')
    elif dimension == 2 and num_components == 3:
        #This is a 3-channel 2D image (e.g., RGB)
        # TODO: Appropriate handling of (Y, X, C) sitk images
        raise ValueError('Multi-channel 2D image - Currently unsupported image dimensionality for conversion.')
    else:
        raise ValueError('Unsupported image dimensionality for conversion.')
    
    # Extract spatial information (currently implemented for 3D images only)
    spacing = np.asarray(sitk_image.GetSpacing())
    origin = np.asarray(sitk_image.GetOrigin())
    dir_array = sitk_image.GetDirection()
    
    # Get affine
    direction = np.array([dir_array[0:3],dir_array[3:6],dir_array[6:9]])
    sr = min(max(direction.shape[0], 1), 3)
    affine: np.ndarray = np.eye(sr + 1)
    affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
    affine[:sr, -1] = origin[:sr]
    affine = orientation_ras_lps(affine)

    # Get spatial shape
    sr = np.array([dir_array[0:3],dir_array[3:6],dir_array[6:9]]).shape[0]
    sr = max(min(sr, 3), 1)
    img_size = list(sitk_image.GetSize())
    shape = np.asarray(img_size[:sr])

    # Get metadata dictionary  (3D image only)
    img_meta_dict = sitk_image.GetMetaDataKeys()
    meta_dict: dict = {}
    for key in img_meta_dict:
        if key.startswith("ITK_"):
            continue
        val = sitk_image.GetMetaData(key)
        meta_dict[key] = np.asarray(val) if type(val).__name__.startswith("itk") else val
    meta_dict['spacing'] = np.asarray(sitk_image.GetSpacing())
    
    # Construct metadata
    header = dict(meta_dict)
    header[MetaKeys.ORIGINAL_AFFINE] = affine
    header[MetaKeys.SPACE] = SpaceKeys.RAS
    header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
    header[MetaKeys.SPATIAL_SHAPE] = shape
    header[MetaKeys.ORIGINAL_CHANNEL_DIM] = float("nan") # Sitk 3D images have no channel
    
    # Create MetaTensor
    metatensor = MetaTensor(torch.tensor(data, dtype=torch.float32), meta=header)
    
    # Create metadata dict
    meta_dict = {}
    for key in header:
        datum = header[key]
        if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
            continue
        meta_dict[key] = str(TraceKeys.NONE) if datum is None else datum  # NoneType to string for default_collate
    return metatensor, meta_dict
    
# Receives a metatensor and convert it to sitk image
# Example usage
# tensor = torch.rand((3, 64, 64, 64))  # Example multi-channel 3D volume
# meta = {'affine': torch.eye(4)}  # Example affine matrix
# metatensor = MetaTensor(tensor, meta=meta)
# sitk_image = MetatensorToSitk(metatensor, channel_index=1)  # Output the second channel
# print(sitk_image)
def MetatensorToSitk(metatensor, channel_index=0, resample=True):
    if not isinstance(metatensor, MetaTensor):
        raise ValueError('Input must be a MetaTensor')
    
    # Convert tensor to numpy and ensure it's in CPU memory
    data = metatensor.cpu().numpy()
    
    # Handle channels and dimensions
    if data.ndim == 4:  # Assuming shape is (C, Z, Y, X) for 3D data
        if channel_index >= data.shape[0]:
            raise ValueError('Channel index out of range')
        data = data[channel_index]              # Select the specified channel
    elif data.ndim == 3: # Assuming shape is (C, Y, X) for 2D data
        # TODO: Implement for SimpleITK 2D images (single and multichannel)
        raise ValueError('2D image - Currently unsupported image dimensionality for conversion.')
    else:
        raise ValueError('Unsupported tensor shape for SimpleITK image conversion')

    original_affine = metatensor.meta.get('original_affine')
    affine = metatensor.meta.get(MetaKeys.AFFINE)
    spatial_shape = metatensor.meta.get(MetaKeys.SPATIAL_SHAPE)
    
    # Resample to original affine
    if resample is True:
        orig_type = type(data)
        data_array = convert_to_tensor(data, track_meta=True)
        if affine is not None:
            data_array.affine = convert_to_tensor(affine, track_meta=False)  # type: ignore
        resampler = SpatialResample(mode=GridSampleMode.BILINEAR, padding_mode=GridSamplePadMode.BORDER, align_corners=False, dtype=np.float32)
        output_array = resampler(data_array[None], dst_affine=original_affine, spatial_size=spatial_shape)
        # convert back at the end
        if isinstance(output_array, MetaTensor):
            output_array.applied_operations = []
        data_array, *_ = convert_data_type(output_array, output_type=orig_type)
        affine, *_ = convert_data_type(output_array.affine, output_type=orig_type)  # type: ignore
        data = data_array[0]        

    # # Create SimpleITK image from numpy array
    affine_lps_to_ras = (metatensor.meta.get(MetaKeys.SPACE, SpaceKeys.LPS) != SpaceKeys.LPS)  
    print(affine_lps_to_ras)
    data_array = convert_data_type(data_array, np.ndarray)[0]
    _is_vec = False
    if _is_vec:
        data_array = np.moveaxis(data_array, -1, 0)  # from channel last to channel first
    data_array = data_array.T.astype(get_equivalent_dtype(np.float32, np.ndarray), copy=True, order="C")
    sitk_image = sitk.GetImageFromArray(data_array, isVector=_is_vec)
    d = len(sitk_image.GetSize())
    if affine is None:
        affine = np.eye(d + 1, dtype=np.float64)
    _affine = convert_data_type(affine, np.ndarray)[0]
    if affine_lps_to_ras:
        _affine = orientation_ras_lps(to_affine_nd(d, _affine))
    spacing = affine_to_spacing(_affine, r=d)
    _direction: np.ndarray = np.diag(1 / spacing)
    _direction = _affine[:d, :d] @ _direction
    sitk_image.SetSpacing(spacing.tolist())
    sitk_image.SetOrigin(_affine[:d, -1].tolist())
    sitk_image.SetDirection(_direction.ravel().tolist())    
    
    
    
    
    
    # # data = np.transpose(data, (2, 1, 0))    # Now it's in (X, Y, Z) as SimpleITK expects for 3D images
    # # sitk_image = sitk.GetImageFromArray(data)
    # affine_lps_to_ras = (
    #             metatensor.meta.get(MetaKeys.SPACE, SpaceKeys.LPS) != SpaceKeys.LPS
    #         )  # do the converting from LPS to RAS only if the space type is currently LPS.

    # # Compute spacing, direction, and origin
    # data_array = data_array.T.astype(get_equivalent_dtype(np.float32, np.ndarray), copy=True, order="C")
    # sitk_image = sitk.GetImageFromArray(data_array, isVector=False)
    # d = len(sitk_image.GetSize())
    # if affine is None:
    #     affine = np.eye(d + 1, dtype=np.float64)
    # _affine = convert_data_type(affine, np.ndarray)[0]
    # if affine_lps_to_ras:
    #     _affine = orientation_ras_lps(to_affine_nd(d, _affine))
    # spacing = affine_to_spacing(_affine, r=d)
    # _direction: np.ndarray = np.diag(1 / spacing)
    # _direction = _affine[:d, :d] @ _direction
    # sitk_image.SetSpacing(spacing.tolist())
    # sitk_image.SetOrigin(_affine[:d, -1].tolist())
    # sitk_image.SetDirection(_direction.ravel().tolist())

    return sitk_image