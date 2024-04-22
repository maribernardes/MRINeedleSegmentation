import SimpleITK as sitk
import torch
from monai.data import MetaTensor
import numpy as np

# Receives an SimpleITK image and converts is to metatensor format:
# Example usage
# sitk_image = sitk.Image(64, 64, 64, sitk.sitkFloat32)  # Example SimpleITK image
# metatensor = SitkToMetatensor(sitk_image)
# print(metatensor)
def SitkToMetatensor(sitk_image):
    # Convert SimpleITK image to numpy array
    data = sitk.GetArrayFromImage(sitk_image)
    
    # Depending on the dimensionality of the image, handle it appropriately
    if len(data.shape) == 3:  # Assuming the shape is either (Z, Y, X) or (Y, X, C)
        if data.shape[-1] == 3 and sitk_image.GetNumberOfComponentsPerPixel() == 3:  # Typical for RGB or other 3-channel data
            # Assuming last dimension is channels and should not be moved
            data = np.moveaxis(data, 0, -1)  # Only move the first axis to the last position
        else:
            # Reorder data array from (Z, Y, X) to (X, Y, Z) for MONAI
            data = np.moveaxis(data, [0, 1, 2], [2, 1, 0])
    else:
        raise ValueError("Unsupported image dimensionality for conversion.")

    # Extract spacing, direction, and origin from SimpleITK Image
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    origin = sitk_image.GetOrigin()

    # Adjust spacing order to match the data axes order for MONAI
    spacing = np.array([spacing[2], spacing[1], spacing[0]])

    # Create affine matrix from direction, adjusting for RAS to LPS conversion
    direction_matrix = np.array(direction).reshape((3, 3)).T  # Transpose to align with MONAI's handling
    direction_matrix[:, 0] *= -1  # Flip X axis for LPS to RAS
    direction_matrix[:, 2] *= -1  # Flip Z axis for LPS to RAS

    # Adjust origin for RAS to LPS conversion
    origin = np.array(origin)
    origin[0] *= -1  # Flip X axis
    origin[1] *= -1  # Flip Y axis

    # Construct affine matrix
    affine = np.eye(4)
    affine[:3, :3] = direction_matrix * spacing[np.newaxis, :]
    affine[:3, 3] = origin
    
    # Create MetaTensor
    meta = {'affine': torch.tensor(affine, dtype=torch.float32)}
    metatensor = MetaTensor(torch.tensor(data, dtype=torch.float32), meta=meta)
    return metatensor

# Receives a metatensor and convert it to sitk image
# Example usage
# tensor = torch.rand((3, 64, 64, 64))  # Example multi-channel 3D volume
# meta = {'affine': torch.eye(4)}  # Example affine matrix
# metatensor = MetaTensor(tensor, meta=meta)
# sitk_image = MetatensorToSitk(metatensor, channel_index=1)  # Output the second channel
# print(sitk_image)
def MetatensorToSitk(metatensor, channel_index=0):
    if not isinstance(metatensor, MetaTensor):
        raise ValueError("Input must be a MetaTensor.")

    # Convert tensor to numpy and ensure it's in CPU memory
    data = metatensor.cpu().numpy()

    # Handle channels
    if data.ndim == 4:  # Assuming shape is (C, Z, Y, X) for 3D data
        if channel_index >= data.shape[0]:
            raise ValueError("Channel index out of range.")
        data = data[channel_index]  # Select the specified channel
    elif data.ndim not in [3, 2]:
        raise ValueError("Unsupported tensor shape for conversion.")

    # If 3D, reorder from (Z, Y, X) to (X, Y, Z)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))  # Now it's in (X, Y, Z) as SimpleITK expects for 3D images
    elif data.ndim == 2:
        # No need to reorder for 2D images as SimpleITK and numpy use the same ordering (Y, X)
        pass

    # Create SimpleITK image from numpy array
    sitk_image = sitk.GetImageFromArray(data)

    # Extract affine matrix and compute spacing, direction, and origin
    affine = metatensor.meta.get('affine', torch.eye(4)).numpy()
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = (affine[:3, :3] / spacing[:, np.newaxis]).flatten()
    origin = affine[:3, 3]

    # Adjust for RAS to LPS conversion
    direction = np.array(direction)
    direction[::3] *= -1  # Flip X components
    direction[1::3] *= -1  # Flip Y components

    origin[0] *= -1  # Flip X
    origin[1] *= -1  # Flip Y

    # Set properties on the SimpleITK image
    sitk_image.SetSpacing(spacing.tolist())
    sitk_image.SetDirection(direction.tolist())
    sitk_image.SetOrigin(origin.tolist())

    return sitk_image