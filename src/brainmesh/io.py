import numpy as np
import pyvista as pv
import nibabel.processing as nibp

from .decorators import time_func


@time_func
def upsample_nib(img, factor=2, order=0):
    """
    Upsamples a nibabel image by a given factor while preserving the
    physical bounding box and affine orientation.

    Parameters:
        img: nibabel.Nifti1Image
        factor: upsampling factor (e.g. 2 means double resolution)
        order: interpolation order (0=nearest, 1=linear, 3=cubic)
    """
    target_shape = list(img.shape)
    for i in range(min(3, len(target_shape))):
        target_shape[i] = int(np.round(target_shape[i] * factor))

    target_affine = img.affine.copy()
    target_affine[:3, :3] /= factor

    offset_multiplier = (1.0 / factor - 1.0) / 2.0
    offset = img.affine[:3, :3] @ np.array([offset_multiplier, offset_multiplier, offset_multiplier])
    target_affine[:3, 3] += offset

    return nibp.resample_from_to(
        img,
        to_vox_map=(tuple(target_shape), target_affine),
        order=order,
    )


@time_func
def nibabel_to_pyvista(nib_img, scalar_name="data"):
    """
    Converts a nibabel image to a pyvista ImageData (UniformGrid).

    Parameters:
        nib_img: nibabel image object
        scalar_name: name for the scalar array in PyVista
    """
    data = nib_img.get_fdata()
    if data.ndim > 3:
        data = data[..., 0]

    affine = nib_img.affine
    spacing = np.array(nib_img.header.get_zooms()[:3])

    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1
    grid.spacing = spacing

    try:
        grid.direction_matrix = affine[:3, :3] / spacing
        grid.origin = affine[:3, 3] - (grid.direction_matrix @ spacing) / 2.0
    except AttributeError:
        grid.origin = affine[:3, 3] - spacing / 2.0

    grid.cell_data[scalar_name] = data.flatten(order='F')
    return grid
