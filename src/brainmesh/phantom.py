"""Synthetic simplified-brain segmentation for end-to-end testing."""
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

from .labels import Label


def make_phantom_seg(shape=(180, 180, 180), spacing=0.5):
    """
    Build a synthetic simplified-brain segmentation as a nibabel image.

    The phantom contains, per hemisphere, an outer cortex shell, a white-
    matter core, and a lateral ventricle. A midline third ventricle sits
    between them, a brainstem cylinder hangs below with a small fourth
    ventricle inside, and two cerebellar hemispheres sit posterior-
    inferior. A small white-matter hyperintensity pocket and a thin CSF
    shell around the whole brain are added for the pipeline to clean up.

    Coordinates follow the RAS convention used by the rest of the
    pipeline: +x = right, +y = anterior, +z = superior, with origin at
    the volume centre.

    Parameters
    ----------
    shape : tuple of int
        Volume shape in voxels (default 180^3 ≈ 90 mm cube at 0.5 mm).
    spacing : float
        Isotropic voxel spacing in mm.
    """
    Nx, Ny, Nz = shape
    data = np.zeros(shape, dtype=np.uint8)

    X = (np.arange(Nx) - Nx / 2 + 0.5) * spacing
    Y = (np.arange(Ny) - Ny / 2 + 0.5) * spacing
    Z = (np.arange(Nz) - Nz / 2 + 0.5) * spacing
    X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")

    def ellipsoid(cx, cy, cz, rx, ry, rz):
        return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2 < 1

    def cylinder_z(cx, cy, r, z_lo, z_hi):
        return ((X - cx) ** 2 + (Y - cy) ** 2 < r ** 2) & (Z >= z_lo) & (Z < z_hi)

    # Cerebral hemispheres: outer cortex shell, WM core, lateral ventricle
    for sign, lc, lw, lv in [
        (-1, Label.LEFT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_WHITE_MATTER, Label.LEFT_LATERAL_VENTRICLE),
        (+1, Label.RIGHT_CEREBRAL_CORTEX, Label.RIGHT_CEREBRAL_WHITE_MATTER, Label.RIGHT_LATERAL_VENTRICLE),
    ]:
        cx, cy, cz = sign * 18, 0, 8
        data[ellipsoid(cx, cy, cz, 22, 28, 22)] = lc
        data[ellipsoid(cx, cy, cz, 18, 24, 18)] = lw
        data[ellipsoid(sign * 8, 0, 8, 4, 12, 5) & (data == lw)] = lv

    # Third ventricle, midline
    data[ellipsoid(0, 0, 4, 2, 8, 4) & (data > 0)] = Label.THIRD_VENTRICLE

    # Brainstem (vertical cylinder, slightly posterior)
    data[cylinder_z(0, 4, 5, -32, -2) & (data == 0)] = Label.BRAIN_STEM

    # Fourth ventricle inside the brainstem
    data[ellipsoid(0, 4, -10, 1.5, 2, 4) & (data == Label.BRAIN_STEM)] = Label.FOURTH_VENTRICLE

    # Cerebellum (posterior-inferior, two hemispheres)
    for sign, cc, cw in [
        (-1, Label.LEFT_CEREBELLUM_CORTEX, Label.LEFT_CEREBELLUM_WHITE_MATTER),
        (+1, Label.RIGHT_CEREBELLUM_CORTEX, Label.RIGHT_CEREBELLUM_WHITE_MATTER),
    ]:
        cx, cy, cz = sign * 11, 22, -12
        data[ellipsoid(cx, cy, cz, 12, 12, 10) & (data == 0)] = cc
        data[ellipsoid(cx, cy, cz, 9, 9, 7) & (data == cc)] = cw

    # WM hyperintensity pocket — fill_wm_hyperintensities should consume this
    data[
        ellipsoid(-15, -8, 10, 1.5, 1.5, 1.5)
        & (data == Label.LEFT_CEREBRAL_WHITE_MATTER)
    ] = Label.WM_HYPOINTENSITIES

    # Thin CSF shell around the whole brain
    brain = data > 0
    shell = ndi.binary_dilation(brain, iterations=3) & ~brain
    data[shell] = Label.CSF

    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = -np.array(shape) / 2.0 * spacing
    return nib.Nifti1Image(data, affine)
