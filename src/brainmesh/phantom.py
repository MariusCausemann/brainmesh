"""Synthetic simplified-brain segmentation for end-to-end testing."""
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

from .anatomy import _connect_by_line
from .labels import Label


def make_phantom_seg(shape=(360, 360, 360), spacing=0.5, scale=2.0):
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
        Volume shape in voxels (default 360^3 ≈ 180 mm cube at 0.5 mm).
    spacing : float
        Isotropic voxel spacing in mm.
    scale : float
        Geometric scale factor for all anatomical structures (default 2.0,
        approximately full adult-brain size). Set to 1.0 for the half-size
        phantom, or other values to scale uniformly.
    """
    Nx, Ny, Nz = shape
    data = np.zeros(shape, dtype=np.uint8)

    X = (np.arange(Nx) - Nx / 2 + 0.5) * spacing
    Y = (np.arange(Ny) - Ny / 2 + 0.5) * spacing
    Z = (np.arange(Nz) - Nz / 2 + 0.5) * spacing
    X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")

    s = scale

    def ellipsoid(cx, cy, cz, rx, ry, rz):
        return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2 < 1

    def cylinder_z(cx, cy, r, z_lo, z_hi):
        return ((X - cx) ** 2 + (Y - cy) ** 2 < r ** 2) & (Z >= z_lo) & (Z < z_hi)

    # Cerebral hemispheres: outer cortex shell, WM core, lateral ventricle
    for sign, lc, lw, lv in [
        (-1, Label.LEFT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_WHITE_MATTER, Label.LEFT_LATERAL_VENTRICLE),
        (+1, Label.RIGHT_CEREBRAL_CORTEX, Label.RIGHT_CEREBRAL_WHITE_MATTER, Label.RIGHT_LATERAL_VENTRICLE),
    ]:
        cx, cy, cz = sign * 14 * s, 0, 8 * s
        data[ellipsoid(cx, cy, cz, 22 * s, 28 * s, 22 * s) & (sign * X > 0)] = lc
        data[ellipsoid(cx, cy, cz, 18 * s, 24 * s, 18 * s) & (sign * X > 0)] = lw
        data[ellipsoid(sign * 8 * s, 0, 8 * s, 4 * s, 12 * s, 5 * s) & (data == lw)] = lv

    # Ventral DC: subcortical structures flanking the midline.
    for sign, lvdc in [
        (-1, Label.LEFT_VENTRAL_DC),
        (+1, Label.RIGHT_VENTRAL_DC),
    ]:
        data[
            ellipsoid(sign * 4 * s, 0, -3 * s, 8 * s, 12 * s, 8 * s)
            & (sign * X > 0)
            & np.isin(data, [Label.LEFT_CEREBRAL_WHITE_MATTER,
                             Label.LEFT_CEREBRAL_CORTEX,
                             Label.RIGHT_CEREBRAL_WHITE_MATTER,
                             Label.RIGHT_CEREBRAL_CORTEX,
                             Label.BRAIN_STEM])
        ] = lvdc

    # Third ventricle, midline
    data[ellipsoid(0, 0, 4 * s, 2 * s, 8 * s, 4 * s) & (data > 0)] = Label.THIRD_VENTRICLE

    # Brainstem (vertical cylinder, slightly posterior)
    data[cylinder_z(0, 4 * s, 5 * s, -30 * s, -2 * s) & (data == 0)] = Label.BRAIN_STEM

    # Fourth ventricle inside the brainstem, with median aperture connection
    data[
        ellipsoid(0, 4 * s, -10 * s, 1.5 * s, 2 * s, 4 * s)
    ] = Label.FOURTH_VENTRICLE
    median_ap_outlet = ellipsoid(0, 10 * s, -20 * s, 1 * s, 1 * s, 1 * s)
    median_aperture = _connect_by_line(
        data == Label.FOURTH_VENTRICLE,
        median_ap_outlet,
        radius=max(1, int(round(s))),
    )
    data[median_aperture] = Label.FOURTH_VENTRICLE

    # Cerebellum (posterior-inferior, two hemispheres)
    for sign, cc, cw in [
        (-1, Label.LEFT_CEREBELLUM_CORTEX, Label.LEFT_CEREBELLUM_WHITE_MATTER),
        (+1, Label.RIGHT_CEREBELLUM_CORTEX, Label.RIGHT_CEREBELLUM_WHITE_MATTER),
    ]:
        cx, cy, cz = sign * 11 * s, 12 * s, -12 * s
        data[ellipsoid(cx, cy, cz, 12 * s, 12 * s, 10 * s) & (data == 0)] = cc
        data[ellipsoid(cx, cy, cz, 9 * s, 9 * s, 7 * s) & (data == cc)] = cw

    # WM hyperintensity pocket — fill_wm_hyperintensities should consume this
    data[
        ellipsoid(-15 * s, -8 * s, 10 * s, 1.5 * s, 1.5 * s, 1.5 * s)
        & (data == Label.LEFT_CEREBRAL_WHITE_MATTER)
    ] = Label.WM_HYPOINTENSITIES

    # Thin CSF shell around the whole brain
    brain = data > 0
    shell = ndi.binary_dilation(brain, iterations=3) & ~brain
    data[shell] = Label.CSF

    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = -np.array(shape) / 2.0 * spacing
    return nib.Nifti1Image(data, affine)
