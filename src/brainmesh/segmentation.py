"""Voxel-label operations for cleaning up and enforcing topology in segmentations."""
import numpy as np
import nbmorph
from numba import njit
from nbmorph import dilate_labels_spherical as dilate

from imagemesh.morphology import (
    binary_fill_holes,
    enforce_min_thickness,
    fill_from_neighbors,
    get_lowest_point,
    set_mask_scalar,
)

from .labels import Label
from .decorators import track_voxel_changes, plot_voxel_changes, time_func


#@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
#@njit(parallel=True, cache=True)
def solidify_csf(data, mask_closing_radius=5, mask_closing_iterations=1, mark_unclassified=False):
    from cc3d import dust
    mask = data > 0
    closed_mask = nbmorph.close_labels_spherical(
        mask, radius=mask_closing_radius, iterations=mask_closing_iterations
    )
    seal = dilate(closed_mask) ^ closed_mask
    holes = binary_fill_holes(mask + seal) & ~(mask + dilate(seal, radius=1))
    set_mask_scalar(data, holes, Label.CSF)
    if mark_unclassified:
        large_holes = dust(holes, threshold=100, connectivity=6)
        set_mask_scalar(data, large_holes, Label.UNCLASSIFIED)
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit(parallel=True, cache=True)
def close_csf_space(data, radius=1, iter=1, brainstem_area_radius=0, mark_unclassified=False):
    closed_mask = nbmorph.close_labels_spherical(data > 0, radius=radius, iterations=iter)
    if brainstem_area_radius:
        brainstem_mask = dilate(data == Label.BRAIN_STEM, radius=brainstem_area_radius)
    else:
        brainstem_mask = np.ones(data.shape, dtype=np.bool_)
    if mark_unclassified:
        set_mask_scalar(data, closed_mask & (data == 0) & brainstem_mask, Label.UNCLASSIFIED)
    else:
        set_mask_scalar(data, closed_mask & (data == 0) & brainstem_mask, Label.CSF)
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def fill_holes_csf(data):
    holes = binary_fill_holes(data > 0) != (data > 0)
    data[holes] = Label.CSF
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def fill_small_unclassified_fragments(data, size):
    from cc3d import dust
    uncl_mask = data==Label.UNCLASSIFIED
    large_unclassified = dust(uncl_mask, threshold=size, connectivity=6)
    set_mask_scalar(data, uncl_mask & ~large_unclassified, Label.CSF)
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def fill_wm_hyperintensities(data):
    return fill_from_neighbors(data, data==Label.WM_HYPOINTENSITIES, 
                        [Label.LEFT_CEREBRAL_WHITE_MATTER,
                         Label.RIGHT_CEREBRAL_WHITE_MATTER])


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def cut_bottom(data, offset=10):
    lowest_z = get_lowest_point(data > 0)[2]
    data[:, :, :lowest_z + offset] = 0
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit(cache=True, parallel=True)
def carve_gruves(data, radius):
    return enforce_min_thickness(data, Label.CSF, radius=radius)


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit(cache=True, parallel=True)
def enforce_csf_layer(data, thickness=1):
    mask = (
        (data > 0)
        & (data != Label.CSF)
        & (data != Label.TENTORIUM)
        & (data != Label.FALX)
        & (data != Label.BRAIN_STEM)
        & (data != Label.UNCLASSIFIED)
        & (data != Label.VESSEL)
    )
    dilated_mask = dilate(mask, radius=thickness, struct_sequence="B")
    mask += data == Label.BRAIN_STEM
    mask += data == Label.UNCLASSIFIED
    mask += data == Label.VESSEL
    mask += data == Label.TENTORIUM
    mask += data == Label.FALX
    return set_mask_scalar(data, dilated_mask > mask, Label.CSF)


def count_background_contacts(data):
    """Background voxels 26-adjacent to parenchyma -- must be 0 after enforce_csf_layer.

    ``contour_labels`` meshes voxel corners, so a single corner contact between the
    background and the parenchyma opens a spurious PAR/background patch. This is why
    ``enforce_csf_layer`` dilates with a box ("B") element: one iteration covers the
    full 26-neighbourhood, which is the thinnest layer that closes corner contacts.

    The spinal buffer is excluded: ``extend_brainstem_caudally`` runs after
    ``enforce_csf_layer`` and deliberately leaves it open at the caudal cut plane.
    """
    par = (
        (data > 0)
        & (data != Label.CSF)
        & (data != Label.TENTORIUM)
        & (data != Label.FALX)
        & (data != Label.BRAIN_STEM)
        & (data != Label.UNCLASSIFIED)
        & (data != Label.SPINAL_BUFFER)
        & (data != Label.VESSEL)
    )
    return (dilate(par, radius=1, struct_sequence="B") & (data == 0)).sum()


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit(cache=True, parallel=True)
def enforce_csf_around_tentorium(data, radius=1):
    tent_mask = data == Label.TENTORIUM
    dil_tent_mask = dilate(tent_mask, radius=radius, struct_sequence="B")
    set_mask_scalar(data, dil_tent_mask & (data==Label.UNCLASSIFIED), Label.TENTORIUM)
    set_mask_scalar(data, dil_tent_mask & ~tent_mask & (data > 0) &
                          (data != Label.FALX), Label.CSF)
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit(cache=True, parallel=True)
def enforce_csf_around_falx(data, radius=1):
    falx_mask = data == Label.FALX
    dil_falx_mask = dilate(falx_mask, radius=radius, struct_sequence="B")
    set_mask_scalar(data, dil_falx_mask & ~falx_mask & (data > 0) & (data != Label.TENTORIUM), Label.CSF)
    return data
