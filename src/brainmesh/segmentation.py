"""Voxel-label operations for cleaning up and enforcing topology in segmentations."""
import numpy as np
import nbmorph
from numba import njit
from scipy import ndimage as ndi
from nbmorph import dilate_labels_spherical as dilate

from .labels import Label
from .decorators import track_voxel_changes, plot_voxel_changes, time_func


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def solidify_csf(data):
    mask = data > 0
    closed_mask = nbmorph.close_labels_spherical(mask, radius=5, iterations=1)
    seal = dilate(closed_mask) ^ closed_mask
    holes = ndi.binary_fill_holes(mask + seal) & ~(mask + dilate(seal, radius=1))
    data[holes] = Label.CSF
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def close_csf_space(data, radius=1, iter=1, brainstem_area_radius=0):
    closed_mask = nbmorph.close_labels_spherical(data > 0, radius=radius, iterations=iter)
    if brainstem_area_radius:
        brainstem_mask = dilate(data == Label.BRAIN_STEM, radius=brainstem_area_radius)
    else:
        brainstem_mask = True
    data[closed_mask & (data == 0) & brainstem_mask] = Label.CSF
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def fill_holes_csf(data):
    holes = ndi.binary_fill_holes(data > 0) != (data > 0)
    data[holes] = Label.CSF
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def fill_wm_hyperintensities(data):
    wm_data = data.copy()
    wm_data[~np.isin(data, [Label.LEFT_CEREBRAL_WHITE_MATTER, Label.RIGHT_CEREBRAL_WHITE_MATTER])] = 0
    data[data == Label.WM_HYPOINTENSITIES] = dilate(wm_data, radius=6)[data == Label.WM_HYPOINTENSITIES]
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def cut_bottom(data, offset=10):
    lowest_z = get_lowest_point(data > 0)[2]
    data[:, :, :lowest_z + offset] = 0
    return data


@njit
def separate_labels(data, l1, l2, dist, newlabel=Label.CSF, except_labels=None, except_region=None):
    m1 = nbmorph.dilate_labels_spherical(np.isin(data, l1), dist)
    m2 = nbmorph.dilate_labels_spherical(np.isin(data, l2), dist)
    if except_region is None:
        except_region = np.zeros(shape=data.shape, dtype=np.bool_)
    if except_labels is None:
        return np.where(m1 & m2 & ~except_region, newlabel, data)
    return np.where(m1 & m2 & ~except_region & ~np.isin(data, except_labels), newlabel, data)


@njit
def enforce_min_thickness(data, label, radius, s="B"):
    mask = data == label
    opened = nbmorph.erode_labels_spherical(mask, radius=radius, struct_sequence=s)
    opened = nbmorph.dilate_labels_spherical(opened, radius=radius, struct_sequence=s)
    diff = mask ^ opened
    dil = nbmorph.dilate_labels_spherical(diff, radius=radius, struct_sequence=s)
    data = np.where(dil, label, data)
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit
def carve_gruves(data, radius):
    return enforce_min_thickness(data, Label.CSF, radius=radius)


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit
def enforce_csf_layer(data, thickness=1):
    mask = (
        (data > 0)
        & (data != Label.CSF)
        & (data != Label.TENTORIUM)
        & (data != Label.FALX)
        & (data != Label.BRAIN_STEM)
    )
    dilated_mask = dilate(mask, radius=thickness, struct_sequence="B")
    mask += data == Label.BRAIN_STEM
    return np.where(dilated_mask > mask, Label.CSF, data)


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_csf_around_tentorium(data, radius=1):
    tent_mask = data == Label.TENTORIUM
    dil_tent_mask = dilate(tent_mask, radius=radius, struct_sequence="B")
    data[dil_tent_mask & ~tent_mask & (data > 0) & (data != Label.FALX)] = Label.CSF
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_csf_around_falx(data, radius=1):
    falx_mask = data == Label.FALX
    dil_falx_mask = dilate(falx_mask, radius=radius, struct_sequence="B")
    data[dil_falx_mask & ~falx_mask & (data > 0) & (data != Label.TENTORIUM)] = Label.CSF
    return data


def get_lowest_point(mask):
    idx = np.argwhere(mask)
    ind = np.argsort(idx[:, 2])
    return idx[ind[0]]

@time_func
def grow_into_region(seed_labels, region_mask, radius=1):
    grown_labels = np.copy(seed_labels)
    grown_labels[~region_mask] = 0
    n_voxels = (grown_labels > 0).sum()
    while True:
        grown_labels = dilate(grown_labels, radius=radius)
        grown_labels[~region_mask] = 0
        new_voxels = (grown_labels > 0).sum() - n_voxels
        if new_voxels == 0:
            break
        n_voxels += new_voxels
    return grown_labels