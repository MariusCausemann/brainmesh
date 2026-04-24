"""Voxel-label operations for cleaning up and enforcing topology in segmentations."""
import numpy as np
import nbmorph
from numba import njit
from scipy import ndimage as ndi
from nbmorph import dilate_labels_spherical as dilate
from nbmorph import erode_labels_spherical as erode

from .labels import Label, VENTRICLE_LABELS
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
def diamond_mode_filter(data):
    """
    Applies a 3D diamond (von Neumann) mode filter.
    The footprint is 7 voxels: the center and its 6 face-sharing neighbors.
    """
    out = data.copy()
    nx, ny, nz = data.shape
    offsets = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                neighbors = np.zeros(7, dtype=data.dtype)
                for idx in range(7):
                    dx, dy, dz = offsets[idx]
                    neighbors[idx] = data[i + dx, j + dy, k + dz]

                best_label = neighbors[0]
                max_count = 0
                for idx in range(7):
                    candidate_label = neighbors[idx]
                    count = 0
                    for jdx in range(7):
                        if neighbors[jdx] == candidate_label:
                            count += 1
                    if count > max_count:
                        max_count = count
                        best_label = candidate_label
                out[i, j, k] = best_label
    return out


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
