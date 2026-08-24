"""Voxel-label operations for cleaning up and enforcing topology in segmentations."""
import numpy as np
import nbmorph
from numba import njit, prange
from nbmorph import dilate_labels_spherical as dilate

from .labels import Label
from .decorators import track_voxel_changes, plot_voxel_changes, time_func

@njit(parallel=True, cache=True)
def set_mask_scalar(arr, mask, value):
    """Sets a scalar value wherever the 3D mask is True (Multi-threaded)."""
    # prange distributes the 2D slices (the 'i' dimension) across CPU cores
    for i in prange(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if mask[i, j, k]:
                    arr[i, j, k] = value
    return arr

@njit(parallel=True, cache=True)
def copy_mask(dest, mask, src):
    """Copies values from src to dest wherever the 3D mask is True (Multi-threaded)."""
    for i in prange(dest.shape[0]):
        for j in range(dest.shape[1]):
            for k in range(dest.shape[2]):
                if mask[i, j, k]:
                    dest[i, j, k] = src[i, j, k]
    return dest


@njit(cache=True)
def binary_fill_holes(img):
    """
    Fills holes in a 3D binary volume using a stack-based flood fill with 6-connectivity.
    """
    depth, rows, cols = img.shape
    external_bg = np.zeros_like(img, dtype=np.bool_)
    
    # Cache optimization: Single (N, 3) matrix using int16
    # Halves memory usage and guarantees contiguous CPU cache hits
    max_pixels = depth * rows * cols
    stack = np.empty((max_pixels, 3), dtype=np.int16)
    top = 0
    
    # --- Initialization ---
    for r in range(rows):
        for c in range(cols):
            if not img[0, r, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = 0, r, c
                external_bg[0, r, c] = True
                top += 1
            if depth > 1 and not img[depth - 1, r, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = depth - 1, r, c
                external_bg[depth - 1, r, c] = True
                top += 1

    for d in range(1, depth - 1):
        for c in range(cols):
            if not img[d, 0, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, 0, c
                external_bg[d, 0, c] = True
                top += 1
            if rows > 1 and not img[d, rows - 1, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, rows - 1, c
                external_bg[d, rows - 1, c] = True
                top += 1
                
    for d in range(1, depth - 1):
        for r in range(1, rows - 1):
            if not img[d, r, 0]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, r, 0
                external_bg[d, r, 0] = True
                top += 1
            if cols > 1 and not img[d, r, cols - 1]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, r, cols - 1
                external_bg[d, r, cols - 1] = True
                top += 1

    # --- Flood Fill (Unrolled for pure speed) ---
    while top > 0:
        top -= 1
        d = stack[top, 0]
        r = stack[top, 1]
        c = stack[top, 2]
        
        # Unrolled 6-connectivity checks
        # d - 1
        if d > 0 and not img[d - 1, r, c] and not external_bg[d - 1, r, c]:
            external_bg[d - 1, r, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d - 1, r, c
            top += 1
        # d + 1
        if d < depth - 1 and not img[d + 1, r, c] and not external_bg[d + 1, r, c]:
            external_bg[d + 1, r, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d + 1, r, c
            top += 1
        # r - 1
        if r > 0 and not img[d, r - 1, c] and not external_bg[d, r - 1, c]:
            external_bg[d, r - 1, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r - 1, c
            top += 1
        # r + 1
        if r < rows - 1 and not img[d, r + 1, c] and not external_bg[d, r + 1, c]:
            external_bg[d, r + 1, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r + 1, c
            top += 1
        # c - 1
        if c > 0 and not img[d, r, c - 1] and not external_bg[d, r, c - 1]:
            external_bg[d, r, c - 1] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r, c - 1
            top += 1
        # c + 1
        if c < cols - 1 and not img[d, r, c + 1] and not external_bg[d, r, c + 1]:
            external_bg[d, r, c + 1] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r, c + 1
            top += 1

    return ~external_bg

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

@njit(parallel=True, cache=True)
def fill_from_neighbors(data, mask, neighbors, max_radius=100):
    if mask.sum() == 0: return data
    nb_mask = data.copy()
    set_mask_scalar(nb_mask, ~np.isin(data, neighbors) | mask, np.uint8(0))

    num_empty_voxels = ((nb_mask==0) & mask).sum()
    for _ in range(max_radius):
        nb_mask = dilate(nb_mask, radius=1)
        num_empty_voxels = ((nb_mask==0) & mask).sum()
        if num_empty_voxels==0:
            break

    copy_mask(data, mask, nb_mask)
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


@njit(cache=True, parallel=True)
def separate_labels(data, l1, l2, dist, newlabel=Label.CSF,
                     except_labels=None, except_region=None):
    m1 = nbmorph.dilate_labels_spherical(np.isin(data, l1), dist)
    m2 = nbmorph.dilate_labels_spherical(np.isin(data, l2), dist)
    if except_region is None:
        except_region = np.zeros(shape=data.shape, dtype=np.bool_)
    if except_labels is None:
        return set_mask_scalar(data, m1 & m2 & ~except_region, newlabel)
    return set_mask_scalar(data, m1 & m2 & ~except_region & ~np.isin(data, except_labels), newlabel)


@njit(cache=True, parallel=True)
def enforce_min_thickness(data, label, radius, s="B"):
    mask = data == label
    opened = nbmorph.erode_labels_spherical(mask, radius=radius, struct_sequence=s)
    opened = nbmorph.dilate_labels_spherical(opened, radius=radius, struct_sequence=s)
    diff = mask ^ opened
    dil = nbmorph.dilate_labels_spherical(diff, radius=radius, struct_sequence=s)
    set_mask_scalar(data, dil, label)
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
    )
    dilated_mask = dilate(mask, radius=thickness, struct_sequence="B")
    mask += data == Label.BRAIN_STEM
    mask += data == Label.UNCLASSIFIED
    mask += data == Label.TENTORIUM
    mask += data == Label.FALX
    return set_mask_scalar(data, dilated_mask > mask, Label.CSF)


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


def get_lowest_point(mask):
    idx = np.argwhere(mask)
    ind = np.argsort(idx[:, 2])
    return idx[ind[0]]

@time_func
@njit(cache=True, parallel=True)
def grow_into_region(seed_labels, region_mask, radius=1):
    grown_labels = np.copy(seed_labels)
    set_mask_scalar(grown_labels, ~region_mask, 0)
    n_voxels = (grown_labels > 0).sum()
    while True:
        grown_labels = dilate(grown_labels, radius=radius, struct_sequence="D")
        set_mask_scalar(grown_labels, ~region_mask, 0)
        new_voxels = (grown_labels > 0).sum() - n_voxels
        if new_voxels == 0:
            break
        n_voxels += new_voxels
    return grown_labels

