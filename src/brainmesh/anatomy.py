"""Anatomically-specific operations for brain segmentation cleanup."""
import numpy as np
import nbmorph
from numba import njit, prange
from nbmorph import dilate_labels_spherical as dilate
from nbmorph import erode_labels_spherical as erode

from .labels import Label, VENTRICLE_LABELS, TISSUE_LABELS
from .decorators import track_voxel_changes, plot_voxel_changes, time_func
from .segmentation import (
    enforce_csf_around_falx,
    enforce_csf_around_tentorium,
    enforce_min_thickness,
    separate_labels,
    get_lowest_point,
)
from .ccl import label, remove_small_objects


@njit(cache=True, parallel=True)
def separate_hemispheres(data, distance=4):
    LVs = [Label.LEFT_LATERAL_VENTRICLE + Label.RIGHT_LATERAL_VENTRICLE]
    cc_interface = (
        dilate(data == Label.LEFT_CEREBRAL_WHITE_MATTER, radius=2)
        & dilate(data == Label.RIGHT_CEREBRAL_WHITE_MATTER, radius=2)
        #& dilate(np.isin(data, LVs), radius=30)
    )
    cc_exclusion_mask = dilate(cc_interface + (data == Label.THIRD_VENTRICLE), radius=20)
    return separate_labels(
        data,
        [Label.RIGHT_CEREBRAL_CORTEX, Label.RIGHT_CEREBRAL_WHITE_MATTER],
        [Label.LEFT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_WHITE_MATTER],
        distance,
        except_region=cc_exclusion_mask,
    )


@njit(cache=True)
def separate_cerebellum_and_cerebrum(data, distance=4):
    return separate_labels(
        data,
        [Label.RIGHT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_CORTEX],
        [Label.RIGHT_CEREBELLUM_CORTEX, Label.LEFT_CEREBELLUM_CORTEX],
        distance,
        except_labels=[np.uint8(70)],
    )


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit(cache=True, parallel=True)
def enforce_cortex_layer(data, thickness=1):
    cc_interface = (
        dilate(data == Label.LEFT_CEREBRAL_WHITE_MATTER, radius=2)
        & dilate(data == Label.RIGHT_CEREBRAL_WHITE_MATTER, radius=2)
    )
    cc_exclusion_mask = dilate(cc_interface, radius=10)

    for cortex_id, wm_id in [
        (Label.LEFT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_WHITE_MATTER),
        (Label.RIGHT_CEREBRAL_CORTEX, Label.RIGHT_CEREBRAL_WHITE_MATTER),
    ]:
        mask = data == wm_id
        dilated_mask = dilate(mask, radius=thickness, struct_sequence="BD")
        dilated_mask *= ~cc_exclusion_mask
        data = np.where(
            np.logical_and(dilated_mask, np.isin(data, [0, Label.CSF])),
            cortex_id,
            data,
        )
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit(cache=True, parallel=True)
def enforce_wm_thickness(data, thickness=1):
    for wm_id in [Label.LEFT_CEREBRAL_WHITE_MATTER, Label.RIGHT_CEREBRAL_WHITE_MATTER]:
        data = enforce_min_thickness(data, wm_id, thickness)
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def create_falx(
    data,
    hemisphere_gap=4,
    territory_smoothing_sigma=20,
    boundary_thickness_radius=1,
    cerebrum_proximity_radius=25,
    non_cerebral_clearance_radius=4,
    cerebellum_clearance_radius=2,
    third_ventricle_clearance_radius=30,
    surrounding_csf_radius=1,
):
    from .gaussian import gaussian

    data = separate_hemispheres(data, distance=hemisphere_gap)
    right_mask = np.isin(data, [Label.RIGHT_CEREBRAL_CORTEX, Label.RIGHT_CEREBRAL_WHITE_MATTER])
    left_mask = np.isin(data, [Label.LEFT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_WHITE_MATTER])

    smooth_right = gaussian(right_mask.astype(np.float32),
                             sigma=territory_smoothing_sigma)
    smooth_left = gaussian(left_mask.astype(np.float32),
                            sigma=territory_smoothing_sigma)
    right_territory = smooth_right > smooth_left

    falx_mask = dilate(right_territory, radius=boundary_thickness_radius) ^ right_territory
    falx_mask += dilate(~right_territory, radius=boundary_thickness_radius) ^ (~right_territory)

    falx_mask[~dilate(right_mask | left_mask, radius=cerebrum_proximity_radius)] = 0
    falx_mask[~nbmorph.dilate_labels_spherical(data > 0, radius=1)] = 0

    cc_interface = (
        dilate(data == Label.LEFT_CEREBRAL_WHITE_MATTER, radius=2)
        & dilate(data == Label.RIGHT_CEREBRAL_WHITE_MATTER, radius=2)
    )
    exl_mask = np.isin(data, [
        Label.RIGHT_VENTRAL_DC,
        Label.LEFT_VENTRAL_DC,
        Label.BRAIN_STEM,
    ] + VENTRICLE_LABELS)
    exl_mask += cc_interface

    falx_mask[dilate(exl_mask, radius=non_cerebral_clearance_radius)] = 0

    cerebellum_mask = np.isin(data, [Label.LEFT_CEREBELLUM_CORTEX, Label.RIGHT_CEREBELLUM_CORTEX])
    falx_mask[dilate(cerebellum_mask, radius=cerebellum_clearance_radius)] = 0
    falx_mask[dilate(data == Label.THIRD_VENTRICLE, radius=third_ventricle_clearance_radius)] = 0
    falx_mask = remove_small_objects(falx_mask, max_size=500)

    data[falx_mask] = Label.FALX
    enforce_csf_around_falx(data, radius=surrounding_csf_radius)
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def create_tentorium(
    data,
    cerebrum_cerebellum_gap=4,
    territory_smoothing_sigma=6,
    phantom_cerebellum_sigma_factor=3,
    boundary_thickness_radius=1,
    cerebrum_cerebellum_proximity_radius=12,
    brainstem_clearance_radius=10,
    mask_thickening_radius=1,
    surrounding_csf_radius=1,
):
    from .gaussian import gaussian

    data = separate_cerebellum_and_cerebrum(data, distance=cerebrum_cerebellum_gap)
    cer_mask = np.isin(data, [
        Label.RIGHT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_CORTEX,
        Label.RIGHT_CEREBRAL_WHITE_MATTER, Label.LEFT_CEREBRAL_WHITE_MATTER,
        Label.LEFT_THALAMUS, Label.RIGHT_THALAMUS
    ])
    ceb_mask = np.isin(data, [Label.LEFT_CEREBELLUM_CORTEX,
                              Label.RIGHT_CEREBELLUM_CORTEX,
                              Label.LEFT_CEREBELLUM_WHITE_MATTER,
                              Label.RIGHT_CEREBELLUM_WHITE_MATTER])

    smooth_cer = gaussian(cer_mask.astype(np.float32),
                           sigma=territory_smoothing_sigma)
    smooth_ceb = gaussian(ceb_mask.astype(np.float32),
                           sigma=territory_smoothing_sigma)
    phantom_ceb = gaussian(
        ceb_mask.astype(np.float32),
        sigma=territory_smoothing_sigma * phantom_cerebellum_sigma_factor,
    )
    cer_territory = smooth_cer > np.maximum(smooth_ceb, phantom_ceb)

    tent_mask = dilate(cer_territory, radius=boundary_thickness_radius) ^ cer_territory
    tent_mask[~dilate(ceb_mask + cer_mask, radius=cerebrum_cerebellum_proximity_radius)] = 0
    tent_mask[dilate(np.isin(data, [Label.BRAIN_STEM,
                                    Label.LEFT_VENTRAL_DC,
                                    Label.RIGHT_VENTRAL_DC,
                                    Label.LEFT_THALAMUS,
                                    Label.RIGHT_THALAMUS]), radius=brainstem_clearance_radius)] = 0
    tent_mask = dilate(tent_mask, radius=mask_thickening_radius)
    tent_mask[data==0] = 0

    tent_mask = remove_small_objects(tent_mask, max_size=int(tent_mask.sum() * 0.2))
    data[tent_mask] = Label.TENTORIUM
    enforce_csf_around_tentorium(data, radius=surrounding_csf_radius)
    data[(~cer_territory) & (phantom_ceb > 0) & (data == Label.FALX)] = Label.CSF
    return data

@track_voxel_changes
def connect_islands(mask, maxdist=np.inf, radius=1):
    if mask.sum() == 0: return mask
    from itertools import combinations
    labels, N = label(mask)
    for i,j in combinations(range(1, N+1), r=2):
        l = _connect_by_line(labels==i, labels==j, radius=radius, maxdist=maxdist)
        mask[l] = True
    return mask

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def build_inferior_lateral_ventricle_horns(
    data,
    horn_closing_radius=15,
    post_close_dilation_radius=1,
    smoothing_radius=2,
    debug=False
):
    from .segmentation import fill_from_neighbors
    LV_INF = [Label.LEFT_INFERIOR_LATERAL_VENTRICLE, Label.RIGHT_INFERIOR_LATERAL_VENTRICLE]
    LV = [Label.LEFT_LATERAL_VENTRICLE, Label.RIGHT_LATERAL_VENTRICLE]
    CP = [Label.LEFT_CHOROID_PLEXUS, Label.RIGHT_CHOROID_PLEXUS]
    for LVINFID, LVID, CPID in zip(LV_INF, LV, CP):
        mask = data == LVINFID
        if mask.sum()==0: continue
        new_mask = remove_small_objects(mask, max_size=int(mask.sum() * 0.05))
        data = fill_from_neighbors(data, mask != new_mask, TISSUE_LABELS)
        mask = new_mask
        mask = connect_islands(mask, radius=1, maxdist=20)
        mask += _connect_by_line(mask, data==LVID, radius=2)
        mask = dilate(
            nbmorph.close_labels_spherical(mask, radius=horn_closing_radius),
            radius=post_close_dilation_radius,
        )
        mask = nbmorph.smooth_labels_spherical(mask, smoothing_radius)
        mask[data == CPID] = 0
        if debug:
            labels, N = label(mask)
            assert N == 1
        data[mask] = LVINFID
    return data


@njit(parallel=True, cache=True)
def _find_closest_pair_numba(coords1, coords2):
    """
    Finds the indices of the closest pair of points between two coordinate arrays.
    Uses prange to parallelize the outer loop safely.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    dims = coords1.shape[1]
    
    # Pre-allocate arrays to store the minimums for each thread
    # This prevents race conditions in the parallel loop
    min_dists = np.full(n1, np.inf)
    best_j = np.zeros(n1, dtype=np.int64)
    
    for i in prange(n1):
        c1 = coords1[i]
        local_min = np.inf
        local_j = -1
        
        for j in range(n2):
            c2 = coords2[j]
            
            # Calculate squared Euclidean distance
            dist_sq = 0.0
            for d in range(dims):
                diff = c1[d] - c2[d]
                dist_sq += diff * diff
                
            if dist_sq < local_min:
                local_min = dist_sq
                local_j = j
                
        min_dists[i] = local_min
        best_j[i] = local_j
        
    # Find the global minimum from the threaded results
    best_i = np.argmin(min_dists)
    return best_i, best_j[best_i]


@njit(cache=True)
def _get_closest_points(m1, m2):

    b1 = m1 ^ erode(m1)
    b2 = m2 ^ erode(m2)

    coords1 = np.argwhere(b1)
    coords2 = np.argwhere(b2)
    idx1, idx2 = _find_closest_pair_numba(coords1, coords2)

    return coords1[idx1], coords2[idx2]

@njit(cache=True)
def _draw_3d_line_numba(shape, p1, p2, maxdist):
    """
    Computes distance and uses a 3D DDA algorithm to draw a line 
    in a pre-allocated array. Fully Numba compatible.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    # 1. Compute Distance mathematically
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # If distance is too far, return empty array and success=False flag
    if dist > maxdist:
        return np.zeros(shape, dtype=np.bool_), dist, False
        
    # 2. Draw Line (DDA Rasterization Algorithm)
    conn = np.zeros(shape, dtype=np.bool_)
    steps = max(abs(dx), abs(dy), abs(dz))
    
    if steps == 0:
        conn[x1, y1, z1] = True
        return conn, dist, True
        
    x_inc = dx / steps
    y_inc = dy / steps
    z_inc = dz / steps
    
    # Cast to float for sub-pixel accuracy during iteration
    x = float(x1)
    y = float(y1)
    z = float(z1)
    
    # March along the vector, rounding to the nearest integer voxel
    for _ in range(steps + 1):
        conn[int(round(x)), int(round(y)), int(round(z))] = True
        x += x_inc
        y += y_inc
        z += z_inc
        
    return conn, dist, True

@njit(cache=True)
def _connect_by_line(m1, m2, radius=2, maxdist=np.inf):
    # Retrieve both points (using the optimized function from earlier)
    pointa, pointb = _get_closest_points(m1, m2)
    
    # Let Numba handle the distance math, logic, and array allocation
    conn, dist, success = _draw_3d_line_numba(m1.shape, pointa, pointb, maxdist)
    
    if not success:
        print(f"maxdist exceeded: {dist}")
        return conn  # This will correctly return the array of all Falses
    
    # Return the connected array (Add your dilate call back here if you still need it!)
    return dilate(conn, radius=radius, struct_sequence="B")

@track_voxel_changes
def solidify_label(data, ID, closing_radius=0, smoothing_radius=0,
                max_island_size=None, max_connection_dist=10,
                connect=True, neighbor_labels=TISSUE_LABELS):
    from .segmentation import fill_from_neighbors
    old_mask = (data == ID)
    if max_island_size is None:
        max_island_size=int(old_mask.sum() * 0.05)
    mask = remove_small_objects(old_mask.copy(), max_size=max_island_size)
    print(f"found small objects: {(mask != old_mask).sum()} voxels")
    if connect:
        mask = connect_islands(mask, radius=1, maxdist=max_connection_dist)
    if closing_radius:
        mask = nbmorph.close_labels_spherical(mask, radius=closing_radius)
    if smoothing_radius:
        mask = nbmorph.smooth_labels_spherical(mask, smoothing_radius)
    data = fill_from_neighbors(data, old_mask & ~mask, neighbor_labels)
    data[mask] = ID
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_connected_ventricles(data, connection_radius=2):
    from .labels import reverse_label_map
    # make sure all parts of each part of the ventricles is connected
    for v_id in VENTRICLE_LABELS:
        print(f"solidifying {reverse_label_map[v_id]}")
        is_CP = v_id==Label.LEFT_CHOROID_PLEXUS or v_id==Label.RIGHT_CHOROID_PLEXUS
        data = solidify_label(data, v_id, connect=~is_CP,
                               neighbor_labels=TISSUE_LABELS + VENTRICLE_LABELS)

    # make sure the AQ (between V3 and V4) is there
    V4_mask = data == Label.FOURTH_VENTRICLE
    V3_mask = data == Label.THIRD_VENTRICLE
    aq_conn = _connect_by_line(V3_mask, V4_mask, radius=connection_radius)
    data[aq_conn] = Label.FOURTH_VENTRICLE

    # make sure the LVs are connected to V3
    RLV_mask = data == Label.RIGHT_LATERAL_VENTRICLE
    LLV_mask = data == Label.LEFT_LATERAL_VENTRICLE
    fm_conn = _connect_by_line(V3_mask, RLV_mask, radius=connection_radius)
    fm_conn += _connect_by_line(V3_mask, LLV_mask, radius=connection_radius)
    data[fm_conn] = Label.THIRD_VENTRICLE

    # test whether all parts are connected
    labels, num_features = label(np.isin(data, VENTRICLE_LABELS))

    if num_features > 1:
        import pyvista as pv
        import os
        grid = pv.ImageData(dimensions=[i + 1 for i in data.shape])
        data[~np.isin(data, VENTRICLE_LABELS)] = 0
        grid.cell_data["data"] = data.flatten(order="F")
        grid.cell_data["label"] = labels.flatten(order="F")
        os.makedirs("debug", exist_ok=True)
        debug_name = f"debug/ventricles_{np.random.randint(low=0, high=10000)}.vti"
        grid.save(debug_name)
        print(f"enforcing connected ventricles failed. See {debug_name}.")
        assert num_features ==1

    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_tight_ventricles(
    data,
    surrounding_layer_thickness=2,
    bottom_exclusion_z_offset=20,
    tissue_fill_radius=10,
):
    ventricle_mask = np.isin(data, VENTRICLE_LABELS)
    v_layer = dilate(ventricle_mask, radius=surrounding_layer_thickness) ^ ventricle_mask
    v_lowest = get_lowest_point(v_layer)
    v_layer[:, :, :v_lowest[2] + bottom_exclusion_z_offset] = 0
    tissue = data.copy()
    tissue[np.isin(tissue, VENTRICLE_LABELS + [Label.CSF])] = 0
    tissue = dilate(tissue, radius=tissue_fill_radius)
    data = np.where(v_layer, tissue, data)
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def extend_brainstem_caudally(
    data,
    footprint_z_offset=12,
    footprint_closing_radius=4,
    csf_buffer_radius=4,
):
    """Extends the brainstem downwards through the CSF to the bottom of the image."""
    lowest_brain_stem = get_lowest_point(data == Label.BRAIN_STEM)[2]
    lowest_csf = get_lowest_point(data == Label.CSF)[2]

    extend_down = 0
    footprint = data[:, :, lowest_brain_stem + footprint_z_offset:lowest_brain_stem + footprint_z_offset + 1] == Label.BRAIN_STEM
    footprint = nbmorph.close_labels_spherical(footprint, radius=footprint_closing_radius)
    footprint_csf = nbmorph.dilate_labels_spherical(footprint, radius=csf_buffer_radius)

    z_min = max(0, min(lowest_brain_stem, lowest_csf) - extend_down)
    z_max = max(lowest_brain_stem, lowest_csf) + footprint_z_offset
    target_block = data[:, :, z_min:z_max]

    for fp, l in [(footprint_csf, Label.CSF), (footprint, Label.BRAIN_STEM)]:
        mask_to_replace = fp & np.logical_or(target_block == Label.CSF, target_block == 0)
        target_block[mask_to_replace] = l
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def extend_brainstem(data, csf_z_tolerance=2, extension_dilation_radius=2):
    dil_stem_mask = data == Label.BRAIN_STEM
    lowest_csf_z = get_lowest_point(data == Label.CSF)[2]
    lowest_bs_z = get_lowest_point(data == Label.BRAIN_STEM)[2]

    while lowest_bs_z > lowest_csf_z + csf_z_tolerance:
        dil_stem_mask = dilate(dil_stem_mask, extension_dilation_radius)
        dil_stem_mask[data > 0] = 0
        lowest_bs_z = get_lowest_point(dil_stem_mask)[2]

    dil_stem_mask[:, :, :lowest_csf_z] = 0
    data[dil_stem_mask] = Label.BRAIN_STEM
    return data
