"""Anatomically-specific operations for brain segmentation cleanup."""
import numpy as np
import nbmorph
import skimage
from skimage import morphology
from skimage import measure
from numba import njit
from edt import edt
from nbmorph import dilate_labels_spherical as dilate
from nbmorph import erode_labels_spherical as erode

from .labels import Label, VENTRICLE_LABELS
from .decorators import track_voxel_changes, plot_voxel_changes, time_func
from .segmentation import (
    enforce_csf_around_falx,
    enforce_csf_around_tentorium,
    enforce_min_thickness,
    separate_labels,
    get_lowest_point,
)


@njit
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


@njit
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
@njit
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
@njit
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
    from skimage.filters import gaussian

    data = separate_hemispheres(data, distance=hemisphere_gap)
    right_mask = np.isin(data, [Label.RIGHT_CEREBRAL_CORTEX, Label.RIGHT_CEREBRAL_WHITE_MATTER])
    left_mask = np.isin(data, [Label.LEFT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_WHITE_MATTER])

    smooth_right = gaussian(right_mask.astype(np.float32),
                             sigma=territory_smoothing_sigma, truncate=2)
    smooth_left = gaussian(left_mask.astype(np.float32),
                            sigma=territory_smoothing_sigma, truncate=2)
    right_territory = smooth_right > smooth_left

    falx_mask = dilate(right_territory, radius=boundary_thickness_radius) ^ right_territory
    falx_mask += dilate(~right_territory, radius=boundary_thickness_radius) ^ (~right_territory)

    falx_mask[~dilate(right_mask | left_mask, radius=cerebrum_proximity_radius)] = 0
    falx_mask[~nbmorph.close_labels_spherical(data > 0, radius=1)] = 0

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
    falx_mask = morphology.remove_small_objects(falx_mask, max_size=500)

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
    from skimage.filters import gaussian

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
                           sigma=territory_smoothing_sigma, truncate=2)
    smooth_ceb = gaussian(ceb_mask.astype(np.float32),
                           sigma=territory_smoothing_sigma, truncate=2)
    phantom_ceb = gaussian(
        ceb_mask.astype(np.float32),
        sigma=territory_smoothing_sigma * phantom_cerebellum_sigma_factor,
        truncate=2
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

    tent_mask = morphology.remove_small_objects(tent_mask, max_size=500)
    data[tent_mask] = Label.TENTORIUM
    enforce_csf_around_tentorium(data, radius=surrounding_csf_radius)
    data[(~cer_territory) & (data == Label.FALX)] = Label.CSF
    return data


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def build_inferior_lateral_ventricle_horns(
    data,
    horn_closing_radius=15,
    post_close_dilation_radius=1,
    smoothing_radius=2,
):
    LV_INF = [Label.LEFT_INFERIOR_LATERAL_VENTRICLE, Label.RIGHT_INFERIOR_LATERAL_VENTRICLE]
    LV = [Label.LEFT_LATERAL_VENTRICLE, Label.RIGHT_LATERAL_VENTRICLE]
    CP = [Label.LEFT_CHOROID_PLEXUS, Label.RIGHT_CHOROID_PLEXUS]
    for LVINFID, LVID, CPID in zip(LV_INF, LV, CP):
        mask = data == LVINFID
        mask = dilate(
            nbmorph.close_labels_spherical(mask, radius=horn_closing_radius),
            radius=post_close_dilation_radius,
        )
        mask = nbmorph.smooth_labels_spherical(mask, smoothing_radius)
        mask[data == CPID] = 0
        data[mask] = LVINFID
    return data


def _get_closest_point(a, b):
    dist = edt(a == False)
    dist[b == False] = np.inf
    minidx = np.unravel_index(np.argmin(dist), a.shape)
    return minidx


def _connect_by_line(m1, m2, radius=2):
    pointa = _get_closest_point(m1, m2)
    pointb = _get_closest_point(m2, m1)
    line = np.array(skimage.draw.line_nd(pointa, pointb, endpoint=True))
    conn = np.zeros_like(m1)
    i, j, k = line
    conn[i, j, k] = 1
    return dilate(conn, radius=radius, struct_sequence="B")


@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_connected_ventricles(data, connection_radius=2, mask_smoothing_radius=2):
    V4_mask = data == Label.FOURTH_VENTRICLE
    V3_mask = data == Label.THIRD_VENTRICLE
    aq_conn = _connect_by_line(V3_mask, V4_mask, radius=connection_radius)
    data[aq_conn] = Label.FOURTH_VENTRICLE

    RLV_mask = data == Label.RIGHT_LATERAL_VENTRICLE
    LLV_mask = data == Label.LEFT_LATERAL_VENTRICLE
    fm_conn = _connect_by_line(V3_mask, RLV_mask, radius=connection_radius)
    fm_conn += _connect_by_line(V3_mask, LLV_mask, radius=connection_radius)
    data[fm_conn] = Label.THIRD_VENTRICLE
    label, num_features = measure.label(np.isin(data, VENTRICLE_LABELS),
                                         connectivity=2, return_num=True)
    if num_features > 1:
        import pyvista as pv
        grid = pv.ImageData(dimensions=[i + 1 for i in data.shape])
        data[~np.isin(data, VENTRICLE_LABELS)] = 0
        grid.cell_data["data"] = data.flatten(order="F")
        grid.cell_data["label"] = label.flatten(order="F")
        debug_name = f"ventricles_{np.random.randint(low=0, high=10000)}.vti"
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
