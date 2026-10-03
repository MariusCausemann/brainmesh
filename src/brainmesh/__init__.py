"""brainmesh — create tetrahedral brain meshes from segmentations."""

from .labels import (Label, reverse_label_map, VENTRICLE_LABELS,
                     GM_LABELS, WM_LABELS,
                     GM_CEREBELLUM_LABELS, WM_CEREBELLUM_LABELS,
                     TISSUE_LABELS, CSF_LABELS, is_csf_marker,
                     SAS_LABEL_OFFSET, SPINAL_ID,
                     fs_aparc_to_sas_marker, sas_marker_to_fs_aparc)
from imagemesh.facets import mark_boundary_facets
from imagemesh.image import get_img, nibabel_to_pyvista, upsample_nib
from imagemesh.io import load_marked_mesh, read_mesh, save_mesh
from imagemesh.mesh_optimizer import run_mesh_optimization
from imagemesh.morphology import enforce_min_thickness, grow_into_region
from imagemesh.tetmesh import filter_by_mask
from .segmentation import (
    solidify_csf,
    close_csf_space,
    fill_holes_csf,
    fill_wm_hyperintensities,
    cut_bottom,
    enforce_csf_layer,
    count_background_contacts,
    enforce_csf_around_tentorium,
    enforce_csf_around_falx,
    fill_small_unclassified_fragments
)
from .anatomy import (
    create_falx,
    create_tentorium,
    enforce_cortex_layer,
    enforce_wm_thickness,
    build_inferior_lateral_ventricle_horns,
    enforce_connected_ventricles,
    enforce_tight_ventricles,
    extend_brainstem,
    extend_brainstem_caudally,
    _connect_by_line
)
from .mesh import (
    remark_csf_with_sas,
    extract_csf,
    mark_interface_facets,
    mark_spinal_boundary,
    spinal_interface_mask,
    mark_facets,
)
from .phantom import make_phantom_seg
from .config import SegmentationConfig
from .vessels import (read_centerlines, centerline_distance, mask_distance,
                      vessel_distance, enforce_vessel_sleeve)

__all__ = [
    "Label",
    "VENTRICLE_LABELS",
    "CSF_LABELS",
    "is_csf_marker",
    "SAS_LABEL_OFFSET",
    "SPINAL_ID",
    "fs_aparc_to_sas_marker",
    "sas_marker_to_fs_aparc",
    "nibabel_to_pyvista",
    "read_mesh",
    "save_mesh",
    "upsample_nib",
    "solidify_csf",
    "close_csf_space",
    "fill_holes_csf",
    "fill_wm_hyperintensities",
    "cut_bottom",
    "enforce_min_thickness",
    "enforce_csf_layer",
    "count_background_contacts",
    "enforce_csf_around_tentorium",
    "enforce_csf_around_falx",
    "create_falx",
    "create_tentorium",
    "enforce_cortex_layer",
    "enforce_wm_thickness",
    "build_inferior_lateral_ventricle_horns",
    "enforce_connected_ventricles",
    "enforce_tight_ventricles",
    "extend_brainstem",
    "extend_brainstem_caudally",
    "remark_csf_with_sas",
    "load_marked_mesh",
    "filter_by_mask",
    "extract_csf",
    "mark_interface_facets",
    "mark_boundary_facets",
    "mark_spinal_boundary",
    "spinal_interface_mask",
    "mark_facets",
    "make_phantom_seg",
    "SegmentationConfig",
    "read_centerlines",
    "centerline_distance",
    "mask_distance",
    "vessel_distance",
    "enforce_vessel_sleeve",
]
