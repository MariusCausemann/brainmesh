"""brainmesh — create tetrahedral brain meshes from segmentations."""

from .labels import (Label, reverse_label_map, VENTRICLE_LABELS,
                     GM_LABELS, WM_LABELS,
                     GM_CEREBELLUM_LABELS, WM_CEREBELLUM_LABELS,
                     TISSUE_LABELS,
                     SAS_LABEL_OFFSET, SPINAL_ID,
                     fs_aparc_to_sas_marker, sas_marker_to_fs_aparc)
from .io import nibabel_to_pyvista, read_mesh, save_mesh, upsample_nib, get_img
from .segmentation import (
    solidify_csf,
    close_csf_space,
    fill_holes_csf,
    fill_wm_hyperintensities,
    cut_bottom,
    enforce_min_thickness,
    enforce_csf_layer,
    enforce_csf_around_tentorium,
    enforce_csf_around_falx,
    grow_into_region
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
from .surface import transfer_labels, coarsen_surface, straighten_spinal_interface
from .mesh import (
    mark_mesh,
    remark_csf_with_sas,
    load_marked_mesh,
    filter_by_mask,
    extract_csf,
    mark_interface_facets,
    mark_boundary_facets,
    mark_spinal_boundary,
    mark_facets,
)
from .mesh_optimizer import run_mesh_optimization
from .phantom import make_phantom_seg
from .config import SegmentationConfig

__all__ = [
    "Label",
    "VENTRICLE_LABELS",
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
    "transfer_labels",
    "coarsen_surface",
    "straighten_spinal_interface",
    "mark_mesh",
    "remark_csf_with_sas",
    "load_marked_mesh",
    "filter_by_mask",
    "extract_csf",
    "mark_interface_facets",
    "mark_boundary_facets",
    "mark_spinal_boundary",
    "mark_facets",
    "make_phantom_seg",
    "SegmentationConfig",
]
