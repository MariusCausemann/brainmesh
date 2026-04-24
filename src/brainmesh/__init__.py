"""brainmesh — create tetrahedral brain meshes from segmentations."""

from .labels import Label, VENTRICLE_LABELS
from .io import nibabel_to_pyvista, upsample_nib
from .segmentation import (
    solidify_csf,
    close_csf_space,
    fill_holes_csf,
    fill_wm_hyperintensities,
    cut_bottom,
    diamond_mode_filter,
    enforce_min_thickness,
    enforce_csf_layer,
    enforce_csf_around_tentorium,
    enforce_csf_around_falx,
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
)
from .surface import transfer_labels, coarsen_surface, straighten_spinal_interface
from .mesh import mark_mesh

__all__ = [
    "Label",
    "VENTRICLE_LABELS",
    "nibabel_to_pyvista",
    "upsample_nib",
    "solidify_csf",
    "close_csf_space",
    "fill_holes_csf",
    "fill_wm_hyperintensities",
    "cut_bottom",
    "diamond_mode_filter",
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
]
