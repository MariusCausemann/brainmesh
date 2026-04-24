"""Convenience script — wraps the segmentation-to-surface pipeline CLI."""
from brainmesh.pipeline import segmentation_to_surface

segmentation_to_surface(
    seg_path="testdata/sub1_gouhfi_hybrid_seg.nii.gz",
    out_dir="results",
    numba_threads=8,
)
