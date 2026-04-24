"""Convenience script — wraps the surface-to-mesh pipeline CLI."""
from brainmesh.pipeline import surface_to_mesh

surface_to_mesh(
    surf_path="results/surf_dec.vtk",
    out_dir="results",
    numba_threads=8,
)
