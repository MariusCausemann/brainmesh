"""End-to-end integration test: full pipeline on the synthetic brain phantom."""
import nibabel as nib
import numpy as np
import pytest

from brainmesh.mesh import (
    extract_csf,
    mark_boundary_facets,
    mark_interface_facets,
)
from brainmesh.phantom import make_phantom_seg
from brainmesh.pipeline import segmentation_to_surface, surface_to_mesh


@pytest.mark.slow
def test_phantom_pipeline_end_to_end(tmp_path):
    """Run cleanup → surface → tets → facets on a 100^3 phantom."""
    seg_path = tmp_path / "phantom_seg.nii.gz"
    nib.save(make_phantom_seg(shape=(100, 100, 100), spacing=0.5), seg_path)

    surf = segmentation_to_surface(seg_path, out_dir=tmp_path, numba_threads=4)
    assert surf.n_cells > 0
    assert "boundary_labels" in surf.cell_data
    region_labels = set(np.unique(surf["boundary_labels"]).tolist())
    # Background plus several brain regions should be present
    assert 0 in region_labels
    assert len(region_labels) >= 5

    mesh = surface_to_mesh(
        tmp_path / "surf.vtk",
        out_dir=tmp_path,
        edge_length_fac=0.08,
        quiet=True,
    )
    assert mesh.n_cells > 0
    assert "marker" in mesh.cell_data
    markers = np.unique(mesh["marker"])
    assert (markers > 0).all()
    assert len(markers) >= 5

    csf = extract_csf(mesh)
    assert csf.n_cells > 0
    assert csf.n_cells < mesh.n_cells

    interfaces = mark_interface_facets(mesh)
    boundaries = mark_boundary_facets(mesh)
    assert interfaces.n_cells > 0
    assert boundaries.n_cells > 0

    # Both facet meshes share the parent's vertex indices
    assert interfaces.n_points == mesh.n_points
    assert boundaries.n_points == mesh.n_points
    np.testing.assert_array_equal(interfaces.points, mesh.points)
    np.testing.assert_array_equal(boundaries.points, mesh.points)

    # Conservation: 4·n_tets = 2·n_interior + n_boundary
    n_total = 4 * mesh.n_cells
    assert (n_total - boundaries.n_cells) % 2 == 0
    n_interior = (n_total - boundaries.n_cells) // 2
    assert n_interior >= interfaces.n_cells
