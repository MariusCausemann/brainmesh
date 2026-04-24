"""Tests for filter / facet-extraction operations on marked tet meshes."""
import numpy as np
import pyvista as pv
import pytest

from brainmesh import Label
from brainmesh.mesh import (
    extract_csf,
    filter_by_label,
    mark_boundary_facets,
    mark_interface_facets,
)


@pytest.fixture(scope="module")
def split_box_mesh():
    """
    A pytetwild-meshed box [0,2]×[0,1]×[0,1] split at x=1.0 into two
    regions with markers 10 (left) and 20 (right).
    """
    import pytetwild

    surf = pv.Box(bounds=(0, 2, 0, 1, 0, 1)).triangulate().subdivide(2)
    mesh = pytetwild.tetrahedralize_pv(
        surf,
        edge_length_fac=0.1,
        stop_energy=10,
        quiet=True,
    )
    centroids = mesh.cell_centers().points
    mesh.cell_data["marker"] = np.where(centroids[:, 0] < 1.0, 10, 20).astype(np.int32)
    return mesh


@pytest.mark.slow
def test_split_box_has_many_tets(split_box_mesh):
    assert split_box_mesh.n_cells > 100
    types = np.unique(split_box_mesh.celltypes)
    assert types.tolist() == [pv.CellType.TETRA]


@pytest.mark.slow
def test_filter_by_label_partitions_mesh(split_box_mesh):
    n_total = split_box_mesh.n_cells
    left = filter_by_label(split_box_mesh, 10)
    right = filter_by_label(split_box_mesh, 20)
    assert left.n_cells > 0 and right.n_cells > 0
    assert left.n_cells + right.n_cells == n_total
    assert np.all(left.cell_data["marker"] == 10)
    assert np.all(right.cell_data["marker"] == 20)


def test_extract_csf_picks_csf_and_ventricles():
    """Two-tet construction — fast, verifies the label set used by extract_csf."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [0, 0, 1], [0, 0, -1]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3,  4, 0, 1, 2, 4])
    cell_types = np.array([pv.CellType.TETRA, pv.CellType.TETRA])

    for csf_label in (Label.CSF, Label.LEFT_LATERAL_VENTRICLE,
                      Label.LEFT_CHOROID_PLEXUS):
        mesh = pv.UnstructuredGrid(cells, cell_types, points)
        mesh.cell_data["marker"] = np.array([csf_label, Label.LEFT_CEREBRAL_CORTEX])
        csf = extract_csf(mesh)
        assert csf.n_cells == 1
        assert csf.cell_data["marker"][0] == csf_label


@pytest.mark.slow
def test_mark_interface_facets_split_box(split_box_mesh):
    interfaces = mark_interface_facets(split_box_mesh)
    assert interfaces.n_cells > 0

    # Point array must match parent — required for FEniCS facet-function use
    assert interfaces.n_points == split_box_mesh.n_points
    np.testing.assert_array_equal(interfaces.points, split_box_mesh.points)

    # Every interface lies between markers 10 and 20
    np.testing.assert_array_equal(interfaces.cell_data["region_a"], 10)
    np.testing.assert_array_equal(interfaces.cell_data["region_b"], 20)
    np.testing.assert_array_equal(interfaces.cell_data["interface_id"], 10 * 1000 + 20)

    # Interface centroid x ≈ split plane
    face_verts = interfaces.faces.reshape(-1, 4)[:, 1:]
    centroid_x = split_box_mesh.points[face_verts][..., 0].mean()
    assert abs(centroid_x - 1.0) < 0.05


@pytest.mark.slow
def test_mark_interface_facets_same_marker_yields_none(split_box_mesh):
    mesh = split_box_mesh.copy()
    mesh.cell_data["marker"][:] = 10
    interfaces = mark_interface_facets(mesh)
    assert interfaces.n_cells == 0
    # Empty PolyData still carries the parent's points
    assert interfaces.n_points == mesh.n_points


@pytest.mark.slow
def test_mark_boundary_facets_split_box(split_box_mesh):
    boundaries = mark_boundary_facets(split_box_mesh)
    assert boundaries.n_cells > 0
    assert boundaries.n_points == split_box_mesh.n_points
    np.testing.assert_array_equal(boundaries.points, split_box_mesh.points)

    # Boundary markers must come from the two region IDs
    assert set(np.unique(boundaries.cell_data["boundary"])) <= {10, 20}

    # Total boundary area should match the box surface (2*(2*1) + 2*(2*1) + 2*(1*1) = 10)
    assert np.isclose(boundaries.area, 10.0, rtol=0.01)

    # And it should agree with PyVista's extract_surface for cross-validation
    surf_area = split_box_mesh.extract_surface().triangulate().area
    assert np.isclose(boundaries.area, surf_area, rtol=0.01)


@pytest.mark.slow
def test_facet_face_count_conservation(split_box_mesh):
    """4·n_tets == 2·n_interior + n_boundary; n_interface ≤ n_interior."""
    interfaces = mark_interface_facets(split_box_mesh)
    boundaries = mark_boundary_facets(split_box_mesh)

    n_total_face_uses = 4 * split_box_mesh.n_cells
    assert (n_total_face_uses - boundaries.n_cells) % 2 == 0
    n_interior = (n_total_face_uses - boundaries.n_cells) // 2
    assert n_interior >= interfaces.n_cells
