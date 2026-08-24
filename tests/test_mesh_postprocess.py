"""Tests for filter / facet-extraction operations on marked tet meshes."""
import numpy as np
import pyvista as pv
import pytest

from brainmesh import Label
from brainmesh.labels import SPINAL_ID
from brainmesh.mesh import (
    extract_csf,
    group_csf_facets_by_region,
    mark_boundary_facets,
    mark_facets,
    mark_interface_facets,
    mark_spinal_boundary,
    remark_csf_with_sas,
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
    surf_area = split_box_mesh.extract_surface(algorithm="dataset_surface").triangulate().area
    assert np.isclose(boundaries.area, surf_area, rtol=0.01)


@pytest.fixture(scope="module")
def spinal_box_mesh():
    """
    Box [0,2]×[0,1]×[0,1] split at z=0.25: the bottom slab is Label.SPINAL_BUFFER,
    everything above is Label.CSF.

    The expected spinal interface is the z=0.25 plane between the two.
    """
    import pytetwild
    surf = pv.Box(bounds=(0, 2, 0, 1, 0, 1)).triangulate().subdivide(2)
    mesh = pytetwild.tetrahedralize_pv(surf, edge_length_fac=0.1, stop_energy=10, quiet=True)
    centroids = mesh.cell_centers().points
    mesh.cell_data["marker"] = np.where(
        centroids[:, 2] < 0.25,
        Label.SPINAL_BUFFER,
        Label.CSF,
    ).astype(np.int32)
    return mesh


@pytest.mark.slow
def test_mark_spinal_boundary_finds_buffer_interface(spinal_box_mesh):
    spinal = mark_spinal_boundary(spinal_box_mesh)
    assert spinal.n_cells > 0
    # All selected facet centroids must sit on the buffer/CSF split plane.
    centroids = spinal.cell_centers().points
    assert np.allclose(centroids[:, 2], 0.25, atol=0.1)
    # The recorded marker is the CSF side, never the buffer.
    assert np.all(spinal.cell_data["boundary"] == Label.CSF)


@pytest.mark.slow
def test_mark_spinal_boundary_preserves_point_array(spinal_box_mesh):
    spinal = mark_spinal_boundary(spinal_box_mesh)
    assert spinal.n_points == spinal_box_mesh.n_points
    np.testing.assert_array_equal(spinal.points, spinal_box_mesh.points)


@pytest.mark.parametrize("csf_marker", [Label.CSF, 11003])
def test_mark_spinal_boundary_normals_point_out_of_csf(csf_marker):
    """
    Winding must give normals pointing away from the CSF and into the buffer.

    Two tets sharing the z=0 face: CSF above, SPINAL_BUFFER below, so the expected
    normal is -z.  Both marker orderings are covered: CSF (24) sorts below the
    buffer (73), a SAS parcel (11003) above it, and only the latter needs flipping.
    """
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [0, 0, 1], [0, 0, -1]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3,  4, 0, 1, 2, 4])
    cell_types = np.array([pv.CellType.TETRA, pv.CellType.TETRA])
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.cell_data["marker"] = np.array([csf_marker, Label.SPINAL_BUFFER], dtype=np.int32)

    spinal = mark_spinal_boundary(mesh)
    assert spinal.n_cells == 1
    assert spinal.cell_data["boundary"][0] == csf_marker

    corners = points[spinal.faces.reshape(-1, 4)[:, 1:]]
    normal = np.cross(corners[0, 1] - corners[0, 0], corners[0, 2] - corners[0, 0])
    assert normal[2] < 0


@pytest.mark.slow
def test_mark_spinal_boundary_ignores_non_csf_neighbours(spinal_box_mesh):
    """A buffer facing white matter is not a spinal opening."""
    mesh = spinal_box_mesh.copy()
    marker = mesh.cell_data["marker"]
    mesh.cell_data["marker"] = np.where(marker == Label.CSF,
                                        Label.LEFT_CEREBRAL_WHITE_MATTER,
                                        marker).astype(np.int32)
    assert mark_spinal_boundary(mesh).n_cells == 0


@pytest.mark.slow
def test_mark_spinal_boundary_without_buffer_is_empty(spinal_box_mesh):
    """No SPINAL_BUFFER in the mesh at all -> no spinal facets."""
    mesh = spinal_box_mesh.copy()
    marker = mesh.cell_data["marker"]
    mesh.cell_data["marker"] = np.where(marker == Label.SPINAL_BUFFER,
                                        Label.BRAIN_STEM,
                                        marker).astype(np.int32)
    assert mark_spinal_boundary(mesh).n_cells == 0


@pytest.mark.slow
def test_mark_facets_tags_spinal_interface(spinal_box_mesh):
    """mark_facets replaces the buffer/CSF encoding with SPINAL_ID."""
    facets = mark_facets(spinal_box_mesh)
    ids = np.asarray(facets.cell_data["interface_id"])
    n_spinal = mark_spinal_boundary(spinal_box_mesh).n_cells
    assert (ids == SPINAL_ID).sum() == n_spinal
    # the raw min(24,73)*base+max(24,73) encoding must be gone
    assert not (ids == int(Label.CSF) * 100000 + int(Label.SPINAL_BUFFER)).any()
    # outer boundary facets keep their region marker
    assert set(np.unique(ids)) <= {SPINAL_ID, int(Label.CSF), int(Label.SPINAL_BUFFER)}


def test_group_csf_facets_separates_unclassified_from_pia():
    """CSF-to-UNCLASSIFIED facets get their own region instead of falling into PIA."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                       [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
    faces = np.hstack([[3, 0, 1, 2], [3, 1, 2, 3], [3, 4, 5, 6]])
    facets = pv.PolyData(points, faces=faces)
    base = 100000
    facets.cell_data["interface_id"] = np.array([
        int(Label.CSF) * base + int(Label.UNCLASSIFIED),      # vessel wall
        int(Label.CSF) * base + int(Label.LEFT_CEREBRAL_CORTEX),  # pia
        SPINAL_ID,                                            # spinal opening
    ], dtype=np.int64)

    out, region_labels = group_csf_facets_by_region(facets)
    region = np.asarray(out.cell_data["region"])
    assert region_labels["UNCLASSIFIED"] == 7
    assert region.tolist() == [region_labels["UNCLASSIFIED"],
                               region_labels["PIA"],
                               region_labels["SPINAL_CSF"]]


def test_remark_csf_with_sas():
    """
    Build a two-tet mesh (one CSF, one WM) and a tiny NIfTI where the CSF tet
    centroid falls on a voxel labelled 1001.  Verify that:
    - the CSF tet gets marker 1001 + SAS_LABEL_OFFSET = 11001
    - the WM tet is untouched
    """
    import nibabel as nib

    # Two-tet mesh sharing a face at z=0
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [0, 0, 1], [0, 0, -1]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3,  4, 0, 1, 2, 4])
    cell_types = np.array([pv.CellType.TETRA, pv.CellType.TETRA])
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    # tet 0 centroid ≈ (0.25, 0.25, 0.25) → CSF; tet 1 centroid ≈ (0.25, 0.25, -0.25) → WM
    mesh.cell_data["marker"] = np.array([Label.CSF, Label.LEFT_CEREBRAL_WHITE_MATTER],
                                        dtype=np.int32)

    # NIfTI: 4×4×4 voxels, 1 mm isotropic, origin at (0,0,0)
    # Voxel (0,0,0) covers [0,1)³ — centroid (0.25,0.25,0.25) lands here → label 1001
    # Voxel (0,0,0) in the negative-z half covers (0.25,0.25,-0.25) → label 0 (background)
    seg_data = np.zeros((4, 4, 4), dtype=np.int32)
    seg_data[0, 0, 0] = 1001   # positive-z half: SAS parcel
    affine = np.eye(4)          # 1 mm voxels, no offset
    sas_img = nib.Nifti1Image(seg_data, affine)

    remark_csf_with_sas(mesh, sas_img)

    from brainmesh.labels import SAS_LABEL_OFFSET
    markers = mesh.cell_data["marker"]
    # CSF tet → remapped to 1001 + SAS_LABEL_OFFSET
    assert markers[0] == 1001 + SAS_LABEL_OFFSET
    # WM tet → unchanged
    assert markers[1] == Label.LEFT_CEREBRAL_WHITE_MATTER


def test_remark_csf_with_sas_zero_voxel_fallback():
    """An all-zero SAS NIfTI (no labeled voxels) must leave all markers unchanged."""
    import nibabel as nib

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3])
    cell_types = np.array([pv.CellType.TETRA])
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.cell_data["marker"] = np.array([Label.CSF], dtype=np.int32)

    # All-zero NIfTI → centroid maps to background
    sas_img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int32), np.eye(4))
    remark_csf_with_sas(mesh, sas_img)

    assert mesh.cell_data["marker"][0] == Label.CSF


@pytest.mark.slow
def test_facet_face_count_conservation(split_box_mesh):
    """4·n_tets == 2·n_interior + n_boundary; n_interface ≤ n_interior."""
    interfaces = mark_interface_facets(split_box_mesh)
    boundaries = mark_boundary_facets(split_box_mesh)

    n_total_face_uses = 4 * split_box_mesh.n_cells
    assert (n_total_face_uses - boundaries.n_cells) % 2 == 0
    n_interior = (n_total_face_uses - boundaries.n_cells) // 2
    assert n_interior >= interfaces.n_cells
