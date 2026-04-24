"""Integration test: full sphere pipeline (surface extraction + tetrahedralisation).

Mark with @pytest.mark.slow — excluded from fast CI runs.
Run manually: pytest -m slow
"""
import numpy as np
import pytest
import pyvista as pv


@pytest.mark.slow
def test_sphere_surface_and_mesh():
    import nbmorph
    import pytetwild
    from brainmesh import mark_mesh

    N, r = 20, 2
    data = np.ones((N, N, N), dtype=np.uint8)
    grid = pv.ImageData(dimensions=(N + 1, N + 1, N + 1), spacing=(1.0 / N,) * 3)
    pts = grid.cell_centers().points.reshape(N, N, N, 3)
    dist = np.linalg.norm(pts - 0.5, axis=-1)
    data[dist < 0.3] = 2
    data[dist < 0.22] = 3

    grid["data"] = data.flatten(order="F")
    grid.resample(interpolation="nearest", sample_rate=r, inplace=True)
    data = grid["data"].reshape(r * N, r * N, r * N)
    data = nbmorph.erode_labels_spherical(data, radius=1, struct_sequence="D")
    data = nbmorph.dilate_labels_spherical(data, radius=1, struct_sequence="B")
    data = nbmorph.dilate_labels_spherical(data, radius=1)

    grid["data"] = data.flatten(order="F")
    surf = grid.contour_labels("all", smoothing=True)

    assert surf.n_cells > 0
    assert "boundary_labels" in surf.cell_data

    mesh = pytetwild.pytetwild.tetrahedralize_pv(
        surf,
        stop_energy=10,
        loglevel=6,
        quiet=True,
        disable_filtering=True,
        edge_length_fac=0.05,
        epsilon=1e-3,
        coarsen=False,
        num_threads=2,
    )
    assert mesh.n_cells > 0

    marked = mark_mesh(mesh, surf)
    assert "marker" in marked.cell_data
    assert marked.n_cells > 0
