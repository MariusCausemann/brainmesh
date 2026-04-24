"""Tetrahedral mesh operations: marking, filtering, and facet extraction."""
import numpy as np
import igl
import pyvista as pv

from .decorators import time_func
from .labels import Label, VENTRICLE_LABELS


@time_func
def mark_mesh(mesh, surf):
    """
    Mark each tetrahedron with its anatomical label using the winding number of the input surfaces.

    Parameters:
        mesh: tetrahedralized pyvista.UnstructuredGrid
        surf: multi-label surface mesh with 'boundary_labels' cell data
    """
    labels = np.unique(surf["boundary_labels"])
    marker = np.zeros(mesh.n_cells, dtype=np.int32)
    query_points = np.array(mesh.cell_centers().points)
    F_global = np.array(surf.faces.reshape(-1, 4)[:, 1:])
    V_global = np.array(surf.points)
    blabels = surf.cell_data["boundary_labels"]

    for i, cid in enumerate(labels):
        if i == 0:
            continue

        mask_out = blabels[:, 0] == cid
        F_out = F_global[mask_out]

        mask_in = blabels[:, 1] == cid
        F_in = F_global[mask_in]

        if len(F_in) > 0:
            F_in_flipped = F_in[:, [0, 2, 1]]
            F_label = np.vstack((F_out, F_in_flipped))
        else:
            F_label = F_out

        fwn = igl.fast_winding_number(V_global, F_label, query_points)
        marker = np.where(marker == 0, cid * np.isclose(fwn, -1, rtol=0.5), marker)

    mesh["marker"] = marker
    mesh = mesh.extract_cells(marker > 0)
    return mesh


def load_marked_mesh(path, label_array="marker"):
    """Load a tetrahedral mesh produced by :func:`mark_mesh` from disk."""
    mesh = pv.read(path)
    if label_array not in mesh.cell_data:
        raise ValueError(
            f"{path} has no '{label_array}' cell-data array — "
            f"was it produced by mark_mesh?"
        )
    return mesh


def filter_by_label(mesh, labels, label_array="marker"):
    """
    Extract the subset of cells whose ``label_array`` value is in ``labels``.

    The returned UnstructuredGrid has its own (reduced) point array — use
    :func:`mark_interface_facets` / :func:`mark_boundary_facets` afterwards
    if you need a facet mesh referencing those points.
    """
    labels = np.atleast_1d(labels)
    mask = np.isin(np.asarray(mesh.cell_data[label_array]), labels)
    return mesh.extract_cells(mask)


def extract_csf(mesh, label_array="marker"):
    """
    Extract the CSF compartment — CSF plus all ventricles and choroid
    plexus (i.e. ``Label.CSF`` plus ``VENTRICLE_LABELS``).
    """
    csf_labels = list(VENTRICLE_LABELS) + [Label.CSF]
    return filter_by_label(mesh, csf_labels, label_array=label_array)


def _tet_face_topology(mesh):
    """
    Group all triangular faces of a tet mesh by their canonical (sorted)
    vertex triple, so adjacent tets sharing a face can be matched.

    Returns a dict with arrays:
      boundary_faces      (M, 3)  outer-surface triangles (one parent tet)
      boundary_parents    (M,)    parent tet index for each boundary face
      interface_faces     (N, 3)  triangles shared by two tets
      interface_parents_a (N,)    parent tet index for one side
      interface_parents_b (N,)    parent tet index for the other side

    Triangle vertex indices are returned in the order they appear in the
    parent tet (not sorted), preserving the original outward orientation.
    Assumes a manifold tet mesh (each face shared by ≤ 2 tets).
    """
    cells = mesh.cells_dict.get(pv.CellType.TETRA)
    if cells is None or len(cells) == 0:
        raise ValueError("Input mesh has no tetrahedral cells.")
    n_tets = len(cells)

    face_indices = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    faces = cells[:, face_indices].reshape(-1, 3)
    parents = np.repeat(np.arange(n_tets), 4)
    canonical = np.sort(faces, axis=1)

    order = np.lexsort(canonical.T[::-1])
    canonical = canonical[order]
    faces = faces[order]
    parents = parents[order]

    same_as_next = np.zeros(len(canonical), dtype=bool)
    same_as_next[:-1] = np.all(canonical[:-1] == canonical[1:], axis=1)
    same_as_prev = np.zeros(len(canonical), dtype=bool)
    same_as_prev[1:] = same_as_next[:-1]

    is_unique = ~same_as_next & ~same_as_prev
    pair_idx = np.where(same_as_next)[0]

    return {
        "boundary_faces": faces[is_unique],
        "boundary_parents": parents[is_unique],
        "interface_faces": faces[pair_idx],
        "interface_parents_a": parents[pair_idx],
        "interface_parents_b": parents[pair_idx + 1],
    }


def _build_facet_polydata(mesh, faces, scalars):
    n = len(faces)
    if n == 0:
        cells = np.empty(0, dtype=np.int64)
    else:
        cells = np.column_stack([np.full(n, 3, dtype=np.int64), faces]).ravel()
    poly = pv.PolyData(np.asarray(mesh.points), faces=cells)
    for name, arr in scalars.items():
        poly.cell_data[name] = np.asarray(arr)
    return poly


def mark_interface_facets(mesh, label_array="marker", encoding_base=1000):
    """
    Build a facet mesh of the interfaces between regions with different
    markers in the input tet mesh.

    Each triangle is labelled with a unique ``interface_id`` encoded as
    ``min(a, b) * encoding_base + max(a, b)`` for the two adjacent
    region markers, and the original markers are also kept on
    ``region_a`` (lower) and ``region_b`` (higher).

    The resulting :class:`pyvista.PolyData` shares the parent's point
    array, so vertex indices line up with the parent tet mesh — ready to
    use as a FEniCS facet function.
    """
    topo = _tet_face_topology(mesh)
    markers = np.asarray(mesh.cell_data[label_array])

    faces = topo["interface_faces"]
    m_a = markers[topo["interface_parents_a"]]
    m_b = markers[topo["interface_parents_b"]]

    diff = m_a != m_b
    faces = faces[diff]
    m_a = m_a[diff]
    m_b = m_b[diff]

    lo = np.minimum(m_a, m_b).astype(np.int64)
    hi = np.maximum(m_a, m_b).astype(np.int64)
    interface_id = lo * encoding_base + hi

    return _build_facet_polydata(
        mesh,
        faces,
        {"interface_id": interface_id, "region_a": lo, "region_b": hi},
    )


def mark_boundary_facets(mesh, label_array="marker"):
    """
    Build a facet mesh of all outer boundary facets of the input tet mesh,
    each labelled by the marker of its single adjacent region.

    The resulting :class:`pyvista.PolyData` shares the parent's point
    array, so vertex indices line up with the parent tet mesh.
    """
    topo = _tet_face_topology(mesh)
    markers = np.asarray(mesh.cell_data[label_array])

    faces = topo["boundary_faces"]
    boundary = markers[topo["boundary_parents"]].astype(np.int64)

    return _build_facet_polydata(mesh, faces, {"boundary": boundary})
