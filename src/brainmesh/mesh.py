"""Tetrahedral mesh operations: marking, filtering, and facet extraction."""
import numpy as np
import igl
import pyvista as pv

from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .decorators import time_func
from .io import get_img
from .labels import (Label, VENTRICLE_LABELS, SAS_LABEL_OFFSET, SPINAL_ID,
                     is_csf_marker)


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
        # |fwn| ≈ 1 inside, ≈ 0 outside; abs covers either orientation convention
        marker = np.where(marker == 0, cid * (np.abs(fwn) > 0.5), marker)

    mesh.cell_data["marker"] = marker
    mesh = mesh.extract_cells(marker > 0)
    return mesh

def smooth_cell_labels(surf, marker_name, target_labels, max_iter=20):
    """
    Smooths cell labels by enforcing that no triangle shares more than 
    one edge with a different region, acting ONLY on the specified labels.
    """
    mesh = surf.copy()
    labels = mesh.cell_data[marker_name].copy()
    n_cells = mesh.n_cells
    
    # Convert to set for O(1) lookup speed inside the loop
    target_set = set(target_labels)
    
    # Pre-compute the edge neighbors
    neighbors_list = [mesh.cell_neighbors(i, 'edges') for i in range(n_cells)]
    
    for iteration in range(max_iter):
        flipped = 0
        
        for i in range(n_cells):
            current_label = labels[i]
            
            # Constraint 1: Skip if the current cell isn't a target label
            if current_label not in target_set:
                continue
                
            neighbor_ids = neighbors_list[i]
            if len(neighbor_ids) == 0:
                continue 
                
            neighbor_labels = labels[neighbor_ids]
            unique_vals, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = unique_vals[np.argmax(counts)]
            max_count = np.max(counts)
            
            # Constraint 2: Only allow flipping TO another target label
            if majority_label not in target_set:
                continue
            
            # The Magic Rule
            if majority_label != current_label and max_count >= 2:
                labels[i] = majority_label
                flipped += 1
                
        if flipped == 0:
            print(f"Selective smoothing converged in {iteration + 1} iterations.")
            break
            
    mesh.cell_data[marker_name] = labels
    return mesh

def remark_csf_with_sas(mesh, sas_img, csf_label=Label.CSF, label_array="marker"):
    """
    Replace ``csf_label`` markers on ``mesh`` with subdivided SAS labels sampled
    from ``sas_img`` (path or nib.Nifti1Image) at each tet centroid via the
    image's inverse affine.

    Tets that map outside the image or to a background (0) voxel keep their
    original CSF marker. Non-CSF tets are never touched.
    Edits ``mesh.cell_data[label_array]`` in place and returns ``mesh``.
    """
    from scipy.spatial import KDTree
    from .io import nibabel_to_pyvista

    sas = nibabel_to_pyvista(get_img(sas_img))
    sas = sas.extract_cells(sas.cell_data["data"] > 0)
    if sas.n_cells == 0:
        return mesh
    kd_tree = KDTree(sas.cell_centers().points)
    mesh_cell_centers = mesh.cell_centers().points.astype(np.float32)
    _, nearest_idx = kd_tree.query(mesh_cell_centers)
    sas_marker = sas.cell_data["data"].astype(np.int32)
    mesh_marker = mesh.cell_data[label_array].copy()
    csf_mask = mesh_marker == csf_label
    mesh_marker[csf_mask] = sas_marker[nearest_idx][csf_mask] + SAS_LABEL_OFFSET
    mesh.cell_data[label_array] = mesh_marker
    return mesh


def load_marked_mesh(path, label_array="marker"):
    """Load a tetrahedral mesh produced by :func:`mark_mesh` from disk."""
    from .io import read_mesh
    mesh = read_mesh(path)
    if label_array not in mesh.cell_data:
        raise ValueError(
            f"{path} has no '{label_array}' cell-data array — "
            f"was it produced by mark_mesh?"
        )
    return mesh


def filter_by_mask(mesh, mask):
    """
    Extract the subset of cells with mask,
    while strictly preserving the original point array.
    """
    # Check if the mesh is homogeneous (every cell takes up the same flat length)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    if len(mesh.cells) % mesh.n_cells != 0:
        raise ValueError("Mesh contains mixed cell types. Cannot use 2D reshape method.")
    
    # Number of integers each cell takes up (e.g., 5 for linear tets: [4, p1, p2, p3, p4])
    stride = len(mesh.cells) // mesh.n_cells
    
    # Reshape, apply the mask to the rows, and flatten back to 1D
    cells_2d = mesh.cells.reshape(mesh.n_cells, stride)
    new_cells = cells_2d[mask].flatten()
    
    new_celltypes = mesh.celltypes[mask]
    
    # Rebuild the grid forcing it to use the full, unpruned points array
    retained_mesh = pv.UnstructuredGrid(new_cells, new_celltypes, mesh.points)
    
    # Transfer the cell data over
    for name in mesh.cell_data:
        retained_mesh.cell_data[name] = mesh.cell_data[name][mask]
    
    for name in mesh.point_data:
        retained_mesh.point_data[name] = mesh.point_data[name]
        
    return retained_mesh


# treat linear + quadratic tets as "volume", linear + quadratic tris as "surface"
VOL_TYPES = (pv.CellType.TETRA, pv.CellType.QUADRATIC_TETRA)
TRI_TYPES = (pv.CellType.TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE)

def _gather(mesh, types, n_corner):
    """(corners (m, n_corner), global_ids (m,)) for the given cell types."""
    cdict, ct = mesh.cells_dict, mesh.celltypes
    corner_blocks, id_blocks = [], []
    for t in types:
        if t in cdict:
            gids = np.where(ct == t)[0]          # ascending, aligns with cdict[t]
            corner_blocks.append(cdict[t][:, :n_corner])
            id_blocks.append(gids)
    if not corner_blocks:
        return np.empty((0, n_corner), int), np.empty(0, int)
    return np.vstack(corner_blocks), np.concatenate(id_blocks)

def _tet_volumes(points, corners):
    p = points[corners]                          # (m, 4, 3)
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0

def largest_face_connected(mesh, by_volume=True, keep_surface_tris=True):
    tet_corners, tet_gids = _gather(mesh, VOL_TYPES, 4)
    n = len(tet_corners)
    if n == 0:
        raise ValueError("no tetrahedral cells found")

    combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_to_local = defaultdict(list)            # corner-face -> local tet indices
    for li, tet in enumerate(tet_corners):
        for a, b, c in combos:
            face_to_local[tuple(sorted((tet[a], tet[b], tet[c])))].append(li)

    rows, cols = [], []
    for cs in face_to_local.values():
        if len(cs) == 2:                         # internal face shared by 2 tets
            x, y = cs
            rows += [x, y]; cols += [y, x]

    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)

    if by_volume:
        weights = np.bincount(labels, weights=_tet_volumes(mesh.points, tet_corners))
    else:
        weights = np.bincount(labels)
    win = weights.argmax()

    keep_local = np.where(labels == win)[0]
    keep_global = tet_gids[keep_local].tolist()

    if keep_surface_tris:
        tri_corners, tri_gids = _gather(mesh, TRI_TYPES, 3)
        if len(tri_gids):
            kept_faces = set()
            for li in keep_local:
                tet = tet_corners[li]
                for a, b, c in combos:
                    kept_faces.add(tuple(sorted((tet[a], tet[b], tet[c]))))
            for ti, tri in zip(tri_gids, tri_corners):
                if tuple(sorted(tri)) in kept_faces:
                    keep_global.append(int(ti))

    return mesh.extract_cells(keep_global)

def extract_reduced_facets(reduced_tets, full_facets):
    """Facets of `full_facets` bounding `reduced_tets`, rebuilt on
    `reduced_tets.points` with its exact node ordering. Linear or quadratic.
    Assumes both objects share point_data['gid'] from the full mesh."""
    TET = (pv.CellType.TETRA, pv.CellType.QUADRATIC_TETRA)
    FACE_IDX = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    def keys(rows):
        a = np.ascontiguousarray(np.sort(rows, axis=1))
        return a.view([('', a.dtype)] * a.shape[1]).ravel()

    # facet connectivity (single triangle type) + point->global map
    # (linear facet meshes are PolyData and have no cells_dict, hence _raw_faces)
    conn = _raw_faces(full_facets)                             # (n, 3 or 6)
    ttype = (pv.CellType.TRIANGLE if conn.shape[1] == 3
             else pv.CellType.QUADRATIC_TRIANGLE)
    fg = (np.asarray(full_facets.point_data['gid']) if 'gid' in full_facets.point_data
          else np.arange(full_facets.n_points))               # facet-pt -> global

    # corner faces of the reduced tets, as global ids
    csf_gid = np.asarray(reduced_tets.point_data['gid']).astype(np.uint32)       # local -> global
    tet_corners = np.vstack([reduced_tets.cells_dict[t][:, :4]
                             for t in TET if t in reduced_tets.cells_dict])
    csf_faces = csf_gid[tet_corners][:, FACE_IDX].reshape(-1, 3)

    keep = np.isin(keys(fg[conn[:, :3]]), keys(csf_faces))

    # remap kept facets straight to reduced_tets-local numbering
    g2l = np.full(int(csf_gid.max()) + 1, -1, np.int64)
    g2l[csf_gid] = np.arange(len(csf_gid), dtype=np.uint32)
    local = g2l[fg[conn[keep]]]
    assert (local >= 0).all(), "kept facet node missing from reduced_tets (gid mismatch)"

    out = pv.UnstructuredGrid({ttype: local}, reduced_tets.points)
    out.point_data.update(reduced_tets.point_data)            # shares csf_mesh point arrays
    for k in full_facets.cell_data:
        out.cell_data[k] = np.asarray(full_facets.cell_data[k])[keep]
    return out

def extract_csf(mesh, label_array="marker", return_facets=False, **facet_kwargs):
    """
    Extract the CSF compartment — ``Label.CSF``, all ventricles and choroid plexus,
    and any SAS-subdivision markers (values > ``SAS_LABEL_OFFSET``); see
    :func:`brainmesh.labels.is_csf_marker`.

    ``Label.UNCLASSIFIED`` (vessels sitting in the SAS) and ``Label.SPINAL_BUFFER``
    are *not* part of it: they stay solid, and their facets against the CSF become
    boundaries of the extracted submesh (the buffer ones as ``SPINAL_ID``).

    When ``return_facets=True``, facets are computed on the **full** mesh so that
    CSF-to-tissue interfaces carry their full ``interface_id`` encoding
    (e.g. ``min(CSF,WM)*100000+max(CSF,WM)``).  The returned facet mesh shares
    the CSF submesh's point array.

    Parameters
    ----------
    mesh          : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array   : str                  cell data array with region markers
    return_facets : bool                 if True, also return the CSF-relevant
                                         facets (same ``interface_id`` scheme as
                                         :func:`mark_facets`)
    **facet_kwargs                       forwarded to :func:`mark_facets`
                                         (e.g. ``encoding_base``)

    Returns
    -------
    csf_mesh : pv.UnstructuredGrid
    facets   : pv.PolyData or pv.UnstructuredGrid  (only when return_facets=True)
    """
    csf_cells = is_csf_marker(mesh.cell_data[label_array])

    if not return_facets:
        csf_mesh = filter_by_mask(mesh, csf_cells)
        return largest_face_connected(csf_mesh.clean())

    mesh["gid"] = np.arange(mesh.n_points)
    csf_mesh = largest_face_connected(filter_by_mask(mesh, csf_cells)).clean()
    assert "gid" in csf_mesh.array_names
    # Compute facets on the full mesh (preserves CSF-to-tissue interface IDs),
    full_facets = mark_facets(mesh, label_array=label_array, **facet_kwargs)

    csf_facets = extract_reduced_facets(csf_mesh, full_facets)

    assert np.allclose(csf_mesh.points, csf_facets.points)
    assert label_array in csf_mesh.array_names
    assert "interface_id" in csf_facets.array_names
    return csf_mesh, csf_facets


def mark_between_regions(ids_a, ids_b, da, db, seg, label_array):
    from scipy.spatial import KDTree
    sega = seg.extract_cells(np.isin(seg.cell_data[label_array], ids_a))
    segb = seg.extract_cells(np.isin(seg.cell_data[label_array], ids_b))
    kd_tree_a = KDTree(sega.cell_centers().points)
    kd_tree_b = KDTree(segb.cell_centers().points)
    dista, _ = kd_tree_a.query(seg.cell_centers().points)
    distb, _ = kd_tree_b.query(seg.cell_centers().points)
    return np.logical_and(dista < da, distb < db)

def dilate_cell_marker(grid, marker, r=1):
    dil_marker = marker.copy()
    for _ in range(r):
        dil_marker = dilate_cell_marker_once(grid, dil_marker)
    return dil_marker

def dilate_cell_marker_once(grid, marker):
    all_cell_centers = grid.cell_centers().points
    dil_marker = marker.copy()
    for i in np.nonzero(marker==0)[0]:
        neighbor_indices = grid.cell_neighbors(i,connections="points")
        neighbor_indices = [j for j in neighbor_indices if marker[j] > 0]
        if len(neighbor_indices)==0: continue
        neighbor_markers = marker[neighbor_indices]
        neighbor_centers = all_cell_centers[neighbor_indices]
        neighbor_dist = np.linalg.norm(all_cell_centers[i] - neighbor_centers, axis=1)
        # possibly weight by distance?
        counts = np.bincount(neighbor_markers, weights=1/neighbor_dist)
        dil_marker[i] = np.argmax(counts)
    return dil_marker

def erode_cell_marker(grid, marker, r=1):
    eroded_marker = marker.copy()
    for _ in range(r):
        new_marker = eroded_marker.copy()
        for i in range(grid.n_cells):
            neighbor_indices = grid.cell_neighbors(i, connections="points")
            if (eroded_marker[i] != eroded_marker[neighbor_indices]).any(): 
                new_marker[i] = 0
        eroded_marker = new_marker
    return eroded_marker

def remove_small_patches(grid, facet_marker, threshold, target_labels=None):
    new_marker = facet_marker.copy()
    if target_labels is None: target_labels = np.unique(facet_marker)
    for fm in target_labels:
        patch = grid.extract_cells(facet_marker==fm).connectivity()
        regionids, counts  = np.unique(patch["RegionId"], return_counts=True)
        for ri, c in zip(regionids, counts):
            if c < threshold: 
                new_marker[np.flatnonzero(facet_marker==fm)[patch["RegionId"]==ri]] = 0
    return new_marker



CSF_REGION_NAMES = []

def group_csf_facets_by_region(facets, encoding_base=100000):
    """
    Assign each CSF facet (from :func:`extract_csf`) to a named anatomical region.

    Decodes the ``interface_id`` cell array and maps each facet to one of the
    regions in :data:`CSF_REGION_NAMES`.  Adds a ``region`` (int32) cell array
    to a copy of ``facets``.  Every facet receives a non-zero region label.

    Region IDs and names
    --------------------

    Parameters
    ----------
    facets           : pv mesh with ``interface_id`` cell array
    encoding_base    : int    encoding base used in ``interface_id`` (default 100000)
    Returns
    -------
    pv mesh  copy of ``facets`` with added ``region`` (int32) cell array
    """
    from .labels import region_dict as sas_region_dict
    from .labels import _sas_lh, _sas_rh

    region_label_dict = {"SPINAL_CSF": 1, "PIA": 3, "LATERAL_VENTRICLES":2,
                         "FALX":4, "TENTORIUM_UPPER":5, "TENTORIUM_LOWER":6,
                         "UNCLASSIFIED":7,
                         "ANTERIOR_PARASAGITTAL_SINUS":8,"POSTERIOR_PARASAGITTAL_SINUS":9}

    ids = np.asarray(facets.cell_data["interface_id"], dtype=np.int64)
    a, b = np.divmod(ids, encoding_base)
    
    region = np.zeros(len(ids), dtype=np.int32)

    def _assign(mask, rid):
        region[mask & (region == 0)] = rid

    tent = int(Label.TENTORIUM)  # 71

    remove_id = int(-1)
    # remove all interfaces between different CSF regions
    _assign(np.logical_and(is_csf_marker(a), is_csf_marker(b)), remove_id)

    # 1. Spinal canal
    _assign(ids == SPINAL_ID, region_label_dict["SPINAL_CSF"])

    # mark lateral ventricle surface
    LVs = list(set(VENTRICLE_LABELS) - set((Label.THIRD_VENTRICLE,
                                            Label.FOURTH_VENTRICLE)))
    _assign(np.logical_xor(np.isin(a, LVs), np.isin(b, LVs)), region_label_dict["LATERAL_VENTRICLES"])

    # mark FALX
    _assign((a==Label.FALX) + (b==Label.FALX), region_label_dict["FALX"])

    # mark up and downward facing parts of tentorium
    _assign((a==tent) + (b==tent), region_label_dict["TENTORIUM_UPPER"])
    infra_tent_ids = list(sas_region_dict["INFRATENTORIAL"])
    region[np.logical_and(a==tent, np.isin(b, infra_tent_ids))] = region_label_dict["TENTORIUM_LOWER"]
    region[np.logical_and(b==tent, np.isin(a, infra_tent_ids))] = region_label_dict["TENTORIUM_LOWER"]

    # vessels and other unclassified material sitting in the SAS
    _assign((a == Label.UNCLASSIFIED) | (b == Label.UNCLASSIFIED),
            region_label_dict["UNCLASSIFIED"])

    # all remaining internal -> tissue
    _assign(ids >= encoding_base, region_label_dict["PIA"])

    # mark specified regions:
    for i, (k, v) in enumerate(sas_region_dict.items()):
        _assign(np.isin(ids, list(v)), 10 + i)
        region_label_dict[k] = 10 + i

    # mark sagittal sinus
    # find right and left SAS labels
    rs_sas_labels = np.unique(ids[(ids > SAS_LABEL_OFFSET + 2000) & 
                                  (ids < SAS_LABEL_OFFSET + 3000)]).tolist()
    ls_sas_labels = np.unique(ids[(ids > SAS_LABEL_OFFSET + 1000) & 
                                  (ids < SAS_LABEL_OFFSET + 2000)]).tolist()

    # and find all facets that are within 10mm of both
    PSD = mark_between_regions(rs_sas_labels, ls_sas_labels, 10, 10, 
                               facets, "interface_id")

    # finally mark the front and back, depending on the SAS label IDs
    anterior_PSD = [1017,1022,1024, 1028]
    posterior_PSD = [1025, 1029, 1005, 1011, 1013]
    region[np.logical_and(PSD, np.isin(ids, _sas_rh(anterior_PSD) + _sas_lh(anterior_PSD)))] = region_label_dict["ANTERIOR_PARASAGITTAL_SINUS"]
    region[np.logical_and(PSD, np.isin(ids, _sas_rh(posterior_PSD) + _sas_lh(posterior_PSD)))] = region_label_dict["POSTERIOR_PARASAGITTAL_SINUS"]
    
    sas_regions = np.unique(region[region >= 10]).tolist()
    out = facets.copy()
    region = remove_small_patches(out, region, threshold=50, target_labels=sas_regions)
    region = dilate_cell_marker(out, region, 10)
    out.cell_data["region"] = region
    out = smooth_cell_labels(out, marker_name="region", target_labels=sas_regions + [region_label_dict["ANTERIOR_PARASAGITTAL_SINUS"],
                                                                                      region_label_dict["POSTERIOR_PARASAGITTAL_SINUS"]])
    out = filter_by_mask(out, out.cell_data["region"] >= 0)
    assert (region==0).sum() == 0
    out.field_data["region_names"] = region_label_dict
    return out, region_label_dict


def _tet_face_topology(mesh):
    """
    Group all triangular (or quadratic triangular) faces of a tet mesh by their 
    canonical (sorted) corner vertex triple, so adjacent tets sharing a face can be matched.

    Returns a dict with arrays:
      boundary_faces      (M, 3 or 6) outer-surface triangles (one parent tet)
      boundary_parents    (M,)        parent tet index for each boundary face
      interface_faces     (N, 3 or 6) triangles shared by two tets
      interface_parents_a (N,)        parent tet index for one side
      interface_parents_b (N,)        parent tet index for the other side
    """
    # Check for linear or quadratic tets
    cells_lin = mesh.cells_dict.get(pv.CellType.TETRA)
    cells_quad = mesh.cells_dict.get(pv.CellType.QUADRATIC_TETRA)

    if cells_quad is not None and len(cells_quad) > 0:
        cells = cells_quad
        # VTK Quadratic Tet Ordering: 
        # Corners: 0,1,2,3. Edges: 4(0,1), 5(1,2), 6(2,0), 7(0,3), 8(1,3), 9(2,3)
        # We extract 6-node faces: [corner1, corner2, corner3, edge1, edge2, edge3]
        face_indices = np.array([
            [1, 2, 3, 5, 9, 8],
            [0, 2, 3, 6, 9, 7],
            [0, 1, 3, 4, 8, 7],
            [0, 1, 2, 4, 5, 6]
        ])
        nodes_per_face = 6
    elif cells_lin is not None and len(cells_lin) > 0:
        cells = cells_lin
        face_indices = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        nodes_per_face = 3
    else:
        raise ValueError("Input mesh has no tetrahedral or quadratic tetrahedral cells.")

    n_tets = len(cells)
    faces = cells[:, face_indices].reshape(-1, nodes_per_face)
    parents = np.repeat(np.arange(n_tets), 4)
    
    # Extract just the 3 corner nodes to sort and match. 
    # This ensures exact matching without relying on mid-node permutations.
    corners = faces[:, :3]
    canonical = np.sort(corners, axis=1)

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


def mark_interface_facets(mesh, label_array="marker", encoding_base=1000,
                          ignore_sas_interfaces=True):
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

    Parameters
    ----------
    ignore_sas_interfaces : bool
        If True (default), drop interfaces where *both* adjacent markers are
        SAS subdivision labels (> ``SAS_LABEL_OFFSET``).  These intra-SAS
        boundaries are rarely needed and can be re-enabled by passing False.
    """
    topo = _tet_face_topology(mesh)
    markers = np.asarray(mesh.cell_data[label_array])
    points = np.asarray(mesh.points)
    tet_centroids = np.asarray(mesh.cell_centers().points)

    faces = topo["interface_faces"]
    parents_a = topo["interface_parents_a"]
    parents_b = topo["interface_parents_b"]
    m_a = markers[parents_a]
    m_b = markers[parents_b]

    diff = m_a != m_b
    faces = faces[diff]
    parents_a = parents_a[diff]
    parents_b = parents_b[diff]
    m_a = m_a[diff]
    m_b = m_b[diff]

    lo = np.minimum(m_a, m_b).astype(np.int64)
    hi = np.maximum(m_a, m_b).astype(np.int64)

    if ignore_sas_interfaces:
        keep = lo <= SAS_LABEL_OFFSET
        faces = faces[keep]
        parents_a = parents_a[keep]
        parents_b = parents_b[keep]
        m_a = m_a[keep]
        m_b = m_b[keep]
        lo = lo[keep]
        hi = hi[keep]

    # Orient face winding so each normal points from the lo-marker region
    # toward the hi-marker region. The raw winding from _tet_face_topology
    # depends on which adjacent tet happened to be sorted first, so we
    # explicitly recompute and flip — same idea as mark_mesh flipping faces
    # whose label is on the "in" side.
    if len(faces) > 0:
        corners = points[faces[:, :3]]
        centroids = corners.mean(axis=1)
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 2] - corners[:, 0]
        normals = np.cross(e1, e2)

        lo_parents = np.where(m_a == lo, parents_a, parents_b)
        lo_centroids = tet_centroids[lo_parents]
        flip = np.einsum("ij,ij->i", normals, centroids - lo_centroids) < 0

        _flip_winding(faces, flip)

    interface_id = lo * encoding_base + hi

    return _build_facet_polydata(
        mesh,
        faces,
        {"interface_id": interface_id, "region_a": lo, "region_b": hi},
    )

def _build_facet_polydata(mesh, faces, scalars):
    """
    Builds a surface mesh containing the extracted interface facets.
    Dynamically creates a PolyData for linear faces (3 nodes) or an 
    UnstructuredGrid for quadratic faces (6 nodes).
    """
    n = len(faces)
    points = np.asarray(mesh.points)

    if n == 0:
        grid = pv.PolyData()
        grid.points = points
    else:
        nodes_per_face = faces.shape[1]
        
        # VTK connectivity format: [n_nodes, p0, p1..., n_nodes, p0, p1...]
        cells = np.column_stack([np.full(n, nodes_per_face, dtype=np.int64), faces]).ravel()

        if nodes_per_face == 3:
            # Linear triangles are perfectly handled by PolyData
            grid = pv.PolyData(points, faces=cells)
            
        elif nodes_per_face == 6:
            # Quadratic triangles MUST be an UnstructuredGrid
            cell_types = np.full(n, pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cells, cell_types, points)
            
        else:
            raise ValueError(f"Expected 3 or 6 nodes per face, got {nodes_per_face}.")

    # Attach the scalar data (interface IDs, markers, etc.)
    for name, arr in scalars.items():
        grid.cell_data[name] = np.asarray(arr)

    return grid

def _raw_faces(facet_mesh):
    """Raw connectivity of a facet mesh: (N, 3) for linear, (N, 6) for quadratic."""
    if isinstance(facet_mesh, pv.PolyData):
        return facet_mesh.faces.reshape(-1, 4)[:, 1:]
    cells = facet_mesh.cells_dict
    for ttype in (pv.CellType.QUADRATIC_TRIANGLE, pv.CellType.TRIANGLE):
        if ttype in cells:
            return cells[ttype]
    return np.empty((0, 6), dtype=np.int64)


def _flip_winding(faces, flip):
    """Reverse the winding of the rows selected by ``flip`` (in place)."""
    if faces.shape[1] == 3:
        faces[flip] = faces[flip][:, [0, 2, 1]]
    else:  # quadratic triangle: swap corners 1<->2 and their opposite edges
        faces[flip] = faces[flip][:, [0, 2, 1, 5, 4, 3]]
    return faces


def spinal_interface_mask(region_a, region_b):
    """
    True where one side of a facet is ``Label.SPINAL_BUFFER`` and the other belongs
    to the CSF compartment (see :func:`brainmesh.labels.is_csf_marker`).

    ``SPINAL_BUFFER`` is the flat slab the segmentation pipeline extrudes below the
    bottom of the image (:func:`brainmesh.anatomy.extend_brainstem_caudally`), so
    this interface *is* the spinal opening — no geometric normal/height test needed.
    Buffer facets against the brainstem or against ``UNCLASSIFIED`` are not spinal
    and keep their ordinary interface id.
    """
    a = np.asarray(region_a)
    b = np.asarray(region_b)
    buf = Label.SPINAL_BUFFER
    return ((a == buf) & is_csf_marker(b)) | ((b == buf) & is_csf_marker(a))


def mark_spinal_boundary(mesh, label_array="marker", encoding_base=100000):
    """
    Mark the spinal interface: the facets between ``Label.SPINAL_BUFFER`` tets and
    the CSF compartment.

    Face winding is oriented so each normal points *out of* the CSF region, i.e.
    down into the buffer — the same outward convention the surrounding CSF boundary
    facets use.

    Parameters
    ----------
    mesh          : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array   : str                  cell data array with region markers
    encoding_base : int                  encoding base used while decoding the
                                         interface ids (default 100000)

    Returns a :class:`pyvista.PolyData` or :class:`pyvista.UnstructuredGrid` that
    shares the parent's point array, with a ``boundary`` cell data array holding the
    marker of the CSF-side tet.
    """
    interfaces = mark_interface_facets(mesh, label_array=label_array,
                                       encoding_base=encoding_base)
    lo = np.asarray(interfaces.cell_data["region_a"])
    hi = np.asarray(interfaces.cell_data["region_b"])

    keep = spinal_interface_mask(lo, hi)
    faces = np.array(_raw_faces(interfaces)[keep])
    lo, hi = lo[keep], hi[keep]

    # mark_interface_facets winds lo -> hi, so facets where the buffer is the lower
    # marker (against a SAS parcel) point the wrong way and need flipping.
    _flip_winding(faces, lo == Label.SPINAL_BUFFER)
    csf_side = np.where(lo == Label.SPINAL_BUFFER, hi, lo)

    return _build_facet_polydata(mesh, faces, {"boundary": csf_side})

def mark_facets(mesh, label_array="marker", encoding_base=100000,
                smooth_sas_labels=False, ignore_sas_interfaces=True):
    """
    Build a combined facet mesh containing all interface and boundary facets.

    Both groups share the parent's point array. A single ``interface_id`` cell
    array encodes the facet type:

    * Spinal facets (``Label.SPINAL_BUFFER`` against the CSF compartment):
      ``SPINAL_ID`` (= 99)
    * Other interface facets (between two labelled regions):
      ``min(a, b) * encoding_base + max(a, b)``
    * Boundary facets (outer surface):
      the region marker of the adjacent tet

    Parameters
    ----------
    mesh                  : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array           : str                  cell data array with region markers
    encoding_base         : int                  multiplier for the interface ID encoding;
                                                 must exceed the maximum label value
                                                 (default 100000, handles SAS markers ≤ ~12035)
    smooth_sas_labels     : bool                 if True, apply majority-vote smoothing
                                                 to SAS boundary labels after extraction
    ignore_sas_interfaces : bool                 if True (default), drop interfaces where both
                                                 adjacent markers are SAS subdivision labels
                                                 (> ``SAS_LABEL_OFFSET``)

    Returns a :class:`pyvista.PolyData` or :class:`pyvista.UnstructuredGrid` that
    shares the parent's point array, with an ``interface_id`` cell data array.
    """
    interfaces = mark_interface_facets(mesh, label_array=label_array,
                                       encoding_base=encoding_base,
                                       ignore_sas_interfaces=ignore_sas_interfaces)
    boundaries = mark_boundary_facets(mesh, label_array=label_array)
    if smooth_sas_labels:
        bnd = boundaries["boundary"]
        sas_labels = np.unique(bnd[bnd > SAS_LABEL_OFFSET]).tolist()
        boundaries = smooth_cell_labels(boundaries, marker_name="boundary", target_labels=sas_labels)

    # The buffer-to-CSF interfaces are the spinal opening — re-tag them as SPINAL_ID
    # so downstream code sees one id instead of CSF/SAS-specific encodings.
    int_ids = np.array(interfaces.cell_data["interface_id"], dtype=np.int64)
    int_ids[spinal_interface_mask(interfaces.cell_data["region_a"],
                                  interfaces.cell_data["region_b"])] = SPINAL_ID

    int_faces = _raw_faces(interfaces)
    bnd_faces = _raw_faces(boundaries)
    bnd_ids = boundaries.cell_data["boundary"]

    all_faces = (np.vstack([int_faces, bnd_faces]) if (len(int_faces) or len(bnd_faces))
                 else np.empty((0, bnd_faces.shape[1]), dtype=np.int64))

    n = len(all_faces)
    if isinstance(interfaces, pv.PolyData):
        cells_flat = np.column_stack([np.full(n, 3, dtype=np.int64), all_faces]).ravel()
        combined = pv.PolyData(mesh.points, faces=cells_flat)
    else:
        nodes_per_face = all_faces.shape[1] if n else 6
        cells_flat = np.column_stack(
            [np.full(n, nodes_per_face, dtype=np.int64), all_faces]
        ).ravel()
        all_ctypes = np.full(n, pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
        combined = pv.UnstructuredGrid(cells_flat, all_ctypes, mesh.points)

    combined.cell_data["interface_id"] = np.concatenate([int_ids, bnd_ids])
    return combined


def mark_boundary_facets(mesh, label_array="marker"):
    """
    Build a facet mesh of all outer boundary facets of the input tet mesh,
    each labelled by the marker of its single adjacent region.

    Face winding is oriented so each normal points outward (away from the
    parent tet centroid).

    The resulting :class:`pyvista.PolyData` shares the parent's point
    array, so vertex indices line up with the parent tet mesh.
    """
    topo = _tet_face_topology(mesh)
    markers = np.asarray(mesh.cell_data[label_array])
    points = np.asarray(mesh.points)
    tet_centroids = np.asarray(mesh.cell_centers().points)

    faces = topo["boundary_faces"]
    parents = topo["boundary_parents"]
    boundary = markers[parents].astype(np.int64)

    # Orient face winding so each normal points outward, away from the
    # parent tet centroid — same convention used in mark_spinal_boundary.
    if len(faces) > 0:
        corners = points[faces[:, :3]]
        centroids = corners.mean(axis=1)
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 2] - corners[:, 0]
        normals = np.cross(e1, e2)

        parent_centroids = tet_centroids[parents]
        flip = np.einsum("ij,ij->i", normals, centroids - parent_centroids) < 0

        _flip_winding(faces, flip)

    return _build_facet_polydata(mesh, faces, {"boundary": boundary})
