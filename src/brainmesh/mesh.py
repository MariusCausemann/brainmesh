"""Tetrahedral mesh operations: marking, filtering, and facet extraction."""
import numpy as np
import igl
import pyvista as pv

from .decorators import time_func
from .io import get_img
from .labels import Label, VENTRICLE_LABELS, SAS_LABEL_OFFSET, SPINAL_ID


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


def extract_csf(mesh, label_array="marker", return_facets=False, **facet_kwargs):
    """
    Extract the CSF compartment — ``Label.CSF``, all ventricles and choroid plexus,
    and any SAS-subdivision markers (values > ``SAS_LABEL_OFFSET``).

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
                                         (e.g. ``max_angle``, ``max_distance``,
                                         ``encoding_base``)

    Returns
    -------
    csf_mesh : pv.UnstructuredGrid
    facets   : pv.PolyData or pv.UnstructuredGrid  (only when return_facets=True)
    """
    all_markers = np.asarray(mesh.cell_data[label_array])
    sas_labels = np.unique(all_markers[all_markers > SAS_LABEL_OFFSET]).tolist()
    csf_labels = list(VENTRICLE_LABELS) + [Label.CSF] + sas_labels
    csf_mesh = filter_by_label(mesh, csf_labels, label_array=label_array)

    if not return_facets:
        return csf_mesh

    # Compute facets on the full mesh (preserves CSF-to-tissue interface IDs),
    # then keep facets whose interface_id involves a CSF marker. Operates purely
    # on cell data, so this works for both linear and quadratic tet meshes.
    full_facets = mark_facets(mesh, label_array=label_array, **facet_kwargs)
    encoding_base = facet_kwargs.get("encoding_base", 100000)
    ids = np.asarray(full_facets.cell_data["interface_id"])
    a, b = np.divmod(ids, encoding_base)  # boundary: a=0, b=id; interface: (min,max)
    csf_arr = np.asarray(csf_labels)
    mask = np.isin(a, csf_arr) | np.isin(b, csf_arr) | (ids == SPINAL_ID)
    csf_facets = full_facets.extract_cells(mask)
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
    from .labels import region_dict, _sas_lh, _sas_rh
    ids = np.asarray(facets.cell_data["interface_id"], dtype=np.int64)
    a, b = np.divmod(ids, encoding_base)

    region = np.zeros(len(ids), dtype=np.int32)

    def _assign(mask, rid):
        region[mask & (region == 0)] = rid

    tent = int(Label.TENTORIUM)  # 71
    sas_labels = np.unique(ids[ids > SAS_LABEL_OFFSET]).tolist()
    csf_labels = VENTRICLE_LABELS + [Label.CSF] + sas_labels

    remove_id = int(-1)
    # remove all interfaces between different CSF regions
    _assign(np.logical_and(np.isin(a, csf_labels),
                           np.isin(b, csf_labels)), remove_id)

    # 1. Spinal canal
    _assign(ids == SPINAL_ID, 1)

    # mark lateral ventricle surface
    LVs = list(set(VENTRICLE_LABELS) - set((Label.THIRD_VENTRICLE,
                                            Label.FOURTH_VENTRICLE)))
    _assign(np.logical_xor(np.isin(a, LVs), np.isin(b, LVs)), 2)

    # mark FALX
    _assign((a==Label.FALX) + (b==Label.FALX), 4)

    # mark up and downward facing parts of tentorium
    _assign((a==tent) + (b==tent), 5)
    infra_tent_ids = list(region_dict["_SAS_INFRATENTORIAL"])
    region[np.logical_and(a==tent, np.isin(b, infra_tent_ids))] = 6
    region[np.logical_and(b==tent, np.isin(a, infra_tent_ids))] = 6

    # all remaining internal -> tissue
    _assign(ids >= encoding_base, 3)

    # mark specified regions:
    for i, (k, v) in enumerate(region_dict.items()):
        print(k, v)
        _assign(np.isin(ids, list(v)), 10 + i)

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
    region[np.logical_and(PSD, np.isin(ids, _sas_rh(anterior_PSD) + _sas_lh(anterior_PSD)))] = 8
    region[np.logical_and(PSD, np.isin(ids, _sas_rh(posterior_PSD) + _sas_lh(posterior_PSD)))] = 9
    
    sas_regions = np.unique(region[region >= 10]).tolist()
    out = facets.copy()
    region = remove_small_patches(out, region, threshold=50, target_labels=sas_regions)
    region = dilate_cell_marker(out, region, 10)
    out.cell_data["region"] = region
    out = smooth_cell_labels(out, marker_name="region", target_labels=sas_regions + [8,9])
    out = out.extract_cells(out.cell_data["region"] >= 0)
    assert (region==0).sum() == 0
    return out


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

    faces = topo["interface_faces"]
    m_a = markers[topo["interface_parents_a"]]
    m_b = markers[topo["interface_parents_b"]]

    diff = m_a != m_b
    faces = faces[diff]
    m_a = m_a[diff]
    m_b = m_b[diff]

    lo = np.minimum(m_a, m_b).astype(np.int64)
    hi = np.maximum(m_a, m_b).astype(np.int64)

    if ignore_sas_interfaces:
        keep = lo <= SAS_LABEL_OFFSET
        faces, lo, hi = faces[keep], lo[keep], hi[keep]

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

def mark_spinal_boundary(mesh, label_array="marker", max_angle=25.0, max_distance=10.0,
                          csf_labels=None):
    """
    Mark the outer CSF boundary facets that correspond to the spinal interface.

    A facet is selected when all three criteria are satisfied:
    1. Its parent tet carries a CSF label (``csf_labels``).
    2. Its outward normal points downward within ``max_angle`` degrees of (0, 0, -1).
    3. Its centroid z-coordinate is within ``max_distance`` (mesh units) of the lowest
       boundary face centroid.

    Parameters
    ----------
    mesh         : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array  : str                  cell data array with region markers
    max_angle    : float                maximum deviation (degrees) from straight down
    max_distance : float                maximum z-distance from the lowest boundary
                                        face centroid (in mesh units)
    csf_labels   : array-like or None   region markers treated as CSF; defaults to
                                        ``[Label.CSF]``

    Returns a :class:`pyvista.PolyData` or :class:`pyvista.UnstructuredGrid` that
    shares the parent's point array, with a ``boundary`` cell data array.
    """
    if csf_labels is None:
        csf_labels = [Label.CSF]

    topo = _tet_face_topology(mesh)
    points = np.asarray(mesh.points)
    markers = np.asarray(mesh.cell_data[label_array])

    faces = topo["boundary_faces"]
    boundary = markers[topo["boundary_parents"]].astype(np.int64)
    print(boundary.max())

    # Corner nodes are always the first 3 (same for linear and quadratic faces).
    corners = points[faces[:, :3]]          # (N, 3, 3)
    centroids = corners.mean(axis=1)        # (N, 3)

    up_normal = np.asarray(mesh.field_data["grid_z_normal"]).flatten()
    print(up_normal)
    down_normal = -up_normal

    # Compute raw face normals from corner winding order.
    e1 = corners[:, 1] - corners[:, 0]
    e2 = corners[:, 2] - corners[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-12)

    # Orient normals outward: flip any that point toward their parent tet centroid.
    tet_centroids = np.asarray(mesh.cell_centers().points)
    parent_centroids = tet_centroids[topo["boundary_parents"]]
    inward = np.einsum("ij,ij->i", normals, parent_centroids - centroids) > 0
    normals[inward] *= -1

    # Criterion 1: parent tet is a CSF cell.
    csf_mask = np.isin(boundary, csf_labels) | (boundary > SAS_LABEL_OFFSET)

    # Criterion 2: outward normal within max_angle of the recovered down_normal.
    dot_products = np.dot(normals, down_normal)
    cos_thresh = np.cos(np.deg2rad(max_angle))
    normal_mask = dot_products > cos_thresh

    # Criterion 3: centroid projection within max_distance of the lowest boundary face.
    # Project all centroids onto the UPWARD normal axis
    projections = np.dot(centroids, up_normal)
    min_proj = projections.min()
    z_mask = projections <= min_proj + max_distance

    spinal_mask = csf_mask & normal_mask & z_mask
    spinal_faces = faces[spinal_mask]
    spinal_boundary = boundary[spinal_mask]

    if len(spinal_faces) > 1:
        # Build a compact PolyData using only corner nodes to find connected components.
        # Works for both linear (3-node) and quadratic (6-node) faces.
        M = len(spinal_faces)
        corners = spinal_faces[:, :3]
        unique_pts, inverse = np.unique(corners, return_inverse=True)
        compact_corners = inverse.reshape(corners.shape)
        faces_flat = np.column_stack(
            [np.full(M, 3, dtype=np.int64), compact_corners]
        ).ravel()
        compact = pv.PolyData(points[unique_pts], faces=faces_flat)
        compact.cell_data["original_id"] = np.arange(M, dtype=np.int64)
        keep_ids = compact.extract_largest().cell_data["original_id"]
        spinal_faces = spinal_faces[keep_ids]
        spinal_boundary = spinal_boundary[keep_ids]

    return _build_facet_polydata(mesh, spinal_faces, {"boundary": spinal_boundary})


def mark_facets(mesh, label_array="marker", max_angle=10.0, max_distance=0.5,
                encoding_base=100000, smooth_sas_labels=False,
                ignore_sas_interfaces=True):
    """
    Build a combined facet mesh containing all interface, boundary, and spinal facets.

    All three groups share the parent's point array. A single ``interface_id`` cell
    array encodes the facet type:

    * Interface facets (between two labelled regions):
      ``min(a, b) * encoding_base + max(a, b)``
    * Regular boundary facets (outer surface, non-spinal):
      the region marker of the adjacent tet
    * Spinal boundary facets (lowest CSF boundary, downward-facing):
      ``SPINAL_ID`` (= 99)

    Parameters
    ----------
    mesh                  : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array           : str                  cell data array with region markers
    max_angle             : float                maximum deviation (degrees) from straight down
                                                 for spinal boundary detection
    max_distance          : float                maximum z-distance from the lowest boundary
                                                 face centroid for spinal boundary detection
                                                 (in mesh units)
    encoding_base         : int                  multiplier for the interface ID encoding;
                                                 must exceed the maximum label value
                                                 (default 100000, handles SAS markers ≤ ~12035)
    smooth_sas_labels     : bool                 if True (default), apply majority-vote smoothing
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
    spinal     = mark_spinal_boundary(mesh, label_array=label_array,
                                      max_angle=max_angle, max_distance=max_distance)

    # Extract raw face arrays (N×3 for linear, N×6 for quadratic).
    def _raw_faces(facet_mesh):
        if isinstance(facet_mesh, pv.PolyData):
            return facet_mesh.faces.reshape(-1, 4)[:, 1:]
        return facet_mesh.cells_dict.get(pv.CellType.QUADRATIC_TRIANGLE,
                                         np.empty((0, 6), dtype=np.int64))

    bnd_faces = _raw_faces(boundaries)
    bnd_ids   = boundaries.cell_data["boundary"]

    # Remove spinal faces from boundaries to avoid duplicates.
    spinal_key_set = set(map(tuple, np.sort(_raw_faces(spinal)[:, :3], axis=1).tolist()))
    is_spinal    = np.array([tuple(r) in spinal_key_set
                             for r in np.sort(bnd_faces[:, :3], axis=1)])
    regular_faces = bnd_faces[~is_spinal]
    regular_ids   = bnd_ids[~is_spinal]
    spinal_faces  = bnd_faces[is_spinal]
    spinal_ids    = np.full(is_spinal.sum(), SPINAL_ID, dtype=np.int64)

    int_faces = _raw_faces(interfaces)
    all_faces = np.vstack([int_faces, regular_faces, spinal_faces]) if (
        len(int_faces) or len(regular_faces) or len(spinal_faces)
    ) else np.empty((0, bnd_faces.shape[1]), dtype=np.int64)

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

    combined.cell_data["interface_id"] = np.concatenate([
        interfaces.cell_data["interface_id"],
        regular_ids,
        spinal_ids,
    ])
    combined.field_data["grid_z_normal"] = mesh.field_data["grid_z_normal"]
    return combined


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
