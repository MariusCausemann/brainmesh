import pyvista as pv
import numpy as np
import vtk

from .mesh import mark_interface_facets


def convert_to_quadratic(tet_mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Converts a linear tetrahedral mesh to a quadratic tetrahedral mesh."""
    filter_quad = vtk.vtkLinearToQuadraticCellsFilter()
    filter_quad.SetInputData(tet_mesh)
    filter_quad.Update()
    return pv.wrap(filter_quad.GetOutput())


def compute_quadratic_quality(quad_mesh: pv.UnstructuredGrid,
                              quality_measure='scaled_jacobian') -> np.ndarray:
    """
    Evaluates quadratic tetrahedral quality by subdividing into 8 linear sub-tets.
    Dynamically slices the inner octahedron using its shortest diagonal to 
    eliminate artificial distortion penalties.
    """
    points = quad_mesh.points
    cells = quad_mesh.cells.reshape(-1, 11)[:, 1:]
    n_tets = len(cells)
    
    # 1. Define the 4 corner tets (these are always the same)
    t0 = cells[:, [0, 4, 6, 7]]
    t1 = cells[:, [4, 1, 5, 8]]
    t2 = cells[:, [6, 5, 2, 9]]
    t3 = cells[:, [7, 8, 9, 3]]
    
    # 2. Extract coordinates of the 6 mid-edge nodes (the octahedron corners)
    p4, p5 = points[cells[:, 4]], points[cells[:, 5]]
    p6, p7 = points[cells[:, 6]], points[cells[:, 7]]
    p8, p9 = points[cells[:, 8]], points[cells[:, 9]]
    
    # 3. Calculate squared lengths of the 3 internal octahedron diagonals
    d0 = np.sum((p4 - p9)**2, axis=1)  # Diagonal 4-9
    d1 = np.sum((p5 - p7)**2, axis=1)  # Diagonal 5-7
    d2 = np.sum((p6 - p8)**2, axis=1)  # Diagonal 6-8
    
    # Find which diagonal is the shortest for each individual tetrahedron
    d_stack = np.vstack((d0, d1, d2))
    best_diag = np.argmin(d_stack, axis=0)
    
    # 4. Pre-allocate arrays for the 4 core sub-tetrahedra
    t4 = np.empty((n_tets, 4), dtype=np.int64)
    t5 = np.empty((n_tets, 4), dtype=np.int64)
    t6 = np.empty((n_tets, 4), dtype=np.int64)
    t7 = np.empty((n_tets, 4), dtype=np.int64)
    
    # 5. Populate core tets using the optimal, right-handed configuration
    m0, m1, m2 = (best_diag == 0), (best_diag == 1), (best_diag == 2)
    
    # Config 0: Slice using Diagonal 4-9
    if np.any(m0):
        t4[m0] = cells[m0][:, [4, 5, 6, 9]]
        t5[m0] = cells[m0][:, [4, 8, 5, 9]]
        t6[m0] = cells[m0][:, [4, 7, 8, 9]]
        t7[m0] = cells[m0][:, [4, 6, 7, 9]]
        
    # Config 1: Slice using Diagonal 5-7
    if np.any(m1):
        t4[m1] = cells[m1][:, [5, 6, 4, 7]]
        t5[m1] = cells[m1][:, [5, 9, 6, 7]]
        t6[m1] = cells[m1][:, [5, 8, 9, 7]]
        t7[m1] = cells[m1][:, [5, 4, 8, 7]]
        
    # Config 2: Slice using Diagonal 6-8
    if np.any(m2):
        t4[m2] = cells[m2][:, [6, 4, 5, 8]]
        t5[m2] = cells[m2][:, [6, 5, 9, 8]]
        t6[m2] = cells[m2][:, [6, 9, 7, 8]]
        t7[m2] = cells[m2][:, [6, 7, 4, 8]]
        
    # 6. Combine, build mesh, and calculate quality
    sub_tets = np.vstack((t0, t1, t2, t3, t4, t5, t6, t7))
    
    proxy_cells = np.column_stack((np.full(len(sub_tets), 4, dtype=np.int64), sub_tets)).ravel()
    cell_types = np.full(len(sub_tets), pv.CellType.TETRA, dtype=np.uint8)
    proxy_mesh = pv.UnstructuredGrid(proxy_cells, cell_types, points)
    
    proxy_with_quality = proxy_mesh.cell_quality(quality_measure=quality_measure)
    sub_tet_qualities = proxy_with_quality.cell_data[quality_measure]
    
    parent_qualities = sub_tet_qualities.reshape((8, n_tets))
    worst_quality_per_parent = parent_qualities.min(axis=0)
    
    return worst_quality_per_parent


def adaptive_snap_boundaries(
    quad_mesh: pv.UnstructuredGrid, 
    target_surface: pv.PolyData,
    label_array: str = "marker",
    only_high_order: bool = False,
    min_quality: float = 0.09,
    max_iters: int = 10,
    decay_step: float = 0.2
):
    """
    Snaps boundary nodes to a target surface adaptively. If an element inverts or falls
    below the required min quality, it iteratively retracts the snapped nodes of that
    element by `decay_step` until the quality requirement is met.
    """
    print("Extracting boundary nodes...")
    global_id_name = "GlobalNodeID"
    quad_mesh.point_data[global_id_name] = np.arange(quad_mesh.n_points)
    
    # 1. Gather External Boundary IDs
    surf = quad_mesh.extract_surface(algorithm='dataset_surface')
    ext_boundary_ids = surf.point_data[global_id_name]
    
    # 2. Gather Internal Interface IDs
    int_boundary_ids = np.array([], dtype=int)
    if label_array in quad_mesh.cell_data:
        interface_facets = mark_interface_facets(quad_mesh, label_array=label_array)
        if interface_facets.n_cells > 0:
            # Connectivity lives in `.faces` for PolyData (linear facets) and `.cells`
            # for UnstructuredGrid (quadratic facets); both follow the VTK stride layout.
            conn = (interface_facets.faces if isinstance(interface_facets, pv.PolyData)
                    else interface_facets.cells)
            nodes_per_face = conn[0]
            faces = conn.reshape(-1, nodes_per_face + 1)[:, 1:]
            int_boundary_ids = np.unique(faces)
            
    all_boundary_ids = np.unique(np.concatenate((ext_boundary_ids, int_boundary_ids)))
    
    # 3. Apply high-order topological filter if requested
    if only_high_order:
        cells = quad_mesh.cells.reshape(-1, 11)
        corner_ids = np.unique(cells[:, 1:5])
        all_boundary_ids = np.setdiff1d(all_boundary_ids, corner_ids)
        
    del quad_mesh.point_data[global_id_name]
    
    if len(all_boundary_ids) == 0:
        return
        
    #  SNAPPING LOGIC
    # calculate the starting and target coordinates
    P_orig = quad_mesh.points[all_boundary_ids].copy()
    _, P_target = target_surface.find_closest_cell(P_orig, return_closest_point=True)
    
    # Initialize the interpolation weight alpha for each node (1.0 = fully snapped)
    alpha = np.ones(len(all_boundary_ids))
    
    # reverse-lookup array: Global Point ID -> Index in boundary array
    p2b = np.full(quad_mesh.n_points, -1)
    p2b[all_boundary_ids] = np.arange(len(all_boundary_ids))
    
    # Extract raw cell definitions
    raw_cells = quad_mesh.cells.reshape(-1, 11)[:, 1:]
    
    for i in range(max_iters):
        # Update the mesh points using the current alpha weights
        quad_mesh.points[all_boundary_ids] = P_orig + alpha[:, np.newaxis] * (P_target - P_orig)
        
        # Compute the quality
        qualities = compute_quadratic_quality(quad_mesh)
        bad_tet_idx = np.where(qualities < min_quality)[0]
        
        if len(bad_tet_idx) == 0:
            print(f"  -> Adaptive snapping converged successfully at iteration {i+1}!")
            break
            
        print(f"  -> Iteration {i+1}: Found {len(bad_tet_idx)} ({100 *len(bad_tet_idx) / len(all_boundary_ids):.3f}%) bad elements. Relaxing nodes...")
        
        # Find which specific nodes belong to the inverted elements
        bad_point_ids = np.unique(raw_cells[bad_tet_idx])
        
        # Map those global point IDs to their index in our boundary array
        bad_bnd_idx = p2b[bad_point_ids]
        
        # Filter out nodes that aren't actually on the boundary (-1)
        bad_bnd_idx = bad_bnd_idx[bad_bnd_idx != -1]
        
        # Relax the alpha for those specific nodes (alpha only ever decreases from 1.0).
        alpha[bad_bnd_idx] = np.maximum(alpha[bad_bnd_idx] - decay_step, 0.0)
        
        # Safety break if all bad boundary nodes have completely reverted to 0.0
        if np.all(alpha[bad_bnd_idx] == 0.0):
            print(f"  -> Reached maximum relaxation (linear state) for problematic nodes at iteration {i+1}.")
            # Apply final 0.0 state before breaking
            quad_mesh.points[all_boundary_ids] = P_orig + alpha[:, np.newaxis] * (P_target - P_orig)
            break


def print_quality_stats(mesh: pv.UnstructuredGrid, mesh_name: str, approx_linear: bool = False):
    """Computes and prints the Scaled Jacobian quality of the mesh."""
    
    # Check if the mesh is quadratic. If so, drop to linear JUST for the quality check.
    # VTK returns -1.0 for all non-linear cell qualities.
    if approx_linear and mesh.celltypes[0] == vtk.VTK_QUADRATIC_TETRA:
        print(f"  [*] Note: {mesh_name} is quadratic. Evaluating quality using corner nodes only.")
        eval_mesh = mesh.linear_copy()
    else:
        eval_mesh = mesh

    if eval_mesh.celltypes[0] == vtk.VTK_QUADRATIC_TETRA:
        print(f"  [*] Note: {mesh_name} is quadratic. Approximating by subdivision into 8 tetra.")
        q_arr = compute_quadratic_quality(eval_mesh)

    else:
        mesh_with_quality = eval_mesh.cell_quality(quality_measure='scaled_jacobian')
        q_arr = mesh_with_quality.cell_data['scaled_jacobian']
    
    print(f"--- {mesh_name} Cell Quality (Scaled Jacobian) ---")
    print(f"  Min:  {q_arr.min():.4f} (values <= 0 indicate inverted cells)")
    print(f"  Mean: {q_arr.mean():.4f}")
    print(f"  Max:  {q_arr.max():.4f}\n")
    return q_arr

