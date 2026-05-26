import pyvista as pv
import numpy as np
import vtk
from tqdm import tqdm

from .mesh import mark_interface_facets


def convert_to_quadratic(tet_mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Converts a linear tetrahedral mesh to a quadratic tetrahedral mesh."""
    filter_quad = vtk.vtkLinearToQuadraticCellsFilter()
    filter_quad.SetInputData(tet_mesh)
    filter_quad.Update()
    return pv.wrap(filter_quad.GetOutput())


def eval_shape_gradients(xi, eta, zeta):
    """
    Evaluates the derivatives of the 10-node tetrahedron shape functions 
    with respect to the reference coordinates (xi, eta, zeta).
    Returns a (10, 3) numpy array.
    """
    L0 = 1.0 - xi - eta - zeta
    
    # dN / dxi
    dN_dxi = [
        1 - 4*L0,      4*xi - 1,   0,             0,
        4*(L0 - xi),   4*eta,      -4*eta,        -4*zeta,   4*zeta,    0
    ]
    
    # dN / deta
    dN_deta = [
        1 - 4*L0,      0,          4*eta - 1,     0,
        -4*xi,         4*xi,       4*(L0 - eta),  -4*zeta,   0,         4*zeta
    ]
    
    # dN / dzeta
    dN_dzeta = [
        1 - 4*L0,      0,          0,             4*zeta - 1,
        -4*xi,         0,          -4*eta,        4*(L0 - zeta), 4*xi,  4*eta
    ]
    
    return np.column_stack([dN_dxi, dN_deta, dN_dzeta])

def compute_quadratic_quality(mesh):
    """
    Computes the minimum exact Jacobian determinant for every 2nd-order 
    tetrahedron in a PyVista mesh.
    
    Returns: A 1D numpy array of the minimum Jacobian determinant per cell.
    """
    # 24 is the VTK cell type code for VTK_QUADRATIC_TETRA
    if 24 not in mesh.cells_dict:
        raise ValueError("No quadratic tetrahedra (type 24) found in the mesh!")
        
    # Get the physical (x, y, z) coordinates for all 10 nodes of every cell
    # Shape of tet_points: (N_cells, 10, 3)
    tet_nodes = mesh.cells_dict[24]
    tet_points = mesh.points[tet_nodes]
    
    # Evaluate at the 4 vertices and the element center
    integration_points = [
        (0.0, 0.0, 0.0),   # Node 0
        (1.0, 0.0, 0.0),   # Node 1
        (0.0, 1.0, 0.0),   # Node 2
        (0.0, 0.0, 1.0),   # Node 3
        (0.25, 0.25, 0.25) # Center
    ]
    
    N_cells = len(tet_nodes)
    min_jacobians = np.full(N_cells, np.inf)
            
    for (xi, eta, zeta) in tqdm(integration_points):
        # Get shape function gradients: Shape (10, 3)
        dNdX = eval_shape_gradients(xi, eta, zeta)
        J = np.einsum('nik,ij->nkj', tet_points, dNdX)
        
        # Unpack the 3 local basis vectors: Shape (N_cells, 3)
        v1, v2, v3 = J[:, :, 0], J[:, :, 1], J[:, :, 2]
        
        # 1. Determinant via Scalar Triple Product: det(J) = v1 · (v2 × v3)
        detJ = np.einsum('ni,ni->n', v1, np.cross(v2, v3))
        
        # 2. Base edge lengths (norms of v1, v2, v3 computed all at once)
        l0, l2, l3 = np.linalg.norm(J, axis=1).T 
        
        # 3. Cross-edge lengths
        l1 = np.linalg.norm(v2 - v1, axis=1)
        l4 = np.linalg.norm(v3 - v1, axis=1)
        l5 = np.linalg.norm(v3 - v2, axis=1)
        
        # 4. Max edge-length product among the 4 corners
        max_length_product = np.max([
            l0 * l2 * l3,
            l0 * l1 * l4,
            l1 * l2 * l5,
            l3 * l4 * l5
        ], axis=0)
        
        # 5. Compute scaled Jacobian and update minimums in-place
        scaled_J = (np.sqrt(2.0) * detJ) / (max_length_product + 1e-14)
        np.minimum(min_jacobians, scaled_J, out=min_jacobians)
            
    return min_jacobians


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
            
        print(f"  -> Iteration {i+1}: Found {len(bad_tet_idx)} ({100 *len(bad_tet_idx) / len(all_boundary_ids):.3f}%) bad elements. Relaxing nodes with decay step {decay_step}...")
        
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

    print("\n  -> Final Alpha Distribution:")
    unique_alphas, counts = np.unique(np.round(alpha, decimals=5), return_counts=True)
    
    # Sort them in descending order (from 1.0 down to 0.0)
    sort_idx = np.argsort(unique_alphas)[::-1]
    unique_alphas = unique_alphas[sort_idx]
    counts = counts[sort_idx]
    
    total_nodes = len(all_boundary_ids)
    for a_val, count in zip(unique_alphas, counts):
        pct = (count / total_nodes) * 100
        print(f"       Alpha = {a_val:.2f}: {count:7d} nodes ({pct:5.2f}%)")
    print("-" * 50)
    return all_boundary_ids

def print_quality_stats(mesh: pv.UnstructuredGrid, mesh_name: str):
    """Computes and prints the Scaled Jacobian quality of the mesh."""
    
    if mesh.celltypes[0] == vtk.VTK_QUADRATIC_TETRA:
        q_arr = compute_quadratic_quality(mesh)

    else:
        mesh_with_quality = mesh.cell_quality(quality_measure='scaled_jacobian')
        q_arr = mesh_with_quality.cell_data['scaled_jacobian']
    
    print(f"--- {mesh_name} Cell Quality (Scaled Jacobian) ---")
    print(f"  Min:  {q_arr.min():.4f} (values <= 0 indicate inverted cells)")
    print(f"  Mean: {q_arr.mean():.4f}")
    print(f"  Max:  {q_arr.max():.4f}\n")
    return q_arr

