import numpy as np
import pyvista as pv
import numba as nb
from tqdm import tqdm

# We import your exact shape gradient function to precompute the matrices
from .curved_mesh import eval_shape_gradients  # Adjust import based on your structure

@nb.njit
def build_node_to_cell_map(n_points, cells):
    """Creates a fast CSR map from node IDs to adjacent cell IDs."""
    counts = np.zeros(n_points, dtype=np.int32)
    for i in range(cells.shape[0]):
        for j in range(10):
            counts[cells[i, j]] += 1

    offsets = np.zeros(n_points + 1, dtype=np.int32)
    for i in range(n_points):
        offsets[i + 1] = offsets[i] + counts[i]

    data = np.zeros(offsets[-1], dtype=np.int32)
    current_offsets = offsets[:-1].copy()

    for i in range(cells.shape[0]):
        for j in range(10):
            pid = cells[i, j]
            idx = current_offsets[pid]
            data[idx] = i
            current_offsets[pid] += 1

    return offsets, data

@nb.njit(inline="always")
def max_length(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z):
    # Squared Edge lengths (avoiding sqrt)
    L0 = v1x**2 + v1y**2 + v1z**2
    L2 = v2x**2 + v2y**2 + v2z**2
    L3 = v3x**2 + v3y**2 + v3z**2
    
    L1 = (v2x-v1x)**2 + (v2y-v1y)**2 + (v2z-v1z)**2
    L4 = (v3x-v1x)**2 + (v3y-v1y)**2 + (v3z-v1z)**2
    L5 = (v3x-v2x)**2 + (v3y-v2y)**2 + (v3z-v2z)**2

    # Squared products
    # p1^2 = (l0 * l2 * l3)^2 = L0 * L2 * L3
    P1 = L0 * L2 * L3
    P2 = L0 * L1 * L4
    P3 = L1 * L2 * L5
    P4 = L3 * L4 * L5

    # Find max of squared products
    max_sq = max(P1, P2, P3, P4)
    return np.sqrt(max_sq)

@nb.njit(fastmath=True)
def calc_single_tet_quality(cell_nodes, points, shape_grads):
    """
    Computes the min scaled Jacobian for a single 10-node tet.
    Highly unrolled for maximum Numba performance.
    """
    min_q = 1e9
    for p in range(5): # Loop over 5 integration points
        # Calculate J = sum(points * grad)
        J00, J01, J02 = 0.0, 0.0, 0.0
        J10, J11, J12 = 0.0, 0.0, 0.0
        J20, J21, J22 = 0.0, 0.0, 0.0
        
        for i in range(10):
            pid = cell_nodes[i]
            px, py, pz = points[pid, 0], points[pid, 1], points[pid, 2]
            
            gx, gy, gz = shape_grads[p, i, 0], shape_grads[p, i, 1], shape_grads[p, i, 2]
            
            J00 += px * gx; J01 += py * gx; J02 += pz * gx
            J10 += px * gy; J11 += py * gy; J12 += pz * gy
            J20 += px * gz; J21 += py * gz; J22 += pz * gz

        # Determinant via scalar triple product
        cross_x = J11 * J22 - J12 * J21
        cross_y = J12 * J20 - J10 * J22
        cross_z = J10 * J21 - J11 * J20
        detJ = J00 * cross_x + J01 * cross_y + J02 * cross_z
        
        # Edge vectors
        v1x, v1y, v1z = J00, J01, J02
        v2x, v2y, v2z = J10, J11, J12
        v3x, v3y, v3z = J20, J21, J22

        max_l = max_length(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z)
        
        q = (1.4142135623730951 * detJ) / (max_l + 1e-14)
        if q < min_q:
            min_q = q
            
    return min_q

@nb.njit
def optimize_nodes_numba(points, cells, internal_ids, offsets, data,
                          shape_grads, iters, step_factor, target_quality):
    """
    Local steepest-ascent optimizer with dynamically scaled local step sizes.
    """
    scales = np.array([1.0, 0.5, 0.1, 0.01]) 
    total_moves = 0
    
    for iteration in range(iters):
        nodes_moved_this_iter = 0
        for pid in internal_ids:
            c_start = offsets[pid]
            c_end = offsets[pid+1]
            
            orig_p = points[pid].copy()
            
            min_dist = 1e9
            for i in range(c_start, c_end):
                cid = data[i]
                for j in range(10): # Check against all nodes in the tet
                    other_pid = cells[cid, j]
                    if other_pid != pid:
                        dx = orig_p[0] - points[other_pid, 0]
                        dy = orig_p[1] - points[other_pid, 1]
                        dz = orig_p[2] - points[other_pid, 2]
                        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                        if dist < min_dist:
                            min_dist = dist
                            
            # Base step is a fraction of the MINIMUM local distance
            if min_dist > 1e8: # Safety fallback for disconnected nodes
                min_dist = 1e-4
            local_base_step = min_dist * step_factor

            # 2. Evaluate current local quality
            best_q = 1e9 
            for i in range(c_start, c_end):
                cid = data[i]
                q = calc_single_tet_quality(cells[cid], points, shape_grads)
                if q < best_q:
                    best_q = q
            
            if best_q >= target_quality:
                continue

            best_p = orig_p.copy()
            improved = False
            
            # 3. Try multiple step sizes dynamically based on the LOCAL length
            for scale in scales:
                step = local_base_step * scale
                
                # Try 6 directions
                for d in range(6):
                    temp_p = orig_p.copy()
                    if d == 0: temp_p[0] += step
                    elif d == 1: temp_p[0] -= step
                    elif d == 2: temp_p[1] += step
                    elif d == 3: temp_p[1] -= step
                    elif d == 4: temp_p[2] += step
                    elif d == 5: temp_p[2] -= step
                    
                    points[pid] = temp_p # Temporarily apply
                    
                    min_q_new = 1e9
                    for i in range(c_start, c_end):
                        cid = data[i]
                        q = calc_single_tet_quality(cells[cid], points, shape_grads)
                        if q < min_q_new: min_q_new = q
                    
                    # If quality strictly improves, mark as best found
                    if min_q_new > best_q + 1e-6:
                        best_q = min_q_new
                        best_p = temp_p.copy()
                        improved = True
                        
                # If we found an improvement at this scale, break to avoid micro-stepping
                if improved:
                    break
                        
            # 4. Permanently update the array to the best position
            points[pid] = best_p
            if improved:
                nodes_moved_this_iter += 1
                
        total_moves += nodes_moved_this_iter
    if nodes_moved_this_iter == 0:
        print(f"no more improvement after {iteration} iterations.")
            
    return points, total_moves


def run_mesh_optimization(mesh, boundary_ids, iters=10, step_factor=0.05, target_quality=0.5):
    """
    Python wrapper for the Numba optimizer.
    step_factor: The maximum fraction of the local element size a node can move per iteration.
    """
    print(f"Starting local Numba optimization for {iters} iterations...")
    
    integration_points = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 
        (0.0, 0.0, 1.0), (0.25, 0.25, 0.25)
    ]
    shape_grads = np.zeros((5, 10, 3))
    for i, (xi, eta, zeta) in enumerate(integration_points):
        shape_grads[i] = eval_shape_gradients(xi, eta, zeta)
        
    cells = mesh.cells.reshape(-1, 11)[:, 1:] 
    points = mesh.points.copy()
    
    all_node_ids = np.arange(mesh.n_points)
    internal_ids = np.setdiff1d(all_node_ids, boundary_ids)
    
    offsets, data = build_node_to_cell_map(mesh.n_points, cells)
    
    # Pass step_factor instead of an absolute step
    optimized_points, total_moved = optimize_nodes_numba(
        points, cells, internal_ids, offsets, data, shape_grads,
         iters, step_factor, target_quality
    )
    
    print(f"  -> Optimizer finished. Total successful node nudges: {total_moved}")
    mesh.points = optimized_points
    return mesh