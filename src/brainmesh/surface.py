"""Surface extraction, smoothing, decimation, and label transfer."""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from .decorators import time_func


def transfer_labels(source_mesh, target_mesh, label_name="boundary_labels"):
    """Transfers cell labels from a dense source mesh to a coarse target mesh via nearest-neighbour."""
    tree = cKDTree(source_mesh.cell_centers().points)
    dist, idx = tree.query(target_mesh.cell_centers().points, k=1)
    target_mesh[label_name] = source_mesh[label_name][idx]
    return target_mesh


def smooth_1d_lines_taubin(line_mesh, iterations=50, lambda_val=0.5, mu_val=-0.5):
    """Smooths a 1D line mesh using Taubin (volume-preserving) smoothing."""
    lines = line_mesh.lines
    edges = []
    i = 0
    while i < len(lines):
        n_pts = lines[i]
        for j in range(n_pts - 1):
            edges.append([lines[i + 1 + j], lines[i + 2 + j]])
        i += n_pts + 1
    edges = np.array(edges)

    counts = np.zeros((len(line_mesh.points), 1))
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(counts, edges[:, 1], 1)
    counts[counts == 0] = 1

    pts = line_mesh.points.copy()
    for _ in range(iterations):
        avg_pts = np.zeros_like(pts)
        np.add.at(avg_pts, edges[:, 0], pts[edges[:, 1]])
        np.add.at(avg_pts, edges[:, 1], pts[edges[:, 0]])
        laplacian = (avg_pts / counts) - pts
        pts += lambda_val * laplacian

        avg_pts = np.zeros_like(pts)
        np.add.at(avg_pts, edges[:, 0], pts[edges[:, 1]])
        np.add.at(avg_pts, edges[:, 1], pts[edges[:, 0]])
        laplacian = (avg_pts / counts) - pts
        pts += mu_val * laplacian

    line_mesh.points = pts
    return line_mesh


@time_func
def straighten_spinal_interface(surf, orig_grid):
    # Get the local Z-axis direction (normal vector of the bottom plane)
    # Assuming standard affine conventions, the 3rd column is the Z-axis direction.
    n = orig_grid.direction_matrix[:, 2]
    n = n / np.linalg.norm(n) # Ensure it's a normalized unit vector

    # 2. Extract rough surface WITHOUT altering the direction matrix
    surf_rough = orig_grid.contour_labels("all", smoothing=False)
    
    # 3. Project rough points onto the normal vector to find the 'bottom'
    projections_rough = np.dot(surf_rough.points, n)
    min_proj_rough = projections_rough.min()
    
    # Mask for the bottom points
    bottom_points_mask = np.isclose(projections_rough, min_proj_rough)
    
    # 4. Flatten the actual surf's bottom points onto the tilted plane
    # Find the target minimum projection on the high-res surface
    min_proj_surf = np.dot(surf.points, n).min()
    
    # Calculate how far the masked points are from the target plane
    projections_masked = np.dot(surf.points[bottom_points_mask], n)
    distances_to_plane = projections_masked - min_proj_surf
    
    # Shift points along the normal vector to snap them to the plane
    surf.points[bottom_points_mask] -= distances_to_plane[:, None] * n
    
    # 5. Extract the bottom patch using cell centers
    centers = surf.cell_centers().points
    center_projections = np.dot(centers, n)
    min_center_proj = center_projections.min()
    
    bottom_patch = (
        surf.extract_cells(
            np.isclose(center_projections, min_center_proj)
        )
        .extract_surface(algorithm=None)
        .extract_feature_edges()
    )

    tree = cKDTree(surf.points)
    _, original_surf_indices = tree.query(bottom_patch.points)

    bottom_patch = smooth_1d_lines_taubin(bottom_patch)
    surf.points[original_surf_indices] = bottom_patch.points
    surf.field_data["grid_z_normal"] = n
    return surf

@time_func
def coarsen_surface(surf, decimation_ratio=0.5):
    surf_dec = surf.decimate(decimation_ratio)
    surf.field_data["grid_z_normal"] = surf_dec.field_data["grid_z_normal"]
    return transfer_labels(surf, surf_dec, "boundary_labels")
