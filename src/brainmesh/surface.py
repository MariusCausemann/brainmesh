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
    grid = orig_grid.copy()
    grid.direction_matrix = np.round(grid.direction_matrix, 0)
    surf_rough = grid.contour_labels("all", smoothing=False)
    bottom_points = np.isclose(surf_rough.points[:, 2], surf_rough.points[:, 2].min())
    surf.points[bottom_points, 2] = surf.points[:, 2].min()
    bottom_patch = (
        surf.extract_cells(
            np.isclose(surf.cell_centers().points[:, 2], surf.points[:, 2].min())
        )
        .extract_surface(algorithm=None)
        .extract_feature_edges()
    )

    tree = cKDTree(surf.points)
    _, original_surf_indices = tree.query(bottom_patch.points)

    bottom_patch = smooth_1d_lines_taubin(bottom_patch)
    surf.points[original_surf_indices] = bottom_patch.points
    return surf


@time_func
def coarsen_surface(surf, decimation_ratio=0.5):
    surf_dec = surf.decimate(decimation_ratio)
    return transfer_labels(surf, surf_dec, "boundary_labels")
