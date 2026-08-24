"""Surface extraction, smoothing, decimation, and label transfer."""
from scipy.spatial import cKDTree

from .decorators import time_func


def transfer_labels(source_mesh, target_mesh, label_name="boundary_labels"):
    """Transfers cell labels from a dense source mesh to a coarse target mesh via nearest-neighbour."""
    tree = cKDTree(source_mesh.cell_centers().points)
    dist, idx = tree.query(target_mesh.cell_centers().points, k=1)
    target_mesh[label_name] = source_mesh[label_name][idx]
    return target_mesh


@time_func
def coarsen_surface(surf, decimation_ratio=0.5):
    surf_dec = surf.decimate(decimation_ratio)
    return transfer_labels(surf, surf_dec, "boundary_labels")
