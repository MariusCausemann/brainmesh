"""Tetrahedral mesh operations: marking and label transfer."""
import numpy as np
import igl

from .decorators import time_func


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
