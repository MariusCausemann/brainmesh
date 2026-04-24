import numba
#numba.config.THREADING_LAYER = 'workqueue'
numba.set_num_threads(8)
import pyvista as pv
import pytetwild
import numpy as np
from src.brainmesh.utils import mark_mesh
import nbmorph

N = 20
r = 2
data = np.ones(shape=(N,N,N), dtype=np.uint8)
grid = pv.ImageData(dimensions=np.array(data.shape) + 1, spacing=(1/N, 1/N, 1/N))
p = grid.cell_centers().points.reshape(N,N,N,3)
data[np.linalg.norm(p-0.5, axis=-1) < 0.3] = 2 
data[np.linalg.norm(p-0.5, axis=-1) < 0.22] = 3 
#data = nbmorph.smooth_labels_spherical(data, radius=1, dilate_radius=1)


grid["data"] = data.flatten(order="F")
grid.resample(interpolation="nearest", sample_rate=r, inplace=True)
data = grid["data"].reshape(r*N,r*N,r*N)
data = nbmorph.erode_labels_spherical(data, radius=1, struct_sequence="D")
data = nbmorph.dilate_labels_spherical(data, radius=1, struct_sequence="B")
data = nbmorph.dilate_labels_spherical(data, radius=1)

grid["data"] = data.flatten(order="F")
grid.save("spheres.vti")
surf = grid.contour_labels("all", smoothing=True)
surf.save("spheres.vtk")

mesh = pytetwild.pytetwild.tetrahedralize_pv(surf, stop_energy=10,
                                            loglevel=6, quiet=False,
                                            disable_filtering=True, 
                                            edge_length_fac=0.05,
                                            epsilon=1e-3,
                                            coarsen=False,
                                            num_threads=6)
mesh.save("mesh.vtk")
mesh = mark_mesh(mesh, surf)
mesh.save("mesh_marked.vtk")