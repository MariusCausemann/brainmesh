import pytetwild
from src.brainmesh.utils import mark_mesh
import pyvista as pv
surf = pv.read("surf_dec.vtk")
mesh = pytetwild.tetrahedralize_pv(surf, stop_energy=10,
                                            loglevel=5, quiet=False,
                                            disable_filtering=True, 
                                            edge_length_fac=0.05,
                                            epsilon=1e-3,
                                            coarsen=False,
                                            num_threads=6)
mesh.save("mesh.vtk")
mesh = mark_mesh(mesh, surf)
mesh.save("mesh_marked.vtk")