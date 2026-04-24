import numba
numba.set_num_threads(8)
import pyvista as pv
import nibabel as nib
import numpy as np
from src.brainmesh.utils import (nibabel_to_pyvista, 
                                enforce_csf_layer, enforce_cortex_layer,
                                enforce_wm_thickness, upsample_nib,
                                create_falx, create_tentorium,
                                extend_brainstem_caudally,
                                enforce_min_thickness,
                                enforce_connected_ventricles,
                                enforce_tight_ventricles,
                                build_inferior_lateral_ventricle_horns,
                                enforce_csf_around_tentorium,
                                enforce_csf_around_falx,
                                straighten_spinal_interface,
                                coarsen_surface,
                                fill_wm_hyperintensities,
                                fill_holes_csf,
                                solidify_csf,
                                diamond_mode_filter,
                                extend_brainstem,
                                close_csf_space,
                                cut_bottom)
from src.brainmesh.synthseg_labels import Label
import nbmorph as nbm

# this scripts attempts to fix known issues with typical brain
# segmentation for FEM flow and transport simulations. 
# In particular, tries to:
# * ensure a continous SAS/CSF sheath around the brain
# * adds falx and tentorium
# * fixes the connectivity of the ventricular system
# * creates a reasonable, straight interface to the spinal compartment
# Finally, it creates a multi-boundary surface mesh with surfaceNets, 
# ready for meshing with fTetwild
# remaining problems / potential issues:
# * fragile reconstruction of the interface towards the spine
# * surfaceNets respects corner connections, creating small "spikes"
# in the surface mesh if some structure is too thin
# (e.g. cortex or CSF thinner than 1 voxel)

# we assume a freesurfer-labeled segmentation at 0.5mm
seg = nib.load("testdata/sub1_gouhfi_hybrid_seg.nii.gz")

# make sure it is in RAS orientation, aligning neck-to-top with z-axis
seg = nib.as_closest_canonical(seg)

data = np.ascontiguousarray(seg.get_fdata().astype(np.uint8))
# the CSF space has holes, especially steming from vessels close to 
# the brain stem. Use closing to remove fill them with CSF.
data = solidify_csf(data)
data = close_csf_space(data, radius=3, iter=1)

# we ignore WM-hyperintensities, and fill them with WM labels
data = fill_wm_hyperintensities(data)

# the continuation towards the spine is often rough, 
# we cut of to get a straight interface
data = cut_bottom(data)
# the bottom part of the BS is often missing, we extend it downwards
data = extend_brainstem(data)

# we want a CSF layer around the hole brain, except at the bottom BS
data = enforce_csf_layer(data, thickness=1)

# store a mask to prevent the labels from growing
orig_mask = nbm.smooth_labels_spherical(data>0, radius=1)
data = nbm.mode_box(data)
# create a falx between the hemispheres
data = create_falx(data, hemisphere_distance=6)
data = create_tentorium(data, distance=3)

# connect (often disconnected) inf. lateral ventricle horns with 
# lateral ventricles
#data = build_inferior_lateral_ventricle_horns(data, radius=3)
# make sure the V4 is not too thin
#data = enforce_min_thickness(data, Label.FOURTH_VENTRICLE, radius=1)
# make sure the ventricles are correctly connected!
#data = enforce_connected_ventricles(data, min_thickness=2)
# add a thin layer of tissue around the ventricles, 
# to unphysiological connections
#data = enforce_tight_ventricles(data, thickness=3)
# smooth a bit
data = nbm.mode_box(data)
data = diamond_mode_filter(data)

data[~orig_mask] = 0

# make sure we have a thin layer of fluid around the tentorium
# the falx, and the brain tissue
data = enforce_csf_around_tentorium(data, radius=1)
data = enforce_csf_around_falx(data, radius=1)
data = enforce_csf_layer(data, thickness=1)
# extend the BS (again?)
data = extend_brainstem_caudally(data, offset=18)

seg = nib.Nifti1Image(data, seg.affine)
nib.save(seg, "results/seg.nii.gz")


grid = nibabel_to_pyvista(seg)
grid["data"] = data.flatten(order="F")

surf = grid.contour_labels("all", smoothing=True)

grid.save("results/seg.vti")
#surf = straighten_spinal_interface(surf, grid)
surf.save("results/surf.vtk")

exit()
surf_dec = coarsen_surface(surf, decimation_ratio=0.8)
surf_dec.save("results/surf_dec.vtk")
del surf, surf_dec
print("finished")

