"""High-level pipeline functions combining segmentation cleanup, surface extraction, and meshing."""
import pathlib

import nibabel as nib
import nbmorph as nbm
import numpy as np
import pyvista as pv


def segmentation_to_surface(seg_path, out_dir="results", *, numba_threads=8):
    """
    Run the full segmentation-cleanup and surface-extraction pipeline.

    Cleans up a FreeSurfer-labelled segmentation (SynthSeg output at 0.5 mm),
    adds falx/tentorium, fixes ventricular connectivity, and extracts a
    multi-boundary surface mesh ready for tetrahedral meshing.

    Parameters
    ----------
    seg_path : str or Path
    out_dir  : str or Path  destination folder for results
    numba_threads : int     thread count passed to numba
    """
    import numba
    numba.set_num_threads(numba_threads)

    from brainmesh import (
        Label,
        nibabel_to_pyvista,
        solidify_csf, close_csf_space,
        fill_wm_hyperintensities, cut_bottom, extend_brainstem,
        enforce_csf_layer,
        diamond_mode_filter,
        create_falx, create_tentorium,
        enforce_csf_around_tentorium, enforce_csf_around_falx,
        extend_brainstem_caudally,
    )

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = nib.load(seg_path)
    seg = nib.as_closest_canonical(seg)
    data = np.ascontiguousarray(seg.get_fdata().astype(np.uint8))

    data = solidify_csf(data)
    data = close_csf_space(data, radius=3, iter=1)
    data = fill_wm_hyperintensities(data)
    data = cut_bottom(data)
    data = extend_brainstem(data)
    data = enforce_csf_layer(data, thickness=1)

    orig_mask = nbm.smooth_labels_spherical(data > 0, radius=1)
    data = nbm.mode_box(data)
    data = create_falx(data, hemisphere_distance=6)
    data = create_tentorium(data, distance=3)
    data = nbm.mode_box(data)
    data = diamond_mode_filter(data)

    data[~orig_mask] = 0
    data = enforce_csf_around_tentorium(data, radius=1)
    data = enforce_csf_around_falx(data, radius=1)
    data = enforce_csf_layer(data, thickness=1)
    data = extend_brainstem_caudally(data, offset=18)

    seg_out = nib.Nifti1Image(data, seg.affine)
    nib.save(seg_out, out_dir / "seg.nii.gz")

    grid = nibabel_to_pyvista(seg_out)
    grid["data"] = data.flatten(order="F")
    grid.save(out_dir / "seg.vti")

    surf = grid.contour_labels("all", smoothing=True)
    surf.save(out_dir / "surf.vtk")
    return surf


def surface_to_mesh(surf_path, out_dir="results", *, numba_threads=8, **tetwild_kwargs):
    """
    Tetrahedralise a surface and mark each cell with its anatomical label.

    Parameters
    ----------
    surf_path    : str or Path  path to a .vtk surface with boundary_labels
    out_dir      : str or Path  destination folder
    numba_threads: int
    **tetwild_kwargs : forwarded to pytetwild.tetrahedralize_pv
    """
    import numba
    import pytetwild
    numba.set_num_threads(numba_threads)

    from brainmesh import mark_mesh

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surf = pv.read(surf_path)

    twild_defaults = dict(
        stop_energy=10,
        loglevel=5,
        quiet=False,
        disable_filtering=True,
        edge_length_fac=0.05,
        epsilon=1e-3,
        coarsen=False,
        num_threads=numba_threads,
    )
    twild_defaults.update(tetwild_kwargs)

    mesh = pytetwild.tetrahedralize_pv(surf, **twild_defaults)
    mesh.save(out_dir / "mesh.vtk")
    mesh = mark_mesh(mesh, surf)
    mesh.save(out_dir / "mesh_marked.vtk")
    return mesh
