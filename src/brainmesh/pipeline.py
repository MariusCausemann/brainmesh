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
        create_falx, create_tentorium,
        enforce_csf_around_tentorium, enforce_csf_around_falx,
        extend_brainstem_caudally,
        get_img,
        build_inferior_lateral_ventricle_horns,
        enforce_connected_ventricles,
        enforce_min_thickness,
        enforce_tight_ventricles,
        coarsen_surface,
        straighten_spinal_interface
    )

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = get_img(seg_path)
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

    # connect (often disconnected) inf. lateral ventricle horns with 
    # lateral ventricles
    data = build_inferior_lateral_ventricle_horns(data, radius=3)
    # make sure the V4 is not too thin
    data = enforce_min_thickness(data, Label.FOURTH_VENTRICLE, radius=1)
    # make sure the ventricles are correctly connected!
    data = enforce_connected_ventricles(data, min_thickness=2)
    # add a thin layer of tissue around the ventricles, 
    # to unphysiological connections
    data = enforce_tight_ventricles(data, thickness=3)

    data = nbm.mode_box(data)
    data = nbm.mode_diamond(data)

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
    surf = straighten_spinal_interface(surf, grid)
    surf.save(out_dir / "surf.vtk")

    surf_dec = coarsen_surface(surf, decimation_ratio=0.9)
    surf_dec.save(out_dir / "surf_dec.vtk")
    return surf


def surface_to_mesh(surf_path, out_dir="results", **tetwild_kwargs):
    """
    Tetrahedralise a surface and mark each cell with its anatomical label.

    Parameters
    ----------
    surf_path    : str or Path  path to a .vtk surface with boundary_labels
    out_dir      : str or Path  destination folder
    **tetwild_kwargs : forwarded to pytetwild.tetrahedralize_pv
    """
    import pytetwild

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
    )
    twild_defaults.update(tetwild_kwargs)

    mesh = pytetwild.tetrahedralize_pv(surf, **twild_defaults)
    mesh.save(out_dir / "mesh.vtk")
    mesh = mark_mesh(mesh, surf)
    mesh.save(out_dir / "mesh_marked.vtk")
    return mesh


def subdivide_SAS(seg_img, parc_img, numba_threads):
    from brainmesh import (Label, get_img, grow_into_region,
                            reverse_label_map,
                            VENTRICLE_LABELS)
    import numba
    numba.set_num_threads(numba_threads)
    # get images
    parc_img, seg_img = get_img(parc_img), get_img(seg_img)
    parc = parc_img.get_fdata().astype(np.uint16)
    seg = seg_img.get_fdata().astype(np.uint16)

    # create the SAS mask to fill
    sas_mask = (seg == Label.CSF)

    # create the "seed" mask - everything but boundaries and CSF
    excluded_labels = [Label.CSF, Label.FALX, Label.TENTORIUM, Label.WM_HYPOINTENSITIES] + VENTRICLE_LABELS
    excluded_mask = np.isin(seg, excluded_labels) + np.isin(parc, excluded_labels)
    parc[excluded_mask] = 0
    assert (parc==Label.CSF).sum() == 0

    # grow seeds into SAS
    labeled_SAS = grow_into_region(parc, ~excluded_mask + sas_mask, radius=2)
    # .. and keep only CSF
    labeled_SAS[~sas_mask] = 0
    labeled_SAS = np.where(np.isin(seg, VENTRICLE_LABELS), seg, labeled_SAS)

    labels, counts = np.unique(labeled_SAS, return_counts=True)
    sort_idx = np.argsort(-counts)
    for l, c in zip(labels[sort_idx], counts[sort_idx]):
        print(f"  - {c:<8} label: {reverse_label_map.get(l, l)}")
    return nib.Nifti1Image(labeled_SAS, parc_img.affine)
