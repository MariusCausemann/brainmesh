"""High-level pipeline functions combining segmentation cleanup, surface extraction, and meshing."""
import pathlib
from dataclasses import asdict

import nibabel as nib
import nbmorph as nbm
import numpy as np
import pyvista as pv

from .config import SegmentationConfig


def segmentation_to_surface(seg_path, out_seg=None, out_surf=None, *,
                            config: SegmentationConfig | None = None,
                            numba_threads=8):
    """
    Run the full segmentation-cleanup and surface-extraction pipeline.

    Cleans up a FreeSurfer-labelled segmentation (SynthSeg output at 0.5 mm),
    adds falx/tentorium, fixes ventricular connectivity, and extracts a
    multi-boundary surface mesh ready for tetrahedral meshing.

    Parameters
    ----------
    seg_path : str or Path
    out_seg  : str or Path, optional  destination file for the cleaned segmentation
    out_surf : str or Path, optional  destination file for the extracted surface
    config   : SegmentationConfig, optional  per-step parameter overrides
    numba_threads : int               thread count passed to numba
    """
    import numba
    numba.set_num_threads(numba_threads)

    from brainmesh import (
        Label,
        nibabel_to_pyvista,
        save_mesh,
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
        straighten_spinal_interface,
        fill_small_unclassified_fragments,
        VENTRICLE_LABELS
    )

    cfg = config or SegmentationConfig()
    seg = get_img(seg_path)
    data = np.ascontiguousarray(seg.get_fdata().astype(np.uint8))
    assert np.shares_memory(data, seg.get_fdata()) == False
    print(f"{(data==Label.CSF).sum() * 0.5**3 *1e-3} ml CSF ")
    data = solidify_csf(data, **asdict(cfg.solidify_csf))
    print(f"{(data==Label.CSF).sum() * 0.5**3 *1e-3} ml CSF ")
    data = close_csf_space(data, **asdict(cfg.close_csf_space))
    #print(f"{(data==Label.CSF).sum() * 0.5**3 *1e-3} ml CSF ")
    data = fill_wm_hyperintensities(data)
    data = cut_bottom(data, **asdict(cfg.cut_bottom))
    print(f"{(data==Label.CSF).sum() * 0.5**3 *1e-3} ml CSF ")
    data = extend_brainstem(data, **asdict(cfg.extend_brainstem))
    data = enforce_csf_layer(data, **asdict(cfg.enforce_csf_layer_pre))
    print(f"{(data==Label.CSF).sum() * 0.5**3 *1e-3} ml CSF ")

    orig_mask = nbm.smooth_labels_spherical(data > 0,
                                            radius=cfg.misc.original_mask_smoothing_radius)
    if cfg.misc.apply_mode_box_pre:
        data = nbm.mode_box(data)
        data[~orig_mask] = 0
    assert data.dtype == np.uint8 
    data = create_falx(data, **asdict(cfg.falx))
    data = create_tentorium(data, **asdict(cfg.tentorium))
    # connect (often disconnected) inf. lateral ventricle horns with
    # lateral ventricles
    data = build_inferior_lateral_ventricle_horns(data, **asdict(cfg.inf_lat_vent_horns))
    # make sure the V4 is not too thin
    data = enforce_min_thickness(data, Label.FOURTH_VENTRICLE,
                                 radius=cfg.min_thickness_v4.radius)
    # make sure the ventricles are correctly connected!
    data = enforce_connected_ventricles(data, **asdict(cfg.connected_ventricles))
    # add a thin layer of tissue around the ventricles,
    # to unphysiological connections
    data = enforce_tight_ventricles(data, **asdict(cfg.tight_ventricles))

    data[~orig_mask] = 0

    if cfg.misc.apply_mode_box_post:
        data = nbm.mode_box(data)
    if cfg.misc.apply_mode_diamond_post:
        data = nbm.mode_diamond(data)
    data[~orig_mask] = 0
    assert data.dtype == np.uint8
    data = enforce_csf_around_tentorium(data, **asdict(cfg.csf_around_tentorium))
    data = enforce_csf_around_falx(data, **asdict(cfg.csf_around_falx))
    data = enforce_csf_layer(data, **asdict(cfg.enforce_csf_layer_post))
    data = extend_brainstem_caudally(data, **asdict(cfg.extend_brainstem_caudally))

    data = fill_small_unclassified_fragments(data, size=100)

    print(f"{(data==Label.CSF).sum() * 0.5**3 *1e-3} ml CSF/SAS")
    print(f"{(np.isin(data, VENTRICLE_LABELS)).sum() * 0.5**3 *1e-3} ml Ventr.")

    assert data.dtype == np.uint8
    seg_out = nib.Nifti1Image(data, seg.affine)
    # We still need the grid for surface extraction
    grid = nibabel_to_pyvista(seg_out)
    grid["data"] = data.flatten(order="F")

    # Only save segmentation outputs if requested
    if out_seg is not None:
        out_seg_path = pathlib.Path(out_seg)
        out_seg_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(seg_out, out_seg_path)
        # Dynamically extract base name (handling .nii.gz double extensions safely)
        seg_basename = out_seg_path.name.split('.')[0]
        out_vti_path = out_seg_path.parent / f"{seg_basename}.vti"
        save_mesh(grid, out_vti_path)

    # Extract surface
    surf = grid.contour_labels("all", smoothing=True)
    #surf = straighten_spinal_interface(surf, grid)

    # Only save surface outputs if requested
    if out_surf is not None:
        out_surf_path = pathlib.Path(out_surf)
        grid["data"] = seg.get_fdata().astype(np.uint8).flatten(order="F")
        origsurf = grid.contour_labels("all", smoothing=True)
        out_surf_orig_path = out_surf_path.parent / f"{out_surf_path.stem}_orig{out_surf_path.suffix}"
        save_mesh(origsurf, out_surf_orig_path)

        save_mesh(surf, out_surf_path)

        # Save the decimated surface, appending '_dec' to the provided surface filename
        surf_dec = coarsen_surface(surf, **asdict(cfg.coarsen_surface))
        out_surf_dec_path = out_surf_path.parent / f"{out_surf_path.stem}_dec{out_surf_path.suffix}"
        save_mesh(surf_dec, out_surf_dec_path)
    
    return surf

def surface_to_mesh(surf_path, out_file=None, **tetwild_kwargs):
    """
    Tetrahedralise a surface and mark each cell with its anatomical label.

    Parameters
    ----------
    surf_path    : str or Path  path to a .vtk surface with boundary_labels
    out_dir      : str or Path  destination folder
    **tetwild_kwargs : forwarded to pytetwild.tetrahedralize_pv
    """
    import pytetwild

    from brainmesh import mark_mesh, read_mesh, save_mesh

    surf = read_mesh(surf_path)

    twild_defaults = dict(
        stop_energy=10,
        loglevel=5,
        quiet=False,
        disable_filtering=True,
        edge_length_fac=0.05,
        epsilon=1e-3,
        coarsen=False,
        num_threads=0
    )
    twild_defaults.update(tetwild_kwargs)

    mesh = pytetwild.tetrahedralize_pv(surf, **twild_defaults)
    mesh = mark_mesh(mesh, surf)
    if "grid_z_normal" in surf.field_data.keys():
        mesh.field_data["grid_z_normal"] = surf.field_data["grid_z_normal"]
    save_mesh(mesh, out_file)
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
    excluded_mask = np.isin(seg, excluded_labels) | np.isin(parc, excluded_labels)
    parc[excluded_mask] = 0
    assert (parc==Label.CSF).sum() == 0

    # grow seeds into SAS
    region_mask = sas_mask + (parc >0) 
    labeled_SAS = grow_into_region(parc, region_mask, radius=1)
    # .. and keep only CSF
    labeled_SAS[~sas_mask] = 0
    labeled_SAS = np.where(np.isin(seg, VENTRICLE_LABELS), seg, labeled_SAS)

    labels, counts = np.unique(labeled_SAS, return_counts=True)
    sort_idx = np.argsort(-counts)
    for l, c in zip(labels[sort_idx], counts[sort_idx]):
        print(f"  - {c:<8} label: {reverse_label_map.get(l, l)}")
    return nib.Nifti1Image(labeled_SAS, parc_img.affine)
