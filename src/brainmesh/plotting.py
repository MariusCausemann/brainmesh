"""Diagnostic plotting for tet meshes and facet meshes (off-screen pyvista + matplotlib compose)."""
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pyvista as pv

from brainmesh.labels import (
    Label,
    SPINAL_ID,
    TISSUE_LABELS,
    VENTRICLE_LABELS,
    reverse_label_map,
)
from brainmesh.mesh import group_csf_facets_by_region

WINDOW = (1024, 1024)

# Standard FreeSurfer LUT colors (RGB 0-1) for solid-tissue + ventricle labels.
# Source: $FREESURFER_HOME/FreeSurferColorLUT.txt. LH/RH share colors per FS convention.
FREESURFER_COLORS = {
    # Cerebral white matter / cortex
    0:   (0, 0, 0, 0),                 # background
    2:   (245/255, 245/255, 245/255),  # LH WM
    41:  (245/255, 245/255, 245/255),  # RH WM
    3:   (205/255,  62/255,  78/255),  # LH cortex
    42:  (205/255,  62/255,  78/255),  # RH cortex
    77:  (200/255,  70/255, 255/255),  # WM hypointensities
    # Cerebellum
    7:   (220/255, 248/255, 164/255),  # LH cerebellum WM
    46:  (220/255, 248/255, 164/255),  # RH cerebellum WM
    8:   (230/255, 148/255,  34/255),  # LH cerebellum cortex
    47:  (230/255, 148/255,  34/255),  # RH cerebellum cortex
    # Subcortical
    10:  (  0/255, 118/255,  14/255),  # thalamus
    49:  (  0/255, 118/255,  14/255),
    11:  (122/255, 186/255, 220/255),  # caudate
    50:  (122/255, 186/255, 220/255),
    12:  (236/255,  13/255, 176/255),  # putamen
    51:  (236/255,  13/255, 176/255),
    13:  ( 12/255,  48/255, 255/255),  # pallidum
    52:  ( 13/255,  48/255, 255/255),
    16:  (119/255, 159/255, 176/255),  # brainstem
    17:  (220/255, 216/255,  20/255),  # hippocampus
    53:  (220/255, 216/255,  20/255),
    18:  (103/255, 255/255, 255/255),  # amygdala
    54:  (103/255, 255/255, 255/255),
    26:  (255/255, 165/255,   0/255),  # accumbens
    58:  (255/255, 165/255,   0/255),
    28:  (165/255,  42/255,  42/255),  # ventral DC
    60:  (165/255,  42/255,  42/255),
    # Ventricles + choroid plexus (FS standard, used for panel 1)
    4:   (120/255,  18/255, 134/255),
    43:  (120/255,  18/255, 134/255),
    5:   (196/255,  58/255, 250/255),
    44:  (196/255,  58/255, 250/255),
    14:  (204/255, 182/255, 142/255),
    15:  ( 42/255, 204/255, 164/255),
    31:  (  0/255, 197/255, 255/255),
    63:  (  0/255, 197/255, 255/255),
    # brainmesh-specific (no FS entry — use anatomically reasonable picks)
    70:  (250/255, 128/255, 114/255),  # FALX  (salmon)
    71:  (240/255, 230/255, 140/255),  # TENTORIUM (khaki)
    72:  (255/255, 255/255, 255/255),  # UNCLASSIFIED (black)
    73:  (255/255, 255/255, 255/255),  # SPINAL_BUFFER (black)

}

VENTRICLE_COLORS = {vid: FREESURFER_COLORS[vid] for vid in VENTRICLE_LABELS
                    if int(vid) in FREESURFER_COLORS}
VENTRICLE_COLORS = {int(k): v for k, v in VENTRICLE_COLORS.items()}

MENINGES_COLORS = {
    int(Label.FALX):      FREESURFER_COLORS[70],
    int(Label.TENTORIUM): FREESURFER_COLORS[71],
}

CSF_COLOR = "lightsteelblue"

def _trim_white(img, tol=5):
    """Crop pure-white border around a rendered RGB image."""
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    mask = (img[..., :3] < (255 - tol)).any(axis=2)
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _render(populate, cpos=None, window=WINDOW, bg="white",
            parallel=True, fit=True, zoom=1.2):
    """populate(plotter) adds actors; configure camera, snapshot, return RGB ndarray.

    cpos:
      - None              → isometric, auto-fit
      - str ("xy"/"yz"/…) → standard plane view, auto-fits
      - tuple of 3 tuples → explicit (position, focal, viewup); fit=True calls reset_camera
    """
    p = pv.Plotter(off_screen=True, window_size=window)
    p.background_color = bg
    populate(p)
    if parallel:
        p.enable_parallel_projection()
    if cpos is None:
        p.view_isometric()
    elif isinstance(cpos, str):
        p.camera_position = cpos
    else:
        p.camera_position = cpos
        if fit:
            p.reset_camera()
    if zoom != 1.0:
        p.camera.zoom(zoom)
    #p.camera.azimuth -= 20.0
    p.camera.elevation -= 20.0
    img = p.screenshot(return_img=True, transparent_background=False)
    p.close()
    return img


def _clip_closed_surface(poly, normal, origin, tolerance=1e-6):
    """vtkClipClosedSurface without pyvista's `n_open_edges > 0` guard.

    The underlying VTK filter still produces a usable result on slightly
    non-manifold inputs (which is common after `extract_surface()` on an
    interface-rich label set), so we bypass the eager Python-side check.
    """
    import vtk
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)
    collection = vtk.vtkPlaneCollection()
    collection.AddItem(plane)
    alg = vtk.vtkClipClosedSurface()
    alg.SetGenerateFaces(True)
    alg.SetInputDataObject(poly)
    alg.SetTolerance(tolerance)
    alg.SetClippingPlanes(collection)
    alg.Update()
    return pv.wrap(alg.GetOutput())


def _add_categorical(p, mesh, scalars_arr, lookup, cmap, **kwargs):
    """add_mesh with each unique value mapped to its own color via `lookup`."""
    idx = np.searchsorted(lookup, scalars_arr).astype(np.int32)
    p.add_mesh(mesh, scalars=idx, cmap=cmap,
               clim=(-0.5, len(lookup) - 0.5),
               show_scalar_bar=False, **kwargs)


def _compose(panels, output, ncols, legend=None,
             suptitle=None, legend_bbox_y=0):
    """panels: list of (img, title). legend: optional list of (label, color)."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.02, "hspace": 0.10})
    flat = axes.ravel()
    for ax, (img, title) in zip(flat, panels):
        ax.imshow(_trim_white(img))
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        ax.set_box_aspect(1)
    for ax in flat[len(panels):]:
        ax.set_axis_off()
        ax.set_box_aspect(1)
    if legend:
        handles = [Patch(facecolor=c, edgecolor="black", label=l) for l, c in legend]
        ncol = min(len(legend), 4)
        fig.legend(handles=handles, loc="lower center", ncol=ncol,
                   frameon=False, bbox_to_anchor=(0.5, legend_bbox_y), fontsize=9,
                   columnspacing=1.2, handlelength=1.2, handleheight=1.1,
                   handletextpad=0.5)
        fig.subplots_adjust(bottom=0.05)
    if suptitle: fig.suptitle(suptitle, y=0.95)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _orthogonal_cpos(bounds, axis, sign=+1, d_factor=2.5):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    d = max(xmax - xmin, ymax - ymin, zmax - zmin) * d_factor
    if axis == "x":
        return [(cx + sign * d, cy, cz), (cx, cy, cz), (0, 0, 1)]
    if axis == "y":
        return [(cx, cy + sign * d, cz), (cx, cy, cz), (0, 0, 1)]
    if axis == "z":
        return [(cx, cy, cz + sign * d), (cx, cy, cz), (0, 1, 0)]
    raise ValueError(f"unknown axis {axis!r}")


def _anatomical_views(bounds, d_factor=2.5):
    return [
        ("Anterior",  _orthogonal_cpos(bounds, "y", +1, d_factor)),
        ("Posterior", _orthogonal_cpos(bounds, "y", -1, d_factor)),
        ("Right",     _orthogonal_cpos(bounds, "x", +1, d_factor)),
        ("Superior",  _orthogonal_cpos(bounds, "z", +1, d_factor)),
        ("Inferior",  _orthogonal_cpos(bounds, "z", -1, d_factor)),
        ("Left",      _orthogonal_cpos(bounds, "x", -1, d_factor)),
    ]


def _extract_by_marker(mesh, label_array, values):
    arr = np.asarray(mesh.cell_data[label_array])
    return mesh.extract_cells(np.isin(arr, values))


def plot_tet_mesh(mesh, output, label_array="marker"):
    """Render a 6-panel diagnostic figure of a marked tetrahedral mesh."""
    if isinstance(mesh, pv.ImageData):
        suptitle = (f"#voxels: {mesh.n_cells:,}, dims: {tuple(mesh.dimensions)}, "
                    f"spacing: {tuple(round(s, 3) for s in mesh.spacing)}")
        mesh = mesh.extract_cells(mesh.cell_data[label_array] > 0)
    markers = np.asarray(mesh.cell_data[label_array])

    # Shared LUT so the same marker → same color across tissue & clip panels.
    # CSF / SAS are rendered separately in CSF_COLOR, so exclude them from the LUT.
    csf_mask_full = (markers == int(Label.CSF)) | (markers >= 10000)
    solid_unique = np.unique(markers[~csf_mask_full])
    palette = [FREESURFER_COLORS.get(int(m), (0.5, 0.5, 0.5, 0)) for m in solid_unique]

    cmap = ListedColormap(palette)

    panels = []

    # Solid tissue (everything except CSF / ventricles / SAS subdivisions)
    def tissue_cb(p):
        sub = _extract_by_marker(mesh, label_array, TISSUE_LABELS)
        if sub.n_cells:
            arr = np.asarray(sub.cell_data[label_array])
            _add_categorical(p, sub, arr, solid_unique, cmap, specular=0.3)
            #p.add_mesh(sub.extract_all_edges(), line_width=0.1, color="lightgrey")

    panels.append((_render(tissue_cb), "Parenchyma (with falx & tentorium)"))


    # Falx + tentorium, with faint context outline
    def men_cb(p):
        outline = mesh.extract_surface(algorithm="dataset_surface")
        p.add_mesh(outline, color="lightgrey", opacity=0.08, show_edges=False)
        for mid, color in MENINGES_COLORS.items():
            sub = _extract_by_marker(mesh, label_array, [mid])
            if sub.n_cells:
                p.add_mesh(sub, color=color, specular=0.3)

    panels.append((_render(men_cb), "Falx & tentorium"))

    #  Ventricular system, one color per label
    present_vent = [v for v in VENTRICLE_COLORS if (markers == v).any()]

    def vent_cb(p):
        for vid in present_vent:
            sub = _extract_by_marker(mesh, label_array, [vid])
            if sub.n_cells:
                p.add_mesh(sub, color=VENTRICLE_COLORS[vid], specular=0.3,
                           smooth_shading=True, show_edges=mesh.celltypes[0] in [10, 24],
                           edge_color="lightgrey")


    panels.append((_render(vent_cb), "Ventricular system"))

    # ── Row 2: orthogonal mid-plane clips (kept half + camera on cut-face side) ─
    clip_specs = [
        ("Sagittal clip",   "x", _orthogonal_cpos(mesh.bounds, "x", -1)),
        ("Coronal clip", "y", _orthogonal_cpos(mesh.bounds, "y", -1)),
        ("Axial clip",   "z", _orthogonal_cpos(mesh.bounds, "z", 1)),
    ]
    bnds = np.array(mesh.bounds).reshape(3, 2)
    origin = bnds[:, 0] + np.array([0.5, 0.45, 0.62]) * (bnds[:, 1] - bnds[:, 0])
    for title, normal, cpos in clip_specs:
        # invert=False keeps the +normal half; camera at sign=-1 sits on the
        # cut-face side, so we look INTO the kept half and see the cross-section.
        clipped = mesh.clip(normal=normal, origin=origin, invert=normal=="z")

        def clip_cb(p, clipped=clipped, label_array=label_array):
            arr = np.asarray(clipped.cell_data[label_array])
            csf_mask = (arr == int(Label.CSF)) | (arr >= 10000)
            csf_part = clipped.extract_cells(csf_mask)
            solid_part = clipped.extract_cells(~csf_mask)
            if solid_part.n_cells:
                solid_arr = np.asarray(solid_part.cell_data[label_array])
                _add_categorical(p, solid_part, solid_arr, solid_unique, cmap)
            if csf_part.n_cells:
                p.add_mesh(csf_part, color=CSF_COLOR, show_scalar_bar=False)

        panels.append((_render(clip_cb, cpos=cpos), title))

    legend = [(reverse_label_map.get(vid, str(vid)), FREESURFER_COLORS[vid])
              for vid in set(np.unique(markers)) - set([0, Label.CSF])]

    if csf_mask_full.any():
        legend.append(("CSF / SAS", CSF_COLOR))

    if mesh.celltypes[0] in [10, 24]:
        if mesh.celltypes[0] == 24:
            num_corner_points = np.unique(mesh.cells.reshape(-1, 11)[:, 1:5]).size
            cell_type = "quadratic tetrahedra"  
        else : 
            num_corner_points = mesh.n_points
            cell_type ="linear tetrahedra"
        suptitle = f"#cells: {mesh.n_cells:,}, #vertices: {num_corner_points:,}, cell type: {cell_type}"
    _compose(panels, output, ncols=3, legend=legend, suptitle=suptitle, legend_bbox_y=-0.18)


def plot_surface_mesh(surf, output, label_array="boundary_labels"):
    """Render a 6-panel diagnostic figure of a multi-label surface mesh.

    Mirrors :func:`plot_tet_mesh` but operates on the multi-label surface that
    the tet mesh was generated from (``boundary_labels`` cell data, shape
    ``(n_cells, 2)``: outside / inside label per triangle). The bottom-row
    cross-section panels use :meth:`pyvista.PolyData.clip_closed_surface` per
    label so each clipped region is a capped solid.
    """
    blabels = np.asarray(surf.cell_data[label_array])
    if blabels.ndim != 2 or blabels.shape[1] != 2:
        raise ValueError(
            f"{label_array!r} must be (n_cells, 2). Got shape {blabels.shape}."
        )
    blabels_a, blabels_b = blabels[:, 0], blabels[:, 1]

    def _label_mask(cid):
        return (blabels_a == cid) | (blabels_b == cid)

    F_global = surf.faces.reshape(-1, 4)[:, 1:]  # (n_cells, 3) triangle connectivity

    def _extract(cid):
        """Closed sub-surface bounding region cid, with normals oriented OUTWARD.

        Mirrors :func:`brainmesh.mesh.mark_mesh`: triangles where cid is on the
        inner side (column 1 of boundary_labels) get their winding flipped so
        every face normal points out of cid. Required for clip_closed_surface
        to cap correctly on slightly non-manifold inputs.
        """
        out_mask = blabels_a == cid
        in_mask = blabels_b == cid
        if not (out_mask.any() or in_mask.any()):
            return None
        F_out = F_global[out_mask]
        F_in = F_global[in_mask]
        if len(F_in) > 0:
            F_in = F_in[:, [0, 2, 1]]  # swap last two verts → flip normal
            F = np.vstack([F_out, F_in])
        else:
            F = F_out
        if len(F) == 0:
            return None
        n = len(F)
        cells_flat = np.column_stack([np.full(n, 3, dtype=np.int64), F]).ravel()
        return pv.PolyData(surf.points, faces=cells_flat)

    all_labels = np.unique(np.concatenate([blabels_a, blabels_b]))
    all_labels = all_labels[all_labels > 0]
    tissue_set = {int(t) for t in TISSUE_LABELS}

    panels = []

    # Parenchyma (with falx & tentorium) — solid-tissue surfaces, FS colors
    def tissue_cb(p):
        for cid in all_labels:
            if int(cid) not in tissue_set:
                continue
            sub = _extract(cid)
            if sub is not None and sub.n_cells:
                p.add_mesh(sub, color=FREESURFER_COLORS.get(int(cid), (0.5, 0.5, 0.5)),
                           specular=0.3)

    panels.append((_render(tissue_cb), "Parenchyma (with falx & tentorium)"))

    # Falx + tentorium with faint OUTER-surface context (skip inner interfaces)
    outer_mask = (blabels_a == 0) | (blabels_b == 0)  # outer skin: background on either side
    outer_outline = (surf.extract_cells(outer_mask).extract_surface(algorithm="dataset_surface")
                     if outer_mask.any() else None)

    def men_cb(p):
        if outer_outline is not None and outer_outline.n_cells:
            p.add_mesh(outer_outline, color="lightgrey", opacity=0.08, show_edges=False)
        for mid in (int(Label.FALX), int(Label.TENTORIUM)):
            sub = _extract(mid)
            if sub is not None and sub.n_cells:
                p.add_mesh(sub, color=FREESURFER_COLORS[mid], specular=0.3)

    panels.append((_render(men_cb), "Falx & tentorium"))

    # Ventricular system, one color per label (no edges)
    present_vent = [int(v) for v in VENTRICLE_LABELS if _label_mask(v).any()]

    def vent_cb(p):
        for vid in present_vent:
            sub = _extract(vid)
            if sub is not None and sub.n_cells:
                p.add_mesh(sub, color=VENTRICLE_COLORS[vid], specular=0.3,
                           smooth_shading=True)

    panels.append((_render(vent_cb), "Ventricular system"))

    # ── Row 2: clip_closed_surface cross-sections, per-label ─────────────────
    bnds = np.array(surf.bounds).reshape(3, 2)
    origin = bnds[:, 0] + np.array([0.5, 0.45, 0.62]) * (bnds[:, 1] - bnds[:, 0])
    # Normal vector → direction "kept" by clip_closed_surface (the side the
    # normal points to). Mirror plot_tet_mesh's convention: keep right /
    # anterior / inferior so the camera (placed on the cut-face side) looks
    # at the cross-section.
    clip_specs = [
        ("Sagittal clip", (1.0, 0.0, 0.0),  _orthogonal_cpos(surf.bounds, "x", -1)),
        ("Coronal clip",  (0.0, 1.0, 0.0),  _orthogonal_cpos(surf.bounds, "y", -1)),
        ("Axial clip",    (0.0, 0.0, -1.0), _orthogonal_cpos(surf.bounds, "z", +1)),
    ]

    def _clip_color(cid_int):
        if cid_int == int(Label.CSF) or cid_int >= 10000:
            return CSF_COLOR
        return FREESURFER_COLORS.get(cid_int, (0.5, 0.5, 0.5))

    for title, normal, cpos in clip_specs:
        def clip_cb(p, normal=normal):
            for cid in all_labels:
                cid_int = int(cid)
                sub = _extract(cid)
                if sub is None or sub.n_cells == 0:
                    continue
                clipped = _clip_closed_surface(sub, normal=normal, origin=origin)
                if clipped.n_cells == 0:
                    continue
                p.add_mesh(clipped, color=_clip_color(cid_int), specular=0.2)

        panels.append((_render(clip_cb, cpos=cpos), title))

    # Legend: every present label that has an FS color, plus an aggregated CSF entry
    present = set(np.unique(np.concatenate([blabels_a, blabels_b])).tolist()) - {0, int(Label.CSF)}
    legend = [(reverse_label_map.get(int(l), str(int(l))), FREESURFER_COLORS[int(l)])
              for l in sorted(present) if int(l) in FREESURFER_COLORS]
    if (((blabels_a == int(Label.CSF)) | (blabels_b == int(Label.CSF))).any()
            or ((blabels_a >= 10000) | (blabels_b >= 10000)).any()):
        legend.append(("CSF / SAS", CSF_COLOR))

    suptitle = f"#triangles: {surf.n_cells:,}, #vertices: {surf.n_points:,}"
    _compose(panels, output, ncols=3, legend=legend, suptitle=suptitle, legend_bbox_y=-0.18)


def plot_facet_mesh(facets, output, group=True, region_labels=None):
    """Render a 6-panel diagnostic figure of a facet mesh.

    region_labels: optional {name: id} mapping (e.g. loaded from the
    ``*_labels.toml`` written by ``brainmesh-group-regions``). Used to build
    the legend when ``group=True``.
    """
    if group and "region" not in facets.cell_data:
        facets, region_labels = group_csf_facets_by_region(facets)

    facets = facets.extract_cells(facets.cell_data["interface_id"] < 100000)
    panels = []

    if group:
        marker = np.asarray(facets.cell_data["region"])
    else:
        marker = np.asarray(facets.cell_data["interface_id"])

    marker_unique = np.unique(marker)
    base = plt.get_cmap("tab20")
    cmap = ListedColormap([base(i % base.N) for i in range(len(marker_unique))])

    def grouped_cb(p, facets=facets, marker=marker):
        _add_categorical(p, facets, marker, marker_unique, cmap)

    for title, cpos in _anatomical_views(facets.bounds):
        panels.append((_render(grouped_cb, cpos=cpos), title))

    legend = None
    if group and region_labels:
        id_to_name = {int(v): str(k) for k, v in region_labels.items()}
        legend = [(id_to_name.get(int(m), str(int(m))), cmap(i))
                  for i, m in enumerate(marker_unique)]

    _compose(panels, output, ncols=3, legend=legend, legend_bbox_y=-0.1)
