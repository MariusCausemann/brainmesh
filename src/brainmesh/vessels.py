"""Vessel lumens and open perivascular (CSF) sleeves from centerlines and/or masks.

Both vessel sources are reduced to one representation: a signed distance field
``s`` (mm, world frame) on the segmentation grid, negative inside the lumen.

* centerlines (``.vtp`` polylines with a point-data ``radius`` array, in mm):
  ``s`` is the exact distance to the union of the per-segment cone capsules
  (radius linearly interpolated along each segment), minus that radius.
* a vessel mask (``.nii``, any grid): resampled onto the segmentation grid and
  converted to a signed Euclidean distance transform.

Several sources are combined by the pointwise minimum (union of lumens). The
lumen is ``s <= 0`` and the sleeve ``0 < s <= sleeve_thickness``; see
:func:`enforce_vessel_sleeve`.
"""

import warnings

import numpy as np
from numba import njit

from .decorators import time_func
from .labels import VENTRICLE_LABELS, Label


def read_centerlines(path, segments=(), min_radius=0.0, radius_scale=1.0, radius_floor=0.0):
    """
    Read vessel centerlines as a list of straight segments.

    Parameters
    ----------
    path         : str or Path   polyline ``.vtp`` with point data ``radius`` (mm)
    segments     : sequence      keep only branches whose cell-data ``segment``
                                 label is listed, either by id or by name from
                                 the field data ``segment_names``; empty keeps all
    min_radius   : float         drop segments whose mean radius (mm) is below this
    radius_scale : float         multiply all radii by this factor
    radius_floor : float         clamp the (scaled) radii to at least this (mm)

    Returns
    -------
    p0, p1 : (n, 3) float64  segment endpoints (world mm)
    r0, r1 : (n,)   float64  radii at the endpoints (mm)
    """
    import pyvista as pv

    poly = pv.read(path)
    if "radius" not in poly.point_data:
        raise ValueError(f"{path}: centerlines need a point-data 'radius' array")
    radius = np.asarray(poly.point_data["radius"], dtype=np.float64)
    keep = _segment_filter(poly, segments)

    lines = poly.lines
    i0, i1 = [], []
    cell, offset = 0, 0
    while offset < len(lines):
        n = lines[offset]
        ids = lines[offset + 1 : offset + 1 + n]
        if keep[cell]:
            i0.append(ids[:-1])
            i1.append(ids[1:])
        offset += n + 1
        cell += 1
    if not i0:
        empty = np.zeros((0, 3))
        return empty, empty, np.zeros(0), np.zeros(0)
    i0, i1 = np.concatenate(i0), np.concatenate(i1)

    pts = np.asarray(poly.points, dtype=np.float64)
    p0, p1 = pts[i0], pts[i1]
    r0 = np.maximum(radius[i0] * radius_scale, radius_floor)
    r1 = np.maximum(radius[i1] * radius_scale, radius_floor)
    # duplicated bifurcation points give zero-length segments
    ok = (np.linalg.norm(p1 - p0, axis=1) > 0) & (0.5 * (r0 + r1) >= min_radius)
    return p0[ok], p1[ok], r0[ok], r1[ok]


def _segment_filter(poly, segments):
    """Boolean per-cell mask of the branches whose ``segment`` label is selected."""
    if len(segments) == 0:
        return np.ones(poly.n_cells, dtype=bool)
    if "segment" not in poly.cell_data:
        raise ValueError("segment filtering needs a cell-data 'segment' array")
    names = [str(n) for n in poly.field_data.get("segment_names", [])]
    ids = set()
    for s in segments:
        if isinstance(s, str):
            if s not in names:
                raise ValueError(f"unknown vessel segment {s!r}; known: {names}")
            ids.add(names.index(s))
        else:
            ids.add(int(s))
    return np.isin(np.asarray(poly.cell_data["segment"]), list(ids))


@njit(cache=True)
def _capsule_distance(dist, affine, inv_lin, inv_off, p0, p1, r0, r1, band, pad_scale):
    """Min-accumulate ``|x - p(t)| - r(t)`` over segments into ``dist`` (in place)."""
    shape = dist.shape
    for s in range(p0.shape[0]):
        d = p1[s] - p0[s]
        o = affine[:3, 3] - p0[s]
        dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        pad = (max(r0[s], r1[s]) + band) * pad_scale
        ia = inv_lin @ p0[s] + inv_off
        ib = inv_lin @ p1[s] + inv_off
        lo = np.empty(3, dtype=np.int64)
        hi = np.empty(3, dtype=np.int64)
        for c in range(3):
            lo[c] = max(int(np.floor(min(ia[c], ib[c]) - pad)), 0)
            hi[c] = min(int(np.ceil(max(ia[c], ib[c]) + pad)), shape[c] - 1)
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    # voxel centre relative to the segment start (world mm)
                    x0 = affine[0, 0] * i + affine[0, 1] * j + affine[0, 2] * k + o[0]
                    x1 = affine[1, 0] * i + affine[1, 1] * j + affine[1, 2] * k + o[1]
                    x2 = affine[2, 0] * i + affine[2, 1] * j + affine[2, 2] * k + o[2]
                    t = 0.0
                    if dd > 0:
                        t = min(max((x0 * d[0] + x1 * d[1] + x2 * d[2]) / dd, 0.0), 1.0)
                    q0, q1, q2 = x0 - t * d[0], x1 - t * d[1], x2 - t * d[2]
                    v = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2) - ((1.0 - t) * r0[s] + t * r1[s])
                    if v < dist[i, j, k]:
                        dist[i, j, k] = v
    return dist


@time_func
def centerline_distance(shape, affine, p0, p1, r0, r1, band):
    """
    Signed distance (mm) to the lumen of a set of tapered centerline segments.

    Only voxels within ``band`` mm of a lumen are evaluated; all others are
    ``+inf``. ``affine`` maps voxel indices to world mm (oblique affines and
    anisotropic voxels are handled exactly).
    """
    affine = np.asarray(affine, dtype=np.float64)
    dist = np.full(shape, np.inf, dtype=np.float32)
    if len(p0) == 0:
        return dist
    # a ball of radius rho mm spans at most rho / sigma_min voxels along any index axis
    pad_scale = 1.0 / np.linalg.svd(affine[:3, :3], compute_uv=False).min()
    inv = np.linalg.inv(affine)
    return _capsule_distance(
        dist,
        affine,
        np.ascontiguousarray(inv[:3, :3]),
        np.ascontiguousarray(inv[:3, 3]),
        np.ascontiguousarray(p0, dtype=np.float64),
        np.ascontiguousarray(p1, dtype=np.float64),
        np.ascontiguousarray(r0, dtype=np.float64),
        np.ascontiguousarray(r1, dtype=np.float64),
        float(band),
        pad_scale,
    )


@time_func
def mask_distance(mask_img, seg_img, labels=()):
    """
    Signed distance (mm) to the lumen given by a vessel mask image.

    The mask (``labels`` selected, or all ``> 0`` if empty) is resampled onto
    the grid of ``seg_img`` (trilinear on the binary mask, thresholded at 0.5)
    and converted to a signed Euclidean distance transform. Distances are
    measured between voxel centres and shifted by half a voxel, so the zero
    level sits on the voxel faces of the resampled mask.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from scipy.ndimage import distance_transform_edt

    from .io import get_img

    mask_img = get_img(mask_img)
    raw = np.asarray(mask_img.dataobj)
    sel = np.isin(raw, list(labels)) if len(labels) else raw > 0
    sel_img = nib.Nifti1Image(sel.astype(np.float32), mask_img.affine)
    res = resample_from_to(sel_img, (seg_img.shape[:3], seg_img.affine), order=1)
    lumen = np.asarray(res.dataobj) >= 0.5

    zooms = np.asarray(seg_img.header.get_zooms()[:3], dtype=np.float64)
    half = 0.5 * zooms.mean()
    if not lumen.any():
        return np.full(lumen.shape, np.inf, dtype=np.float32)
    outside = distance_transform_edt(~lumen, sampling=zooms)
    inside = distance_transform_edt(lumen, sampling=zooms)
    return np.where(lumen, half - inside, outside - half).astype(np.float32)


def vessel_distance(seg_img, centerlines=None, mask=None, cfg=None):
    """
    Signed lumen distance (mm) on the grid of ``seg_img`` from any combination
    of vessel sources; ``None`` if no source is given.

    Parameters
    ----------
    seg_img     : nib.Nifti1Image   (canonical) segmentation defining the grid
    centerlines : str or Path, optional  ``.vtp`` centerlines with radii
    mask        : str, Path or nib image, optional  vessel mask
    cfg         : VesselCfg, optional
    """
    from .config import VesselCfg

    if centerlines is None and mask is None:
        return None
    cfg = cfg or VesselCfg()
    zooms = np.asarray(seg_img.header.get_zooms()[:3], dtype=np.float64)
    band = cfg.sleeve_thickness + zooms.max()
    dist = np.full(seg_img.shape[:3], np.inf, dtype=np.float32)
    if centerlines is not None:
        p0, p1, r0, r1 = read_centerlines(
            centerlines,
            segments=cfg.segments,
            min_radius=cfg.min_radius,
            radius_scale=cfg.radius_scale,
            radius_floor=cfg.radius_floor,
        )
        print(f"{len(p0)} centerline segments, {np.linalg.norm(p1 - p0, axis=1).sum():.0f} mm")
        dist = np.minimum(
            dist, centerline_distance(dist.shape, seg_img.affine, p0, p1, r0, r1, band)
        )
    if mask is not None:
        dist = np.minimum(dist, mask_distance(mask, seg_img, labels=cfg.mask_labels))
    return dist


VESSEL_PROTECTED_LABELS = (Label.FALX, Label.TENTORIUM, *VENTRICLE_LABELS)


@time_func
def enforce_vessel_sleeve(
    data,
    dist,
    sleeve_thickness=0.5,
    subdomain=False,
    min_csf_voxels=1,
    min_lumen_voxels=27,
    voxel_size=None,
    protected=VESSEL_PROTECTED_LABELS,
    ventricle_clearance=3,
):
    """
    Write vessel lumens and their perivascular CSF sleeves into ``data``.

    * sleeve (``0 < dist <= sleeve_thickness``) -> ``Label.CSF``
    * lumen (``dist <= 0``) -> ``Label.VESSEL`` if ``subdomain`` else ``Label.CSF``

    The sleeve replaces tissue as well as CSF (open PVS through parenchyma),
    but the domain never grows (``data == 0`` stays background) and the
    ``protected`` labels (falx, tentorium, ventricles) are never overwritten.
    Neither is anything within ``ventricle_clearance`` voxels of a ventricle or
    the choroid plexus: that is the tissue jacket of
    ``enforce_tight_ventricles``, and a sleeve through it would open a
    ventricle to the SAS (the only real openings are the V4 foramina, which
    already are open).
    With ``subdomain``, lumen fragments smaller than ``min_lumen_voxels``
    (26-connected; e.g. cut off where a vessel crosses a protected membrane)
    become CSF instead of tiny subdomains, and a box jacket of
    ``min_csf_voxels`` CSF voxels around the lumen is enforced on top, so a
    lumen never touches tissue even where the sleeve is thinner than the voxels.

    Parameters
    ----------
    data             : uint8 array     segmentation (modified in place)
    dist             : float array     signed lumen distance (mm), same shape
    sleeve_thickness : float           sleeve thickness (mm)
    subdomain        : bool            keep the lumen as its own label
    min_csf_voxels   : int             CSF jacket around the lumen (voxels, subdomain only)
    min_lumen_voxels : int             smallest lumen fragment kept (voxels, subdomain only)
    voxel_size       : sequence, optional  voxel size (mm), only used for a warning
    protected        : sequence        labels the vessels never overwrite
    ventricle_clearance : int          keep-out distance (voxels) around the
                                       ventricles and choroid plexus
    """
    from cc3d import dust
    from nbmorph import dilate_labels_spherical as dilate

    from .segmentation import set_mask_scalar

    if voxel_size is not None and sleeve_thickness < min(voxel_size):
        warnings.warn(
            f"vessel sleeve ({sleeve_thickness} mm) is thinner than a voxel "
            f"({min(voxel_size):.2f} mm); it will be broken or missing in places",
            stacklevel=2,
        )
    writable = (data > 0) & ~np.isin(data, protected)
    ventricles = np.isin(data, VENTRICLE_LABELS).view(np.uint8)
    if ventricle_clearance > 0:
        # the box covers the (default, smaller) structuring element of the jacket
        writable &= dilate(ventricles, radius=ventricle_clearance, struct_sequence="B") == 0
    lumen = (dist <= 0) & writable
    sleeve = (dist > 0) & (dist <= sleeve_thickness) & writable
    set_mask_scalar(data, sleeve, Label.CSF)
    if not subdomain:
        return set_mask_scalar(data, lumen, Label.CSF)
    if min_csf_voxels > 0 and ventricle_clearance > 0:
        # leave room for the CSF jacket outside the keep-out zone, so a VESSEL
        # lumen never touches the ventricle jacket; the rest of it becomes CSF
        margin = ventricle_clearance + min_csf_voxels
        near = dilate(ventricles, radius=margin, struct_sequence="B") > 0
        set_mask_scalar(data, lumen & near, Label.CSF)
        lumen &= ~near
    if min_lumen_voxels > 1:
        kept = dust(lumen, threshold=min_lumen_voxels, connectivity=26)
        set_mask_scalar(data, lumen & ~kept, Label.CSF)
        lumen = kept
    set_mask_scalar(data, lumen, Label.VESSEL)
    if min_csf_voxels > 0:
        jacket = dilate(lumen, radius=min_csf_voxels, struct_sequence="B") & ~lumen
        set_mask_scalar(data, jacket & writable & (data != Label.VESSEL), Label.CSF)
    return data
