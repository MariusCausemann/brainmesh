"""Brain-specific operations on marked tet meshes: CSF extraction, SAS and spinal facets."""
import numpy as np
import pyvista as pv
from imagemesh import facets as _facets
from imagemesh.facets import (
    _build_facet_polydata,
    _flip_winding,
    _raw_faces,
    dilate_cell_marker,
    mark_between_regions,
    mark_boundary_facets,
    remove_small_patches,
    smooth_cell_labels,
)
from imagemesh.image import get_img
from imagemesh.tetmesh import extract_reduced_facets, filter_by_mask, largest_face_connected

from .labels import (Label, VENTRICLE_LABELS, SAS_LABEL_OFFSET, SPINAL_ID,
                     is_csf_marker)


def mark_interface_facets(mesh, label_array="marker", encoding_base=1000,
                          ignore_sas_interfaces=True):
    """
    Interface facets between regions of different markers (see
    :func:`imagemesh.facets.mark_interface_facets`). With
    ``ignore_sas_interfaces`` (default), interfaces between two SAS subdivision
    labels (> ``SAS_LABEL_OFFSET``) are dropped.
    """
    return _facets.mark_interface_facets(
        mesh, label_array=label_array, encoding_base=encoding_base,
        ignore_above=SAS_LABEL_OFFSET if ignore_sas_interfaces else None,
    )


def remark_csf_with_sas(mesh, sas_img, csf_label=Label.CSF, label_array="marker"):
    """
    Replace ``csf_label`` markers on ``mesh`` with subdivided SAS labels sampled
    from ``sas_img`` (path or nib.Nifti1Image) at each tet centroid via the
    image's inverse affine.

    Tets that map outside the image or to a background (0) voxel keep their
    original CSF marker. Non-CSF tets are never touched.
    Edits ``mesh.cell_data[label_array]`` in place and returns ``mesh``.
    """
    from scipy.spatial import KDTree
    from imagemesh.image import nibabel_to_pyvista

    sas = nibabel_to_pyvista(get_img(sas_img))
    sas = sas.extract_cells(sas.cell_data["data"] > 0)
    if sas.n_cells == 0:
        return mesh
    kd_tree = KDTree(sas.cell_centers().points)
    mesh_cell_centers = mesh.cell_centers().points.astype(np.float32)
    _, nearest_idx = kd_tree.query(mesh_cell_centers)
    sas_marker = sas.cell_data["data"].astype(np.int32)
    mesh_marker = mesh.cell_data[label_array].copy()
    csf_mask = mesh_marker == csf_label
    mesh_marker[csf_mask] = sas_marker[nearest_idx][csf_mask] + SAS_LABEL_OFFSET
    mesh.cell_data[label_array] = mesh_marker
    return mesh


def extract_csf(mesh, label_array="marker", return_facets=False, **facet_kwargs):
    """
    Extract the CSF compartment — ``Label.CSF``, all ventricles and choroid plexus,
    and any SAS-subdivision markers (values > ``SAS_LABEL_OFFSET``); see
    :func:`brainmesh.labels.is_csf_marker`.

    ``Label.UNCLASSIFIED`` (vessels sitting in the SAS), ``Label.VESSEL`` and
    ``Label.SPINAL_BUFFER`` are *not* part of it: they stay solid, and their facets
    against the CSF become boundaries of the extracted submesh (the buffer ones as ``SPINAL_ID``).

    When ``return_facets=True``, facets are computed on the **full** mesh so that
    CSF-to-tissue interfaces carry their full ``interface_id`` encoding
    (e.g. ``min(CSF,WM)*100000+max(CSF,WM)``).  The returned facet mesh shares
    the CSF submesh's point array.

    Parameters
    ----------
    mesh          : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array   : str                  cell data array with region markers
    return_facets : bool                 if True, also return the CSF-relevant
                                         facets (same ``interface_id`` scheme as
                                         :func:`mark_facets`)
    **facet_kwargs                       forwarded to :func:`mark_facets`
                                         (e.g. ``encoding_base``)

    Returns
    -------
    csf_mesh : pv.UnstructuredGrid
    facets   : pv.PolyData or pv.UnstructuredGrid  (only when return_facets=True)
    """
    csf_cells = is_csf_marker(mesh.cell_data[label_array])

    if not return_facets:
        csf_mesh = filter_by_mask(mesh, csf_cells)
        return largest_face_connected(csf_mesh.clean())

    mesh["gid"] = np.arange(mesh.n_points)
    csf_mesh = largest_face_connected(filter_by_mask(mesh, csf_cells)).clean()
    assert "gid" in csf_mesh.array_names
    # Compute facets on the full mesh (preserves CSF-to-tissue interface IDs),
    full_facets = mark_facets(mesh, label_array=label_array, **facet_kwargs)

    csf_facets = extract_reduced_facets(csf_mesh, full_facets)

    assert np.allclose(csf_mesh.points, csf_facets.points)
    assert label_array in csf_mesh.array_names
    assert "interface_id" in csf_facets.array_names
    return csf_mesh, csf_facets




CSF_REGION_NAMES = []

def group_csf_facets_by_region(facets, encoding_base=100000):
    """
    Assign each CSF facet (from :func:`extract_csf`) to a named anatomical region.

    Decodes the ``interface_id`` cell array and maps each facet to one of the
    regions in :data:`CSF_REGION_NAMES`.  Adds a ``region`` (int32) cell array
    to a copy of ``facets``.  Every facet receives a non-zero region label.

    Region IDs and names
    --------------------

    Parameters
    ----------
    facets           : pv mesh with ``interface_id`` cell array
    encoding_base    : int    encoding base used in ``interface_id`` (default 100000)
    Returns
    -------
    pv mesh  copy of ``facets`` with added ``region`` (int32) cell array
    """
    from .labels import region_dict as sas_region_dict
    from .labels import _sas_lh, _sas_rh

    region_label_dict = {"SPINAL_CSF": 1, "PIA": 3, "LATERAL_VENTRICLES":2,
                         "FALX":4, "TENTORIUM_UPPER":5, "TENTORIUM_LOWER":6,
                         "UNCLASSIFIED":7,
                         "ANTERIOR_PARASAGITTAL_SINUS":8,"POSTERIOR_PARASAGITTAL_SINUS":9,
                         "VESSEL_WALL":30}

    ids = np.asarray(facets.cell_data["interface_id"], dtype=np.int64)
    a, b = np.divmod(ids, encoding_base)
    
    region = np.zeros(len(ids), dtype=np.int32)

    def _assign(mask, rid):
        region[mask & (region == 0)] = rid

    tent = int(Label.TENTORIUM)  # 71

    remove_id = int(-1)
    # remove all interfaces between different CSF regions
    _assign(np.logical_and(is_csf_marker(a), is_csf_marker(b)), remove_id)

    # 1. Spinal canal
    _assign(ids == SPINAL_ID, region_label_dict["SPINAL_CSF"])

    # mark lateral ventricle surface
    LVs = list(set(VENTRICLE_LABELS) - set((Label.THIRD_VENTRICLE,
                                            Label.FOURTH_VENTRICLE)))
    _assign(np.logical_xor(np.isin(a, LVs), np.isin(b, LVs)), region_label_dict["LATERAL_VENTRICLES"])

    # mark FALX
    _assign((a==Label.FALX) + (b==Label.FALX), region_label_dict["FALX"])

    # mark up and downward facing parts of tentorium
    _assign((a==tent) + (b==tent), region_label_dict["TENTORIUM_UPPER"])
    infra_tent_ids = list(sas_region_dict["INFRATENTORIAL"])
    region[np.logical_and(a==tent, np.isin(b, infra_tent_ids))] = region_label_dict["TENTORIUM_LOWER"]
    region[np.logical_and(b==tent, np.isin(a, infra_tent_ids))] = region_label_dict["TENTORIUM_LOWER"]

    # vessels and other unclassified material sitting in the SAS
    _assign((a == Label.UNCLASSIFIED) | (b == Label.UNCLASSIFIED),
            region_label_dict["UNCLASSIFIED"])

    # walls of reconstructed vessel lumens (Label.VESSEL)
    _assign((a == Label.VESSEL) | (b == Label.VESSEL), region_label_dict["VESSEL_WALL"])

    # all remaining internal -> tissue
    _assign(ids >= encoding_base, region_label_dict["PIA"])

    # mark specified regions:
    for i, (k, v) in enumerate(sas_region_dict.items()):
        _assign(np.isin(ids, list(v)), 10 + i)
        region_label_dict[k] = 10 + i

    # mark sagittal sinus
    # find right and left SAS labels
    rs_sas_labels = np.unique(ids[(ids > SAS_LABEL_OFFSET + 2000) & 
                                  (ids < SAS_LABEL_OFFSET + 3000)]).tolist()
    ls_sas_labels = np.unique(ids[(ids > SAS_LABEL_OFFSET + 1000) & 
                                  (ids < SAS_LABEL_OFFSET + 2000)]).tolist()

    # and find all facets that are within 10mm of both
    PSD = mark_between_regions(rs_sas_labels, ls_sas_labels, 10, 10, 
                               facets, "interface_id")

    # finally mark the front and back, depending on the SAS label IDs
    anterior_PSD = [1017,1022,1024, 1028]
    posterior_PSD = [1025, 1029, 1005, 1011, 1013]
    region[np.logical_and(PSD, np.isin(ids, _sas_rh(anterior_PSD) + _sas_lh(anterior_PSD)))] = region_label_dict["ANTERIOR_PARASAGITTAL_SINUS"]
    region[np.logical_and(PSD, np.isin(ids, _sas_rh(posterior_PSD) + _sas_lh(posterior_PSD)))] = region_label_dict["POSTERIOR_PARASAGITTAL_SINUS"]
    
    sas_ids = [region_label_dict[k] for k in sas_region_dict]
    sas_regions = np.unique(region[np.isin(region, sas_ids)]).tolist()
    out = facets.copy()
    region = remove_small_patches(out, region, threshold=50, target_labels=sas_regions)
    region = dilate_cell_marker(out, region, 10)
    out.cell_data["region"] = region
    out = smooth_cell_labels(out, marker_name="region", target_labels=sas_regions + [region_label_dict["ANTERIOR_PARASAGITTAL_SINUS"],
                                                                                      region_label_dict["POSTERIOR_PARASAGITTAL_SINUS"]])
    out = filter_by_mask(out, out.cell_data["region"] >= 0)
    assert (region==0).sum() == 0
    out.field_data["region_names"] = region_label_dict
    return out, region_label_dict


def spinal_interface_mask(region_a, region_b):
    """
    True where one side of a facet is ``Label.SPINAL_BUFFER`` and the other belongs
    to the CSF compartment (see :func:`brainmesh.labels.is_csf_marker`).

    ``SPINAL_BUFFER`` is the flat slab the segmentation pipeline extrudes below the
    bottom of the image (:func:`brainmesh.anatomy.extend_brainstem_caudally`), so
    this interface *is* the spinal opening — no geometric normal/height test needed.
    Buffer facets against the brainstem or against ``UNCLASSIFIED`` are not spinal
    and keep their ordinary interface id.
    """
    a = np.asarray(region_a)
    b = np.asarray(region_b)
    buf = Label.SPINAL_BUFFER
    return ((a == buf) & is_csf_marker(b)) | ((b == buf) & is_csf_marker(a))


def mark_spinal_boundary(mesh, label_array="marker", encoding_base=100000):
    """
    Mark the spinal interface: the facets between ``Label.SPINAL_BUFFER`` tets and
    the CSF compartment.

    Face winding is oriented so each normal points *out of* the CSF region, i.e.
    down into the buffer — the same outward convention the surrounding CSF boundary
    facets use.

    Parameters
    ----------
    mesh          : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array   : str                  cell data array with region markers
    encoding_base : int                  encoding base used while decoding the
                                         interface ids (default 100000)

    Returns a :class:`pyvista.PolyData` or :class:`pyvista.UnstructuredGrid` that
    shares the parent's point array, with a ``boundary`` cell data array holding the
    marker of the CSF-side tet.
    """
    interfaces = mark_interface_facets(mesh, label_array=label_array,
                                       encoding_base=encoding_base)
    lo = np.asarray(interfaces.cell_data["region_a"])
    hi = np.asarray(interfaces.cell_data["region_b"])

    keep = spinal_interface_mask(lo, hi)
    faces = np.array(_raw_faces(interfaces)[keep])
    lo, hi = lo[keep], hi[keep]

    # mark_interface_facets winds lo -> hi, so facets where the buffer is the lower
    # marker (against a SAS parcel) point the wrong way and need flipping.
    _flip_winding(faces, lo == Label.SPINAL_BUFFER)
    csf_side = np.where(lo == Label.SPINAL_BUFFER, hi, lo)

    return _build_facet_polydata(mesh, faces, {"boundary": csf_side})

def mark_facets(mesh, label_array="marker", encoding_base=100000,
                smooth_sas_labels=False, ignore_sas_interfaces=True):
    """
    Build a combined facet mesh containing all interface and boundary facets.

    Both groups share the parent's point array. A single ``interface_id`` cell
    array encodes the facet type:

    * Spinal facets (``Label.SPINAL_BUFFER`` against the CSF compartment):
      ``SPINAL_ID`` (= 99)
    * Other interface facets (between two labelled regions):
      ``min(a, b) * encoding_base + max(a, b)``
    * Boundary facets (outer surface):
      the region marker of the adjacent tet

    Parameters
    ----------
    mesh                  : pv.UnstructuredGrid  marked tetrahedral mesh
    label_array           : str                  cell data array with region markers
    encoding_base         : int                  multiplier for the interface ID encoding;
                                                 must exceed the maximum label value
                                                 (default 100000, handles SAS markers ≤ ~12035)
    smooth_sas_labels     : bool                 if True, apply majority-vote smoothing
                                                 to SAS boundary labels after extraction
    ignore_sas_interfaces : bool                 if True (default), drop interfaces where both
                                                 adjacent markers are SAS subdivision labels
                                                 (> ``SAS_LABEL_OFFSET``)

    Returns a :class:`pyvista.PolyData` or :class:`pyvista.UnstructuredGrid` that
    shares the parent's point array, with an ``interface_id`` cell data array.
    """
    interfaces = mark_interface_facets(mesh, label_array=label_array,
                                       encoding_base=encoding_base,
                                       ignore_sas_interfaces=ignore_sas_interfaces)
    boundaries = mark_boundary_facets(mesh, label_array=label_array)
    if smooth_sas_labels:
        bnd = boundaries["boundary"]
        sas_labels = np.unique(bnd[bnd > SAS_LABEL_OFFSET]).tolist()
        boundaries = smooth_cell_labels(boundaries, marker_name="boundary", target_labels=sas_labels)

    # The buffer-to-CSF interfaces are the spinal opening — re-tag them as SPINAL_ID
    # so downstream code sees one id instead of CSF/SAS-specific encodings.
    int_ids = np.array(interfaces.cell_data["interface_id"], dtype=np.int64)
    int_ids[spinal_interface_mask(interfaces.cell_data["region_a"],
                                  interfaces.cell_data["region_b"])] = SPINAL_ID

    int_faces = _raw_faces(interfaces)
    bnd_faces = _raw_faces(boundaries)
    bnd_ids = boundaries.cell_data["boundary"]

    all_faces = (np.vstack([int_faces, bnd_faces]) if (len(int_faces) or len(bnd_faces))
                 else np.empty((0, bnd_faces.shape[1]), dtype=np.int64))

    n = len(all_faces)
    if isinstance(interfaces, pv.PolyData):
        cells_flat = np.column_stack([np.full(n, 3, dtype=np.int64), all_faces]).ravel()
        combined = pv.PolyData(mesh.points, faces=cells_flat)
    else:
        nodes_per_face = all_faces.shape[1] if n else 6
        cells_flat = np.column_stack(
            [np.full(n, nodes_per_face, dtype=np.int64), all_faces]
        ).ravel()
        all_ctypes = np.full(n, pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
        combined = pv.UnstructuredGrid(cells_flat, all_ctypes, mesh.points)

    combined.cell_data["interface_id"] = np.concatenate([int_ids, bnd_ids])
    return combined
