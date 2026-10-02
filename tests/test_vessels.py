"""Unit tests for vessel lumens and perivascular CSF sleeves."""

import nibabel as nib
import numpy as np
import pytest
import pyvista as pv

from brainmesh import Label
from brainmesh.config import SegmentationConfig, VesselCfg
from brainmesh.segmentation import count_background_contacts
from brainmesh.vessels import (
    centerline_distance,
    enforce_vessel_sleeve,
    mask_distance,
    read_centerlines,
    vessel_distance,
)


def _voxel_centres(shape, affine):
    ijk = np.indices(shape).reshape(3, -1).T
    return nib.affines.apply_affine(affine, ijk).reshape(*shape, 3)


def _capsule(x, a, b, ra, rb):
    """Brute-force signed distance to a tapered capsule (reference implementation)."""
    d = b - a
    t = np.clip(((x - a) @ d) / (d @ d), 0, 1)
    q = x - (a + t[..., None] * d)
    return np.linalg.norm(q, axis=-1) - ((1 - t) * ra + t * rb)


def _write_centerlines(path, polylines, radii, segment_ids, names):
    pts = np.concatenate(polylines)
    lines, offset = [], 0
    for pl in polylines:
        lines += [len(pl), *range(offset, offset + len(pl))]
        offset += len(pl)
    poly = pv.PolyData(pts, lines=np.array(lines))
    poly.point_data["radius"] = np.concatenate(radii)
    poly.cell_data["segment"] = np.array(segment_ids, dtype=np.int8)
    poly.field_data["segment_names"] = np.array(names)
    poly.save(path)
    return path


# a 10 mm cube at 0.5 mm, with a straight vessel along x through its middle
SHAPE = (20, 20, 20)
AFFINE = np.diag([0.5, 0.5, 0.5, 1.0])
A = np.array([-5.0, 4.75, 4.75])
B = np.array([15.0, 4.75, 4.75])


def test_straight_segment_matches_cylinder():
    dist = centerline_distance(
        SHAPE, AFFINE, A[None], B[None], np.array([1.0]), np.array([1.0]), band=1.0
    )
    x = _voxel_centres(SHAPE, AFFINE)
    ref = np.linalg.norm(x[..., 1:] - A[1:], axis=-1) - 1.0
    inband = ref <= 1.0
    np.testing.assert_allclose(dist[inband], ref[inband], atol=1e-5)
    # far outside the band (beyond the padded bounding box) nothing is evaluated
    assert np.isinf(dist[ref > 3.0]).all()


def test_tapered_segment_interpolates_radius():
    dist = centerline_distance(
        SHAPE, AFFINE, A[None], B[None], np.array([2.0]), np.array([0.0]), band=1.0
    )
    x = _voxel_centres(SHAPE, AFFINE)
    ref = _capsule(x, A, B, 2.0, 0.0)
    inband = ref <= 1.0
    np.testing.assert_allclose(dist[inband], ref[inband], atol=1e-5)
    # the lumen narrows along x
    lumen = (dist <= 0).sum(axis=(1, 2))
    assert lumen[0] > lumen[-1]


def test_oblique_affine():
    rot, _ = np.linalg.qr(np.array([[1.0, 0.3, 0.1], [-0.2, 1.0, 0.4], [0.1, -0.3, 1.0]]))
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([0.4, 0.5, 0.6])
    affine[:3, 3] = [-1.0, 2.0, -3.0]
    x = _voxel_centres(SHAPE, affine)
    a, b = x[2, 3, 4], x[17, 15, 12]
    dist = centerline_distance(
        SHAPE, affine, a[None], b[None], np.array([0.8]), np.array([1.2]), band=1.5
    )
    ref = _capsule(x, a, b, 0.8, 1.2)
    inband = ref <= 1.5
    assert inband.sum() > 100
    np.testing.assert_allclose(dist[inband], ref[inband], atol=1e-5)


def test_read_centerlines_filters(tmp_path):
    x = np.linspace(0, 4, 5)
    pl0 = np.c_[x, 0 * x, 0 * x]
    pl1 = np.c_[0 * x, x, 0 * x]
    pl1 = np.vstack([pl1, pl1[-1:]])  # duplicated end point -> zero-length segment
    path = _write_centerlines(
        tmp_path / "cl.vtp",
        [pl0, pl1],
        [np.full(5, 1.0), np.full(6, 0.2)],
        [1, 2],
        ["unknown", "A", "B"],
    )
    p0, _, r0, _ = read_centerlines(path)
    assert len(p0) == 4 + 4  # zero-length segment dropped
    assert len(read_centerlines(path, segments=["A"])[0]) == 4
    assert len(read_centerlines(path, segments=[2])[0]) == 4
    assert len(read_centerlines(path, min_radius=0.5)[0]) == 4
    _, _, r0, r1 = read_centerlines(path, radius_scale=2.0, radius_floor=0.5)
    assert r0.min() == 0.5 and r0.max() == 2.0
    with pytest.raises(ValueError, match="unknown vessel segment"):
        read_centerlines(path, segments=["C"])


def _cylinder_mask(shape, affine, radius, label=1):
    x = _voxel_centres(shape, affine)
    m = np.linalg.norm(x[..., 1:] - A[1:], axis=-1) <= radius
    return nib.Nifti1Image((m * label).astype(np.uint8), affine)


def test_mask_distance_same_grid():
    seg = nib.Nifti1Image(np.zeros(SHAPE, np.uint8), AFFINE)
    dist = mask_distance(_cylinder_mask(SHAPE, AFFINE, 2.0), seg)
    x = _voxel_centres(SHAPE, AFFINE)
    ref = np.linalg.norm(x[..., 1:] - A[1:], axis=-1) - 2.0
    assert np.abs(dist - ref).max() <= 0.5  # within one voxel


def test_mask_distance_resampled_and_labels():
    # the mask lives on a finer, shifted grid holding two labels
    fine = np.diag([0.25, 0.25, 0.25, 1.0])
    fine[:3, 3] = -0.4
    shape = (42, 42, 42)
    m = np.asarray(_cylinder_mask(shape, fine, 2.0, label=1).dataobj)
    m[:, :4] = 5  # a slab we don't want as vessel
    mask = nib.Nifti1Image(m, fine)
    seg = nib.Nifti1Image(np.zeros(SHAPE, np.uint8), AFFINE)

    x = _voxel_centres(SHAPE, AFFINE)
    ref = np.linalg.norm(x[..., 1:] - A[1:], axis=-1) - 2.0
    dist = mask_distance(mask, seg, labels=[1])
    assert np.abs(dist - ref).max() <= 0.5
    assert (mask_distance(mask, seg)[:, :2] <= 0).any()  # all > 0 includes the slab


def test_vessel_distance_union(tmp_path):
    x = np.linspace(-5, 15, 21)
    far = np.c_[x, np.full_like(x, 1.0), np.full_like(x, 8.0)]
    cl = _write_centerlines(tmp_path / "cl.vtp", [far], [np.full(21, 0.6)], [1], ["unknown", "A"])
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(_cylinder_mask(SHAPE, AFFINE, 1.5), mask_path)
    seg = nib.Nifti1Image(np.zeros(SHAPE, np.uint8), AFFINE)

    assert vessel_distance(seg) is None
    d_cl = vessel_distance(seg, centerlines=cl)
    d_m = vessel_distance(seg, mask=mask_path)
    d = vessel_distance(seg, centerlines=cl, mask=mask_path)
    np.testing.assert_array_equal(d, np.minimum(d_cl, d_m))
    assert ((d <= 0) == ((d_cl <= 0) | (d_m <= 0))).all()
    assert (d_cl <= 0).any() and (d_m <= 0).any()


@pytest.fixture
def block():
    """Tissue slab inside a CSF shell, with a falx plane and a ventricle cube."""
    data = np.zeros(SHAPE, dtype=np.uint8)
    data[1:-1, 1:-1, 1:-1] = Label.CSF
    data[3:-3, 3:-3, 3:-3] = Label.LEFT_CEREBRAL_WHITE_MATTER
    data[12, 3:-3, 3:-3] = Label.FALX
    data[5:8, 12:15, 12:15] = Label.LEFT_LATERAL_VENTRICLE
    return data


def _straight_vessel(radius=1.0):
    return centerline_distance(
        SHAPE, AFFINE, A[None], B[None], np.array([radius]), np.array([radius]), band=2.0
    )


@pytest.mark.parametrize("subdomain", [False, True])
def test_sleeve_labels(block, subdomain):
    dist = _straight_vessel()
    out = enforce_vessel_sleeve(
        block.copy(), dist, sleeve_thickness=0.75, subdomain=subdomain, ventricle_clearance=0
    )
    protected = np.isin(block, [Label.FALX, Label.LEFT_LATERAL_VENTRICLE])
    np.testing.assert_array_equal(out[protected], block[protected])
    np.testing.assert_array_equal(out[block == 0], 0)  # the domain never grows

    writable = (block > 0) & ~protected
    sleeve = (dist > 0) & (dist <= 0.75) & writable
    assert (out[sleeve] == Label.CSF).all()
    lumen = (dist <= 0) & writable
    assert (out[lumen] == (Label.VESSEL if subdomain else Label.CSF)).all()
    assert (out == Label.VESSEL).any() == subdomain
    assert count_background_contacts(out) == 0


def test_subdomain_csf_jacket(block):
    # a sleeve thinner than a voxel still leaves a CSF jacket around the lumen
    out = enforce_vessel_sleeve(
        block.copy(), _straight_vessel(), sleeve_thickness=0.1, subdomain=True, min_csf_voxels=1
    )
    vessel = out == Label.VESSEL
    nbrs = np.zeros_like(vessel)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                nbrs |= np.roll(vessel, (di, dj, dk), axis=(0, 1, 2))
    around = out[nbrs & ~vessel]
    assert np.isin(around, [Label.CSF, Label.FALX, 0]).all()


def test_small_lumen_fragments_become_csf(block):
    dist = _straight_vessel()
    dist[5, 5, 5] = -1.0  # an isolated single-voxel lumen, inside the tissue
    out = enforce_vessel_sleeve(
        block.copy(), dist, sleeve_thickness=0.75, subdomain=True, min_lumen_voxels=27
    )
    assert out[5, 5, 5] == Label.CSF
    assert (out == Label.VESSEL).sum() > 27  # the main tube is kept


def test_thin_sleeve_warns(block):
    with pytest.warns(UserWarning, match="thinner than a voxel"):
        enforce_vessel_sleeve(
            block.copy(), _straight_vessel(), sleeve_thickness=0.3, voxel_size=(0.5, 0.5, 0.5)
        )


def test_vessel_config():
    assert SegmentationConfig().vessels == VesselCfg()
    cfg = SegmentationConfig.from_dict({"vessels": {"subdomain": True, "segments": ["MCA-L"]}})
    assert cfg.vessels.subdomain and cfg.vessels.segments == ["MCA-L"]
    assert cfg.vessels.sleeve_thickness == 0.5
