"""Tests for the numba-based connected component labeling routines."""
import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import morphology as skmorph

from brainmesh.ccl import ccl_3d_6conn, ccl_3d_26conn, label, remove_small_objects


def _same_partition(a, b):
    """Return True when two label volumes induce the same partition of voxels.

    Two labelings are equivalent iff the relation
    "voxels with the same label in a" equals "voxels with the same label in b",
    independent of the actual label values used.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        return False
    # Map every pair (a_i, b_i) to a unique key; the labelings agree iff each
    # a-label maps to exactly one b-label and vice versa.
    pairs = np.stack([a, b], axis=1)
    unique_pairs = {tuple(p) for p in pairs.tolist()}
    a_to_b = {}
    b_to_a = {}
    for av, bv in unique_pairs:
        if a_to_b.setdefault(av, bv) != bv:
            return False
        if b_to_a.setdefault(bv, av) != av:
            return False
    return True


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_ccl_empty_volume():
    img = np.zeros((5, 6, 7), dtype=bool)
    out, n = ccl_3d_26conn(img)
    assert n == 0
    assert out.shape == img.shape
    assert np.all(out == 0)


def test_ccl_single_voxel():
    img = np.zeros((4, 4, 4), dtype=bool)
    img[2, 1, 3] = True
    out, n = ccl_3d_26conn(img)
    assert n == 1
    assert out[2, 1, 3] == 1
    assert (out > 0).sum() == 1


def test_ccl_full_volume():
    img = np.ones((3, 4, 5), dtype=bool)
    out, n = ccl_3d_26conn(img)
    assert n == 1
    assert np.all(out == 1)


def test_ccl_two_isolated_blobs():
    img = np.zeros((10, 10, 10), dtype=bool)
    img[1:3, 1:3, 1:3] = True
    img[6:9, 6:9, 6:9] = True
    out, n = ccl_3d_26conn(img)
    assert n == 2
    # Blob 1 and blob 2 must have distinct labels and be internally consistent
    lbl1 = out[1, 1, 1]
    lbl2 = out[7, 7, 7]
    assert lbl1 != lbl2
    assert np.all(out[1:3, 1:3, 1:3] == lbl1)
    assert np.all(out[6:9, 6:9, 6:9] == lbl2)


def test_ccl_diagonal_connectivity():
    """26-connectivity must merge voxels that touch only at a corner."""
    img = np.zeros((4, 4, 4), dtype=bool)
    img[1, 1, 1] = True
    img[2, 2, 2] = True  # diagonal neighbor (only corner contact)
    out, n = ccl_3d_26conn(img)
    assert n == 1
    assert out[1, 1, 1] == out[2, 2, 2]


def test_ccl_matches_scipy_random(rng):
    img = rng.random((15, 17, 19)) > 0.65
    out, n = ccl_3d_26conn(img)
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity in 3D
    scipy_lbl, n_scipy = ndi.label(img, structure=structure)
    assert n == n_scipy
    assert _same_partition(out, scipy_lbl)


def test_ccl_matches_scipy_dense(rng):
    img = rng.random((12, 12, 12)) > 0.3
    out, n = ccl_3d_26conn(img)
    structure = np.ones((3, 3, 3), dtype=bool)
    scipy_lbl, n_scipy = ndi.label(img, structure=structure)
    assert n == n_scipy
    assert _same_partition(out, scipy_lbl)


def test_ccl_matches_skimage(rng):
    img = rng.random((10, 11, 13)) > 0.5
    out, n = ccl_3d_26conn(img)
    sk_lbl = skmorph.label(img, connectivity=3)
    n_sk = int(sk_lbl.max())
    assert n == n_sk
    assert _same_partition(out, sk_lbl)


def test_label_alias_default_is_6conn(rng):
    img = rng.random((8, 9, 10)) > 0.5
    out_ref, n_ref = ccl_3d_6conn(img)
    out, n = label(img)
    assert n == n_ref
    np.testing.assert_array_equal(out, out_ref)


def test_label_alias_dispatches_on_connectivity(rng):
    img = rng.random((8, 9, 10)) > 0.5
    out6, n6 = label(img, connectivity=6)
    out26, n26 = label(img, connectivity=26)
    out6_ref, n6_ref = ccl_3d_6conn(img)
    out26_ref, n26_ref = ccl_3d_26conn(img)
    np.testing.assert_array_equal(out6, out6_ref)
    np.testing.assert_array_equal(out26, out26_ref)
    assert n6 == n6_ref and n26 == n26_ref


def test_remove_small_objects_drops_small_component():
    img = np.zeros((10, 10, 10), dtype=bool)
    img[1, 1, 1] = True  # size-1 component
    img[5:8, 5:8, 5:8] = True  # size-27 component
    out = remove_small_objects(img, max_size=5)
    assert not out[1, 1, 1]
    assert np.all(out[5:8, 5:8, 5:8])


def test_remove_small_objects_keeps_threshold_sized():
    """Components with size == max_size must be kept (>= comparison)."""
    img = np.zeros((8, 8, 8), dtype=bool)
    img[1, 1, 1:4] = True  # size 3 component
    out = remove_small_objects(img, max_size=3)
    assert np.all(out[1, 1, 1:4])


def test_remove_small_objects_drops_below_threshold():
    img = np.zeros((8, 8, 8), dtype=bool)
    img[1, 1, 1:4] = True  # size 3 component
    out = remove_small_objects(img, max_size=4)
    assert not out.any()


@pytest.mark.parametrize(
    "connectivity, sk_connectivity",
    [(6, 1), (26, 3)],
)
def test_remove_small_objects_matches_skimage(rng, connectivity, sk_connectivity):
    img = rng.random((12, 12, 12)) > 0.55
    threshold = 4
    out = remove_small_objects(
        img, max_size=threshold, connectivity=connectivity
    )

    # brainmesh keeps components with size >= threshold. skimage's new
    # max_size parameter removes components with size <= max_size, so passing
    # max_size = threshold - 1 produces the same keep-set. The connectivity
    # convention differs: skimage uses 1 / 2 / 3, brainmesh uses 6 / (18) / 26.
    expected = skmorph.remove_small_objects(
        img.copy(), max_size=threshold - 1, connectivity=sk_connectivity
    )
    np.testing.assert_array_equal(out, expected)


def test_remove_small_objects_all_kept(rng):
    img = rng.random((8, 8, 8)) > 0.4
    out = remove_small_objects(img, max_size=1)
    np.testing.assert_array_equal(out, img)


def test_remove_small_objects_invalid_connectivity_raises():
    img = np.ones((3, 3, 3), dtype=bool)
    with pytest.raises(ValueError):
        remove_small_objects(img, max_size=1, connectivity=8)


# ---------------------------------------------------------------------------
# ccl_3d_6conn — face-only (6-connectivity) labeling
# ---------------------------------------------------------------------------
def test_ccl_6conn_empty_volume():
    img = np.zeros((5, 6, 7), dtype=bool)
    out, n = ccl_3d_6conn(img)
    assert n == 0
    assert out.shape == img.shape
    assert np.all(out == 0)


def test_ccl_6conn_single_voxel():
    img = np.zeros((4, 4, 4), dtype=bool)
    img[2, 1, 3] = True
    out, n = ccl_3d_6conn(img)
    assert n == 1
    assert out[2, 1, 3] == 1
    assert (out > 0).sum() == 1


def test_ccl_6conn_full_volume():
    img = np.ones((3, 4, 5), dtype=bool)
    out, n = ccl_3d_6conn(img)
    assert n == 1
    assert np.all(out == 1)


def test_ccl_6conn_two_isolated_blobs():
    img = np.zeros((10, 10, 10), dtype=bool)
    img[1:3, 1:3, 1:3] = True
    img[6:9, 6:9, 6:9] = True
    out, n = ccl_3d_6conn(img)
    assert n == 2
    lbl1 = out[1, 1, 1]
    lbl2 = out[7, 7, 7]
    assert lbl1 != lbl2
    assert np.all(out[1:3, 1:3, 1:3] == lbl1)
    assert np.all(out[6:9, 6:9, 6:9] == lbl2)


def test_ccl_6conn_corner_touching_is_separate():
    """Voxels that touch only at a corner are NOT connected under 6-conn.

    This is the defining difference from 26-connectivity, where the same input
    would be a single component (see test_ccl_diagonal_connectivity).
    """
    img = np.zeros((4, 4, 4), dtype=bool)
    img[1, 1, 1] = True
    img[2, 2, 2] = True  # corner contact only
    out, n = ccl_3d_6conn(img)
    assert n == 2
    assert out[1, 1, 1] != out[2, 2, 2]


def test_ccl_6conn_edge_touching_is_separate():
    """Voxels sharing only an edge (two coords differing by 1) are also
    disconnected under 6-conn."""
    img = np.zeros((4, 4, 4), dtype=bool)
    img[1, 1, 1] = True
    img[1, 2, 2] = True  # edge contact only (shares one axis value)
    out, n = ccl_3d_6conn(img)
    assert n == 2


def test_ccl_6conn_face_touching_is_connected():
    """Voxels sharing a face must be connected under 6-conn."""
    img = np.zeros((4, 4, 4), dtype=bool)
    img[1, 1, 1] = True
    img[1, 1, 2] = True  # face contact
    out, n = ccl_3d_6conn(img)
    assert n == 1
    assert out[1, 1, 1] == out[1, 1, 2]


def test_ccl_6conn_matches_scipy_random(rng):
    img = rng.random((15, 17, 19)) > 0.65
    out, n = ccl_3d_6conn(img)
    # scipy default structure for ndi.label is 6-connectivity in 3D
    scipy_lbl, n_scipy = ndi.label(img)
    assert n == n_scipy
    assert _same_partition(out, scipy_lbl)


def test_ccl_6conn_matches_scipy_dense(rng):
    img = rng.random((12, 12, 12)) > 0.3
    out, n = ccl_3d_6conn(img)
    scipy_lbl, n_scipy = ndi.label(img)
    assert n == n_scipy
    assert _same_partition(out, scipy_lbl)


def test_ccl_6conn_matches_skimage(rng):
    img = rng.random((10, 11, 13)) > 0.5
    out, n = ccl_3d_6conn(img)
    sk_lbl = skmorph.label(img, connectivity=1)
    n_sk = int(sk_lbl.max())
    assert n == n_sk
    assert _same_partition(out, sk_lbl)


def test_ccl_6conn_gives_at_least_as_many_components_as_26conn(rng):
    """26-connectivity is a superset relation of 6-connectivity, so the
    6-conn labeling has >= as many components on any input."""
    img = rng.random((14, 15, 16)) > 0.55
    _, n6 = ccl_3d_6conn(img)
    _, n26 = ccl_3d_26conn(img)
    assert n6 >= n26


def test_ccl_6conn_differs_from_26conn_on_corner_example():
    """A concrete witness that 6- and 26-connectivity disagree."""
    img = np.zeros((5, 5, 5), dtype=bool)
    img[1, 1, 1] = True
    img[2, 2, 2] = True
    _, n6 = ccl_3d_6conn(img)
    _, n26 = ccl_3d_26conn(img)
    assert n6 == 2
    assert n26 == 1
