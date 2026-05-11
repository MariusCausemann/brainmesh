"""Unit tests for voxel-label segmentation operations."""
import numpy as np
import pytest
import scipy.ndimage as ndi

from brainmesh import Label
from brainmesh.segmentation import (
    binary_fill_holes,
    close_csf_space,
    cut_bottom,
    enforce_csf_layer,
    fill_holes_csf,
    fill_wm_hyperintensities,
    get_lowest_point,
    solidify_csf,
)


def test_get_lowest_point():
    mask = np.zeros((5, 5, 10), dtype=bool)
    mask[2, 2, 3] = True
    mask[2, 2, 7] = True
    pt = get_lowest_point(mask)
    assert pt[2] == 3


def test_fill_holes_csf(tiny_seg):
    data = tiny_seg.copy()
    # Punch a hole in the brain mask that is surrounded by tissue
    data[10, 10, 10] = 0
    result = fill_holes_csf(data)
    assert result[10, 10, 10] != 0


def test_cut_bottom():
    data = np.ones((10, 10, 10), dtype=np.uint8)
    data[:, :, :3] = 0  # simulate background at bottom
    result = cut_bottom(data, offset=2)
    # The lowest non-zero z is 3; offset=2 cuts everything below z=5
    assert np.all(result[:, :, :5] == 0)


def test_enforce_csf_layer_adds_csf(tiny_seg):
    data = tiny_seg.copy()
    # Remove existing CSF layer
    data[data == Label.CSF] = Label.LEFT_CEREBRAL_WHITE_MATTER
    result = enforce_csf_layer(data, thickness=1)
    assert np.any(result == Label.CSF)


def test_fill_wm_hyperintensities(tiny_seg):
    data = tiny_seg.copy()
    # Plant a hyperintensity voxel adjacent to WM
    wm_idx = np.argwhere(data == Label.LEFT_CEREBRAL_WHITE_MATTER)[0]
    data[tuple(wm_idx)] = Label.WM_HYPOINTENSITIES
    result = fill_wm_hyperintensities(data)
    assert result[tuple(wm_idx)] != Label.WM_HYPOINTENSITIES

def test_solidify_csf_kwargs_accepted(tiny_seg):
    """solidify_csf must accept the new keyword arguments without error."""
    data = tiny_seg.copy()
    result = solidify_csf(data, mask_closing_radius=3, mask_closing_iterations=1)
    assert result.dtype == np.uint8


def test_solidify_csf_larger_radius_fills_more(tiny_seg):
    """A larger closing radius should fill at least as many voxels as a small radius."""
    data = tiny_seg.copy()
    result_small = solidify_csf(data.copy(), mask_closing_radius=1)
    result_large = solidify_csf(data.copy(), mask_closing_radius=5)
    csf_small = (result_small == Label.CSF).sum()
    csf_large = (result_large == Label.CSF).sum()
    assert csf_large >= csf_small


# ---------------------------------------------------------------------------
# binary_fill_holes — compare against scipy.ndimage.binary_fill_holes
# ---------------------------------------------------------------------------
def _scipy_fill_6conn(img):
    """scipy reference with the same 6-connectivity used by brainmesh."""
    structure = ndi.generate_binary_structure(3, 1)  # 6-connectivity in 3D
    return ndi.binary_fill_holes(img, structure=structure)


def test_binary_fill_holes_empty_volume():
    img = np.zeros((5, 6, 7), dtype=bool)
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_full_volume():
    img = np.ones((4, 5, 6), dtype=bool)
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_no_holes():
    """A foreground blob with no interior holes must be returned unchanged."""
    img = np.zeros((10, 10, 10), dtype=bool)
    img[2:8, 2:8, 2:8] = True
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_single_hole():
    img = np.zeros((10, 10, 10), dtype=bool)
    img[2:8, 2:8, 2:8] = True
    img[4, 4, 4] = False  # single interior hole
    out = binary_fill_holes(img)
    expected = img.copy()
    expected[4, 4, 4] = True
    np.testing.assert_array_equal(out, expected)


def test_binary_fill_holes_hole_touching_boundary_not_filled():
    """A "hole" that connects to the outside through 6-connectivity should not
    be filled — it isn't enclosed."""
    img = np.zeros((10, 10, 10), dtype=bool)
    img[2:8, 2:8, 2:8] = True
    # Tunnel the hole all the way out through a face
    img[3:8, 4, 4] = False  # carves a channel from the +x face inward
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_matches_scipy_simple():
    img = np.zeros((12, 12, 12), dtype=bool)
    img[2:10, 2:10, 2:10] = True
    img[5, 5, 5] = False
    img[6, 7, 4:6] = False
    np.testing.assert_array_equal(binary_fill_holes(img), _scipy_fill_6conn(img))


def test_binary_fill_holes_matches_scipy_random():
    rng = np.random.default_rng(7)
    # Build a connected hollow shell with random interior holes inside.
    img = np.zeros((15, 16, 17), dtype=bool)
    img[2:13, 2:14, 2:15] = True
    interior = np.zeros_like(img)
    interior[5:10, 5:10, 5:10] = rng.random((5, 5, 5)) > 0.5
    img &= ~interior
    np.testing.assert_array_equal(binary_fill_holes(img), _scipy_fill_6conn(img))


def test_binary_fill_holes_hollow_sphere(tiny_seg):
    """The CSF/WM-shell fixture has WM enclosed by CSF; punching a hole inside
    WM should be re-filled."""
    mask = tiny_seg > 0
    # Carve a single-voxel hole well inside the WM core
    wm_idx = np.argwhere(tiny_seg == Label.LEFT_CEREBRAL_WHITE_MATTER)
    center = wm_idx[len(wm_idx) // 2]
    mask[tuple(center)] = False
    filled = binary_fill_holes(mask)
    assert filled[tuple(center)]
    # Whole-volume agreement with scipy
    np.testing.assert_array_equal(filled, _scipy_fill_6conn(mask))
