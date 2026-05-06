"""Unit tests for voxel-label segmentation operations."""
import numpy as np
import pytest

from brainmesh import Label
from brainmesh.segmentation import (
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
