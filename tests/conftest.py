"""Shared test fixtures providing small synthetic segmentation volumes."""
import numpy as np
import pyvista as pv
import pytest


@pytest.fixture
def tiny_seg():
    """20^3 volume with three concentric spherical labels (background, CSF, WM)."""
    from brainmesh import Label

    N = 20
    data = np.ones((N, N, N), dtype=np.uint8)
    grid = pv.ImageData(dimensions=(N + 1, N + 1, N + 1), spacing=(1.0 / N,) * 3)
    pts = grid.cell_centers().points.reshape(N, N, N, 3)
    dist = np.linalg.norm(pts - 0.5, axis=-1)
    data[dist < 0.4] = Label.CSF
    data[dist < 0.25] = Label.LEFT_CEREBRAL_WHITE_MATTER
    return data


@pytest.fixture
def tiny_grid(tiny_seg):
    """PyVista ImageData for the tiny synthetic segmentation."""
    N = tiny_seg.shape[0]
    grid = pv.ImageData(dimensions=(N + 1, N + 1, N + 1), spacing=(1.0 / N,) * 3)
    grid["data"] = tiny_seg.flatten(order="F")
    return grid
