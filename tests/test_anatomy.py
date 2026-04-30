"""Tests for anatomy.py — verifies new kwargs are accepted and have the expected effect."""
import numpy as np
import pytest
from scipy.ndimage import label as scipy_label

from brainmesh import Label
from brainmesh.anatomy import (
    build_inferior_lateral_ventricle_horns,
    enforce_connected_ventricles,
    enforce_tight_ventricles,
    extend_brainstem,
    extend_brainstem_caudally,
)
from brainmesh.labels import VENTRICLE_LABELS


@pytest.fixture
def brainstem_seg():
    """Minimal volume with brainstem at the top, background in the middle, CSF at z=0 only.

    This layout is required by extend_brainstem, which grows the brainstem mask
    downward through background voxels until it reaches within csf_z_tolerance of
    the lowest CSF voxel.  If CSF fills all voxels below the brainstem the loop
    would never converge, so we keep just one CSF layer at z=0.
    """
    data = np.zeros((10, 10, 20), dtype=np.uint8)
    data[:, :, 0:1] = Label.CSF       # one thin CSF layer at the very bottom
    # background at z=1:10 provides space for the brainstem to grow into
    data[3:7, 3:7, 10:15] = Label.BRAIN_STEM
    return data


@pytest.fixture
def ventricle_seg():
    """Volume with V3, V4, and lateral ventricles surrounded by WM."""
    data = np.zeros((30, 30, 30), dtype=np.uint8)
    # WM background within brain
    data[5:25, 5:25, 5:25] = Label.LEFT_CEREBRAL_WHITE_MATTER
    # Ventricle compartments separated by a thin gap
    data[12:18, 12:18, 12:15] = Label.THIRD_VENTRICLE
    data[12:18, 12:18, 17:20] = Label.FOURTH_VENTRICLE
    data[8:12, 12:18, 12:18] = Label.LEFT_LATERAL_VENTRICLE
    data[18:22, 12:18, 12:18] = Label.RIGHT_LATERAL_VENTRICLE
    return data


class TestExtendBrainstem:
    def test_kwargs_accepted(self, brainstem_seg):
        result = extend_brainstem(
            brainstem_seg.copy(),
            csf_z_tolerance=2,
            extension_dilation_radius=2,
        )
        assert result.dtype == np.uint8

    def test_larger_tolerance_stops_earlier(self, brainstem_seg):
        """Higher csf_z_tolerance means less downward growth."""
        result_tight = extend_brainstem(brainstem_seg.copy(), csf_z_tolerance=1)
        result_loose = extend_brainstem(brainstem_seg.copy(), csf_z_tolerance=6)
        bs_tight = (result_tight == Label.BRAIN_STEM).sum()
        bs_loose = (result_loose == Label.BRAIN_STEM).sum()
        # Loose tolerance stops sooner → fewer brainstem voxels added
        assert bs_tight >= bs_loose


class TestExtendBrainstemCaudally:
    def test_kwargs_accepted(self, brainstem_seg):
        result = extend_brainstem_caudally(
            brainstem_seg.copy(),
            footprint_z_offset=2,
            footprint_closing_radius=1,
            csf_buffer_radius=1,
        )
        assert result.dtype == np.uint8

    def test_larger_buffer_adds_more_csf(self, brainstem_seg):
        result_small = extend_brainstem_caudally(
            brainstem_seg.copy(), footprint_z_offset=2, csf_buffer_radius=1
        )
        result_large = extend_brainstem_caudally(
            brainstem_seg.copy(), footprint_z_offset=2, csf_buffer_radius=3
        )
        csf_small = (result_small == Label.CSF).sum()
        csf_large = (result_large == Label.CSF).sum()
        assert csf_large >= csf_small


class TestEnforceConnectedVentricles:
    def test_kwargs_accepted(self, ventricle_seg):
        result = enforce_connected_ventricles(
            ventricle_seg.copy(),
            connection_radius=2,
            mask_smoothing_radius=2,
        )
        assert result.dtype == np.uint8

    def test_all_ventricles_become_one_connected_component(self):
        """V3, V4, LLV, and RLV start as 4 separate islands and must all be
        reachable from each other after enforce_connected_ventricles runs."""
        data = np.zeros((30, 30, 30), dtype=np.uint8)
        # WM background so tissue dilations inside the function have non-zero material
        data[3:27, 3:27, 3:27] = Label.LEFT_CEREBRAL_WHITE_MATTER
        # V3 in the center
        data[13:17, 13:17, 13:17] = Label.THIRD_VENTRICLE
        # V4 separated from V3 by a 2-voxel gap along z (z=11:13)
        data[13:17, 13:17, 8:11] = Label.FOURTH_VENTRICLE
        # LLV separated from V3 by a 2-voxel gap along x (x=11:13)
        data[7:11, 13:17, 13:17] = Label.LEFT_LATERAL_VENTRICLE
        # RLV separated from V3 by a 2-voxel gap along x (x=17:19)
        data[19:23, 13:17, 13:17] = Label.RIGHT_LATERAL_VENTRICLE

        # Precondition: 4 separate connected components before the function runs
        all_v_before = np.isin(data, VENTRICLE_LABELS)
        _, n_before = scipy_label(all_v_before)
        assert n_before == 4, f"Fixture should have 4 disconnected ventricles, got {n_before}"

        result = enforce_connected_ventricles(
            data.copy(),
            connection_radius=2,
            mask_smoothing_radius=1,
        )

        all_v_after = np.isin(result, VENTRICLE_LABELS)
        _, n_after = scipy_label(all_v_after)
        assert n_after == 1, f"Expected all ventricles in 1 component after connecting, got {n_after}"


class TestEnforceTightVentricles:
    def test_kwargs_accepted(self, ventricle_seg):
        result = enforce_tight_ventricles(
            ventricle_seg.copy(),
            surrounding_layer_thickness=2,
            bottom_exclusion_z_offset=1,
            tissue_fill_radius=3,
        )
        assert result.dtype == np.uint8

    def test_jacket_voxels_become_tissue(self):
        """CSF voxels immediately adjacent to a ventricle (above the outlet zone)
        should be replaced by the surrounding tissue label."""
        data = np.zeros((20, 20, 20), dtype=np.uint8)
        # WM fills the brain region; ventricle sits in the upper z half
        data[2:18, 2:18, 2:18] = Label.LEFT_CEREBRAL_WHITE_MATTER
        data[7:13, 7:13, 10:16] = Label.LEFT_LATERAL_VENTRICLE
        # Plant CSF on two exposed faces of the ventricle
        data[6, 7:13, 11:15] = Label.CSF   # side face (x direction)
        data[7:13, 7:13, 16] = Label.CSF   # top face (z direction)

        result = enforce_tight_ventricles(
            data.copy(),
            surrounding_layer_thickness=2,
            bottom_exclusion_z_offset=0,  # no outlet exclusion; full jacket filled
            tissue_fill_radius=5,
        )

        non_tissue = [0, Label.CSF] + list(VENTRICLE_LABELS)
        # Both CSF patches are adjacent to the ventricle and above the outlet → must be tissue
        assert result[6, 10, 13] not in non_tissue, \
            f"Side CSF patch not converted: {result[6, 10, 13]}"
        assert result[10, 10, 16] not in non_tissue, \
            f"Top CSF patch not converted: {result[10, 10, 16]}"

    def test_bottom_outlet_left_unchanged(self):
        """When bottom_exclusion_z_offset is very large the entire jacket is the
        outlet zone and no voxel should be overwritten."""
        data = np.zeros((20, 20, 20), dtype=np.uint8)
        data[2:18, 2:18, 2:18] = Label.LEFT_CEREBRAL_WHITE_MATTER
        data[7:13, 7:13, 10:16] = Label.LEFT_LATERAL_VENTRICLE
        # CSF just below the ventricle (in the outlet zone)
        data[7:13, 7:13, 9] = Label.CSF

        result = enforce_tight_ventricles(
            data.copy(),
            surrounding_layer_thickness=2,
            bottom_exclusion_z_offset=100,  # clears the entire jacket → nothing modified
            tissue_fill_radius=5,
        )
        # CSF at the outlet must remain intact
        assert (result[7:13, 7:13, 9] == Label.CSF).all(), \
            "Outlet CSF should not be overwritten"


class TestBuildInfLatVentHorns:
    def test_kwargs_accepted(self):
        data = np.zeros((20, 20, 20), dtype=np.uint8)
        data[8:12, 8:12, 8:12] = Label.LEFT_INFERIOR_LATERAL_VENTRICLE
        result = build_inferior_lateral_ventricle_horns(
            data.copy(),
            horn_closing_radius=3,
            post_close_dilation_radius=1,
            smoothing_radius=1,
        )
        assert result.dtype == np.uint8
        # Closing + dilation should not shrink the region
        assert (result == Label.LEFT_INFERIOR_LATERAL_VENTRICLE).sum() >= \
               (data == Label.LEFT_INFERIOR_LATERAL_VENTRICLE).sum()

    def test_bridges_disconnected_horn_pieces(self):
        """Two disconnected blobs of the inferior lateral ventricle horn must be
        merged into a single connected region by the spherical closing pass."""
        data = np.zeros((30, 30, 30), dtype=np.uint8)
        lbl = Label.LEFT_INFERIOR_LATERAL_VENTRICLE
        # Two cubes separated by a 4-voxel gap along z (z=10:14)
        data[10:16, 10:16, 5:10] = lbl
        data[10:16, 10:16, 14:19] = lbl

        # Precondition: really are two separate components
        _, n_before = scipy_label(data == lbl)
        assert n_before == 2, f"Fixture should have 2 islands, got {n_before}"

        result = build_inferior_lateral_ventricle_horns(
            data.copy(),
            horn_closing_radius=5,  # radius 5 bridges a 4-voxel gap
            post_close_dilation_radius=1,
            smoothing_radius=1,
        )

        _, n_after = scipy_label(result == lbl)
        assert n_after == 1, f"Expected 1 connected horn after bridging, got {n_after}"
