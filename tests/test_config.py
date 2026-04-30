"""Tests for SegmentationConfig — dataclass defaults, TOML loading, and error handling."""
import pathlib
from dataclasses import asdict

import pytest

from brainmesh.config import (
    SegmentationConfig,
    FalxCfg,
    TentoriumCfg,
    TightVentriclesCfg,
    ExtendBrainstemCaudallyCfg,
    ConnectedVentriclesCfg,
    SolidifyCSFCfg,
    FillWMHyperCfg,
    ExtendBrainstemCfg,
    InfLatVentHornsCfg,
    PipelineMiscCfg,
)

REPO_ROOT = pathlib.Path(__file__).parent.parent


class TestDefaults:
    def test_instantiation(self):
        cfg = SegmentationConfig()
        assert isinstance(cfg, SegmentationConfig)

    def test_falx_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.falx.hemisphere_gap == 6
        assert cfg.falx.territory_smoothing_sigma == 20.0
        assert cfg.falx.cerebrum_proximity_radius == 20
        assert cfg.falx.non_cerebral_clearance_radius == 4
        assert cfg.falx.cerebellum_clearance_radius == 2
        assert cfg.falx.third_ventricle_clearance_radius == 30

    def test_tentorium_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.tentorium.cerebrum_cerebellum_gap == 3
        assert cfg.tentorium.territory_smoothing_sigma == 6.0
        assert cfg.tentorium.phantom_cerebellum_sigma_factor == 3.0
        assert cfg.tentorium.brainstem_clearance_radius == 10

    def test_brainstem_caudally_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.extend_brainstem_caudally.footprint_z_offset == 18
        assert cfg.extend_brainstem_caudally.footprint_closing_radius == 4
        assert cfg.extend_brainstem_caudally.csf_buffer_radius == 4

    def test_tight_ventricles_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.tight_ventricles.surrounding_layer_thickness == 3
        assert cfg.tight_ventricles.bottom_exclusion_z_offset == 20
        assert cfg.tight_ventricles.tissue_fill_radius == 10

    def test_connected_ventricles_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.connected_ventricles.connection_radius == 2
        assert cfg.connected_ventricles.mask_smoothing_radius == 2

    def test_misc_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.misc.original_mask_smoothing_radius == 1
        assert cfg.misc.apply_mode_box_pre is True
        assert cfg.misc.apply_mode_box_post is True
        assert cfg.misc.apply_mode_diamond_post is True

    def test_solidify_csf_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.solidify_csf.mask_closing_radius == 5
        assert cfg.solidify_csf.mask_closing_iterations == 1

    def test_fill_wm_hyper_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.fill_wm_hyperintensities.wm_search_radius == 6

    def test_coarsen_surface_defaults(self):
        cfg = SegmentationConfig()
        assert cfg.coarsen_surface.decimation_ratio == 0.9


class TestFromDict:
    def test_full_override_single_section(self):
        cfg = SegmentationConfig.from_dict({"falx": {"hemisphere_gap": 4}})
        assert cfg.falx.hemisphere_gap == 4
        # Other falx fields preserved
        assert cfg.falx.territory_smoothing_sigma == 20.0
        # Unrelated sections preserved
        assert cfg.tentorium.cerebrum_cerebellum_gap == 3

    def test_multiple_section_overrides(self):
        cfg = SegmentationConfig.from_dict({
            "falx": {"hemisphere_gap": 8},
            "tentorium": {"cerebrum_cerebellum_gap": 5},
        })
        assert cfg.falx.hemisphere_gap == 8
        assert cfg.tentorium.cerebrum_cerebellum_gap == 5

    def test_empty_dict_gives_defaults(self):
        cfg = SegmentationConfig.from_dict({})
        assert asdict(cfg) == asdict(SegmentationConfig())

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ValueError, match="unknown_section"):
            SegmentationConfig.from_dict({"unknown_section": {}})

    def test_unknown_nested_key_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            SegmentationConfig.from_dict({"falx": {"bogus": 1}})

    def test_misc_booleans(self):
        cfg = SegmentationConfig.from_dict({"misc": {"apply_mode_box_pre": False}})
        assert cfg.misc.apply_mode_box_pre is False
        assert cfg.misc.apply_mode_box_post is True


class TestToml:
    def test_default_toml_matches_hardcoded_defaults(self):
        """configs/default.toml must reproduce exactly the in-code defaults."""
        toml_path = REPO_ROOT / "configs" / "default.toml"
        assert toml_path.exists(), "configs/default.toml is missing"
        cfg_toml = SegmentationConfig.from_toml(toml_path)
        assert asdict(cfg_toml) == asdict(SegmentationConfig())

    def test_partial_toml(self, tmp_path):
        toml_file = tmp_path / "partial.toml"
        toml_file.write_text('[falx]\nhemisphere_gap = 4\n')
        cfg = SegmentationConfig.from_toml(toml_file)
        assert cfg.falx.hemisphere_gap == 4
        assert cfg.falx.territory_smoothing_sigma == 20.0  # default preserved

    def test_toml_unknown_key_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[falx]\nbogus = 99\n')
        with pytest.raises(ValueError, match="bogus"):
            SegmentationConfig.from_toml(toml_file)

    def test_toml_unknown_section_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[not_a_real_section]\nfoo = 1\n')
        with pytest.raises(ValueError, match="not_a_real_section"):
            SegmentationConfig.from_toml(toml_file)
