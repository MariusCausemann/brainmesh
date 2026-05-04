"""Typed configuration for the segmentation cleanup pipeline.

Each pipeline step has a small dataclass whose field names match the kwargs
of the corresponding function, so the pipeline can do
``func(data, **asdict(cfg.<step>))``. ``SegmentationConfig`` aggregates them
all and supports loading partial overrides from a TOML file.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import tomllib


@dataclass
class SolidifyCSFCfg:
    mask_closing_radius: int = 5
    mask_closing_iterations: int = 1


@dataclass
class CloseCSFSpaceCfg:
    radius: int = 3
    iter: int = 1
    brainstem_area_radius: int = 0


@dataclass
class FillWMHyperCfg:
    wm_search_radius: int = 6


@dataclass
class CutBottomCfg:
    offset: int = 10


@dataclass
class ExtendBrainstemCfg:
    csf_z_tolerance: int = 2
    extension_dilation_radius: int = 2


@dataclass
class EnforceCSFLayerCfg:
    thickness: int = 1


@dataclass
class FalxCfg:
    hemisphere_gap: int = 6
    territory_smoothing_sigma: float = 20.0
    boundary_thickness_radius: int = 1
    cerebrum_proximity_radius: int = 25
    non_cerebral_clearance_radius: int = 4
    cerebellum_clearance_radius: int = 2
    third_ventricle_clearance_radius: int = 30
    surrounding_csf_radius: int = 1


@dataclass
class TentoriumCfg:
    cerebrum_cerebellum_gap: int = 3
    territory_smoothing_sigma: float = 6.0
    phantom_cerebellum_sigma_factor: float = 3.0
    boundary_thickness_radius: int = 1
    cerebrum_cerebellum_proximity_radius: int = 12
    brainstem_clearance_radius: int = 10
    mask_thickening_radius: int = 1
    surrounding_csf_radius: int = 1


@dataclass
class InfLatVentHornsCfg:
    horn_closing_radius: int = 15
    post_close_dilation_radius: int = 1
    smoothing_radius: int = 2


@dataclass
class MinThicknessV4Cfg:
    radius: int = 1


@dataclass
class ConnectedVentriclesCfg:
    connection_radius: int = 2
    mask_smoothing_radius: int = 2


@dataclass
class TightVentriclesCfg:
    surrounding_layer_thickness: int = 3
    bottom_exclusion_z_offset: int = 20
    tissue_fill_radius: int = 10


@dataclass
class ExtendBrainstemCaudallyCfg:
    footprint_z_offset: int = 18
    footprint_closing_radius: int = 4
    csf_buffer_radius: int = 4


@dataclass
class EnforceCSFAroundCfg:
    radius: int = 1


@dataclass
class CoarsenSurfaceCfg:
    decimation_ratio: float = 0.9


@dataclass
class PipelineMiscCfg:
    original_mask_smoothing_radius: int = 1
    apply_mode_box_pre: bool = True
    apply_mode_box_post: bool = True
    apply_mode_diamond_post: bool = True


@dataclass
class SegmentationConfig:
    solidify_csf: SolidifyCSFCfg = field(default_factory=SolidifyCSFCfg)
    close_csf_space: CloseCSFSpaceCfg = field(default_factory=CloseCSFSpaceCfg)
    fill_wm_hyperintensities: FillWMHyperCfg = field(default_factory=FillWMHyperCfg)
    cut_bottom: CutBottomCfg = field(default_factory=CutBottomCfg)
    extend_brainstem: ExtendBrainstemCfg = field(default_factory=ExtendBrainstemCfg)
    enforce_csf_layer_pre: EnforceCSFLayerCfg = field(default_factory=EnforceCSFLayerCfg)
    enforce_csf_layer_post: EnforceCSFLayerCfg = field(default_factory=EnforceCSFLayerCfg)
    falx: FalxCfg = field(default_factory=FalxCfg)
    tentorium: TentoriumCfg = field(default_factory=TentoriumCfg)
    inf_lat_vent_horns: InfLatVentHornsCfg = field(default_factory=InfLatVentHornsCfg)
    min_thickness_v4: MinThicknessV4Cfg = field(default_factory=MinThicknessV4Cfg)
    connected_ventricles: ConnectedVentriclesCfg = field(default_factory=ConnectedVentriclesCfg)
    tight_ventricles: TightVentriclesCfg = field(default_factory=TightVentriclesCfg)
    csf_around_tentorium: EnforceCSFAroundCfg = field(default_factory=EnforceCSFAroundCfg)
    csf_around_falx: EnforceCSFAroundCfg = field(default_factory=EnforceCSFAroundCfg)
    extend_brainstem_caudally: ExtendBrainstemCaudallyCfg = field(default_factory=ExtendBrainstemCaudallyCfg)
    coarsen_surface: CoarsenSurfaceCfg = field(default_factory=CoarsenSurfaceCfg)
    misc: PipelineMiscCfg = field(default_factory=PipelineMiscCfg)

    @classmethod
    def from_toml(cls, path: str | Path) -> "SegmentationConfig":
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationConfig":
        return _build(cls, d)


def _build(cls, d):
    """Recursively build a dataclass from a (partial) dict, rejecting unknown keys."""
    if not is_dataclass(cls):
        return d
    field_map = {f.name: f for f in fields(cls)}
    extra = set(d) - set(field_map)
    if extra:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(extra)}")
    kwargs = {}
    for name, f in field_map.items():
        if name not in d:
            continue
        sub_default = f.default_factory() if callable(f.default_factory) else None
        if is_dataclass(sub_default):
            kwargs[name] = _build(type(sub_default), d[name])
        else:
            kwargs[name] = d[name]
    return cls(**kwargs)
