# brainmesh

Create tetrahedral brain meshes from FreeSurfer/SynthSeg segmentations for FEM simulations.

## What it does

Starting from a labelled NIfTI segmentation (e.g. SynthSeg at 0.5 mm), `brainmesh`:

1. **Cleans the segmentation** — fills CSF holes, adds falx and tentorium, repairs ventricular connectivity, fixes the brainstem/spine interface.
2. **Extracts a multi-boundary surface mesh** using PyVista's `contour_labels`.
3. **Tetrahedralises** the surface with fTetWild (`pytetwild`).
4. **Marks** each tetrahedron with its anatomical label via the winding-number method.

## Installation

Requires a conda environment for packages not available on PyPI (vtk ≥ 9.6):

```bash
conda env create -f environment.yml
conda activate brainmesh
pip install -e ".[dev]"
```

## Usage

### Command line

```bash
# Step 1: segmentation → surface mesh
brainmesh-surface testdata/sub1.nii.gz --out-seg results/seg.nii.gz --out-surf results/surf.vtk

# With a subject-specific parameter file (copy configs/default.toml and edit)
brainmesh-surface testdata/sub1.nii.gz --config my_subject.toml --out-surf results/surf.vtk

# The most commonly tuned parameters are also available as direct CLI flags:
brainmesh-surface testdata/sub1.nii.gz \
    --hemisphere-gap 8 \
    --cerebrum-cerebellum-gap 4 \
    --brainstem-caudal-z-offset 20 \
    --out-surf results/surf.vtk

# Step 2: surface → tetrahedral mesh
brainmesh-mesh results/surf.vtk -o results/

# Step 3 (optional): subdivide the SAS by cortical parcellation, then re-label CSF tets
brainmesh-subdivideSAS --segfile results/seg.nii.gz --parcfile parc.nii.gz -o results/sas_subdivide.nii.gz
brainmesh-remark-sas results/mesh_marked.vtk results/sas_subdivide.nii.gz -o results/mesh_marked_sas.vtk

# Step 4 (optional): convert to quadratic and snap boundaries to the surface mesh
brainmesh-curve-mesh -i results/mesh_marked.vtk -t results/surf.vtk -o results/mesh_curved.vtk

# Extract and combine interface and boundary facets into a single mesh
brainmesh-mark-facets results/mesh_marked_sas.vtk -o results/facets.vtk

# Extract the CSF compartment as a submesh together with its facets
# (facets are computed on the full mesh so CSF-tissue interfaces are preserved)
brainmesh-extract-csf results/mesh_marked_sas.vtk -o results/csf_mesh.vtk --facets results/csf_facets.vtk
```

`brainmesh-remark-sas` accepts the following options:

| Flag | Default | Description |
|------|---------|-------------|
| `mesh` | *(required)* | Marked tetrahedral mesh (`.vtk`, `.vtu`, …) |
| `sas` | *(required)* | SAS subdivision NIfTI (`.nii.gz`) from `brainmesh-subdivideSAS` |
| `-o` / `--output` | `mesh_marked_sas.vtk` | Output path for the re-labelled mesh |
| `--label-array` | `marker` | Cell data array used for region markers |

`brainmesh-mark-facets` accepts the following options:

| Flag | Default | Description |
|------|---------|-------------|
| `mesh` | *(required)* | Marked tetrahedral mesh (`.vtk`, `.vtu`, …) |
| `-o` / `--output` | `facets.vtk` | Output path for the combined facet mesh |
| `--label-array` | `marker` | Cell data array used for region markers |

The output carries a single `interface_id` cell array using the following scheme:

| `interface_id` range | Meaning | Decode |
|---|---|---|
| `99` | Spinal opening (`SPINAL_ID`) | — |
| `2–12035` | Outer boundary | value = adjacent region marker |
| `≥ 100000` | Internal interface | `a, b = divmod(id, 100000)` |

**Volume marker IDs** (tet mesh `marker` array):

| Range | Meaning |
|---|---|
| `2–77` | FreeSurfer aseg anatomy (unchanged) |
| `11001–11035` | LH SAS parcels — decode: `fs_aparc = marker - 10000` |
| `12001–12035` | RH SAS parcels — decode: `fs_aparc = marker - 10000` |

`brainmesh-extract-csf` accepts the following options:

| Flag | Default | Description |
|------|---------|-------------|
| `mesh` | *(required)* | Marked tetrahedral mesh (`.vtk`, `.vtu`, …) |
| `-o` / `--output` | `csf_mesh.vtk` | Output path for the CSF submesh |
| `--facets` | `csf_facets.vtk` | Output path for the CSF facet mesh |
| `--label-array` | `marker` | Cell data array used for region markers |
| `--max-angle` | `10.0` | Max angle (degrees) from downward for spinal boundary detection |
| `--max-distance` | `0.5` | Max z-distance from the lowest boundary face for spinal detection (mesh units) |

The facet mesh uses the same `interface_id` scheme as `brainmesh-mark-facets`, but is restricted to facets bounding the CSF compartment — including CSF-to-tissue interfaces with their full encoding (e.g. `min(CSF,WM)*100000+max(CSF,WM)`).

`brainmesh-curve-mesh` accepts the following options:

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | *(required)* | Linear tetrahedral mesh (`.vtk`, `.vtu`, …) |
| `-t` / `--target` | *(required)* | Target surface for snapping (`.vtk`, `.stl`, `.ply`, `.obj`, …) |
| `-o` / `--output` | `snapped_output.vtk` | Output path |
| `--min-quality-factor` | `0.8` | Minimum allowed quality as a fraction of the original mesh's minimum — nodes are relaxed if snapping would drop below this threshold |

### Tuning the segmentation pipeline

Every numerical parameter in the segmentation cleanup steps is configurable via a TOML file. `configs/default.toml` is a fully-annotated reference that reproduces the built-in defaults — copy it, edit only the values you want to change, and pass it with `--config`:

```toml
# my_subject.toml — only overriding the values that need to change
[falx]
hemisphere_gap = 8           # larger gap for a brain with bridging tissue

[tentorium]
cerebrum_cerebellum_gap = 5  # adjust separation distance

[extend_brainstem_caudally]
footprint_z_offset = 22      # sample footprint higher up the brainstem
```

All other parameters retain their defaults. Missing sections and missing keys within a section both fall back to defaults. Passing an unknown key raises an error.

`brainmesh-surface` also accepts a handful of the most commonly tuned parameters as direct CLI flags (these override the config file if both are given):

| Flag | Config key | Description |
|---|---|---|
| `--config` | — | Path to TOML config file |
| `--hemisphere-gap` | `falx.hemisphere_gap` | Voxel gap forced between hemispheres |
| `--cerebrum-cerebellum-gap` | `tentorium.cerebrum_cerebellum_gap` | Gap between cerebrum and cerebellum |
| `--brainstem-caudal-z-offset` | `extend_brainstem_caudally.footprint_z_offset` | Z-offset for brainstem extrusion footprint |
| `--ventricle-jacket-thickness` | `tight_ventricles.surrounding_layer_thickness` | Tissue jacket around ventricles |
| `--decimation-ratio` | `coarsen_surface.decimation_ratio` | Triangle reduction ratio for decimated output |

### Python API

```python
from brainmesh.pipeline import segmentation_to_surface, surface_to_mesh
from brainmesh.config import SegmentationConfig
from brainmesh.curved_mesh import convert_to_quadratic, adaptive_snap_boundaries

# Use built-in defaults
surf = segmentation_to_surface("testdata/sub1.nii.gz",
                               out_seg="results/seg.nii.gz",
                               out_surf="results/surf.vtk")

# Load from a TOML file
cfg = SegmentationConfig.from_toml("my_subject.toml")
surf = segmentation_to_surface("testdata/sub1.nii.gz",
                               out_surf="results/surf.vtk",
                               config=cfg)

# Or override individual fields programmatically
cfg = SegmentationConfig()
cfg.falx.hemisphere_gap = 8
cfg.tentorium.territory_smoothing_sigma = 15.0
surf = segmentation_to_surface("testdata/sub1.nii.gz", config=cfg)

mesh = surface_to_mesh("results/surf.vtk", out_file="results/mesh.vtk")

# Optionally curve the mesh
quad_mesh = convert_to_quadratic(mesh)
adaptive_snap_boundaries(quad_mesh, surf)
quad_mesh.save("results/mesh_curved.vtk")
```

## Package layout

```
src/brainmesh/
  labels.py       — FreeSurfer/SynthSeg label definitions (Label namedtuple)
  io.py           — nibabel ↔ PyVista conversion, upsampling
  config.py       — SegmentationConfig dataclasses + TOML loader
  segmentation.py — voxel-label cleanup (CSF enforcement, smoothing, ...)
  anatomy.py      — anatomically-specific ops (falx, tentorium, brainstem, ...)
  surface.py      — surface extraction, decimation, label transfer
  mesh.py         — tetrahedral meshing and winding-number marking
  curved_mesh.py  — quadratic conversion and boundary snapping
  pipeline.py     — high-level end-to-end functions
  cli.py          — argparse entry points

configs/
  default.toml    — annotated reference config (copy & edit per subject)
```

## Running tests

```bash
pytest                  # fast unit tests only
pytest -m slow          # include integration tests (requires pytetwild)
```

## License

MIT — see [LICENSE](LICENSE).
