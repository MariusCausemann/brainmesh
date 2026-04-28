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
brainmesh-surface testdata/sub1_gouhfi_hybrid_seg.nii.gz -o results/

# Step 2: surface → tetrahedral mesh
brainmesh-mesh results/surf.vtk -o results/

# Step 3 (optional): subdivide the SAS by cortical parcellation, then re-label CSF tets
brainmesh-subdivideSAS --segfile results/seg.nii.gz --parcfile parc.nii.gz -o results/sas_subdivide.nii.gz
brainmesh-remark-sas results/mesh_marked.vtk results/sas_subdivide.nii.gz -o results/mesh_marked_sas.vtk

# Step 4 (optional): convert to quadratic and snap boundaries to the surface mesh
brainmesh-curve-mesh -i results/mesh_marked.vtk -t results/surf.vtk -o results/mesh_curved.vtk

# Extract and combine interface and boundary facets into a single mesh
brainmesh-mark-facets results/mesh_marked_sas.vtk -o results/facets.vtk
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

The output carries a single `interface_id` cell array: interface facets are encoded as `min(a, b) * 100000 + max(a, b)` for the two adjacent region markers (decode with `a, b = id // 100000, id % 100000`); boundary facets use their region marker directly; spinal facets are encoded as `0`.

`brainmesh-curve-mesh` accepts the following options:

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | *(required)* | Linear tetrahedral mesh (`.vtk`, `.vtu`, …) |
| `-t` / `--target` | *(required)* | Target surface for snapping (`.vtk`, `.stl`, `.ply`, `.obj`, …) |
| `-o` / `--output` | `snapped_output.vtk` | Output path |
| `--min-quality-factor` | `0.8` | Minimum allowed quality as a fraction of the original mesh's minimum — nodes are relaxed if snapping would drop below this threshold |

### Python API

```python
from brainmesh.pipeline import segmentation_to_surface, surface_to_mesh
from brainmesh.curved_mesh import convert_to_quadratic, adaptive_snap_boundaries

surf = segmentation_to_surface("testdata/sub1.nii.gz", out_dir="results")
mesh = surface_to_mesh("results/surf.vtk", out_dir="results")

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
  segmentation.py — voxel-label cleanup (CSF enforcement, smoothing, ...)
  anatomy.py      — anatomically-specific ops (falx, tentorium, brainstem, ...)
  surface.py      — surface extraction, decimation, label transfer
  mesh.py         — tetrahedral meshing and winding-number marking
  curved_mesh.py  — quadratic conversion and boundary snapping
  pipeline.py     — high-level end-to-end functions
  cli.py          — argparse entry points
```

## Running tests

```bash
pytest                  # fast unit tests only
pytest -m slow          # include integration tests (requires pytetwild)
```

## License

MIT — see [LICENSE](LICENSE).
