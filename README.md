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
```

### Python API

```python
from brainmesh.pipeline import segmentation_to_surface, surface_to_mesh

surf = segmentation_to_surface("testdata/sub1.nii.gz", out_dir="results")
mesh = surface_to_mesh("results/surf.vtk", out_dir="results")
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
