"""End-to-end demo: build a synthetic brain phantom and run the full
brainmesh pipeline (cleanup -> surfaces -> tets -> facets) on it.

Usage:
    python examples/phantom_pipeline.py
    python examples/phantom_pipeline.py -o /tmp/phantom_out --shape 150
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from brainmesh.mesh import (
    extract_csf,
    mark_boundary_facets,
    mark_interface_facets,
)
from brainmesh.phantom import make_phantom_seg
from brainmesh.pipeline import segmentation_to_surface, surface_to_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out-dir", default="phantom_results",
                        help="Output directory (default: phantom_results)")
    parser.add_argument("--shape", type=int, default=360,
                        help="Cubic volume side length in voxels (default: 360)")
    parser.add_argument("--spacing", type=float, default=0.5,
                        help="Voxel spacing in mm (default: 0.5)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of numba threads")
    parser.add_argument("--edge-length-fac", type=float, default=0.05,
                        help="fTetWild target edge length")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> Generating phantom segmentation ({args.shape}^3 @ {args.spacing} mm)")
    img = make_phantom_seg(shape=(args.shape,) * 3, spacing=args.spacing)
    seg_path = out_dir / "phantom_seg.nii.gz"
    nib.save(img, seg_path)
    print(f"    Saved {seg_path}")

    print("==> Step 1: cleanup + surface extraction")
    segmentation_to_surface(seg_path, out_dir=out_dir, numba_threads=args.threads)

    print("==> Step 2: tetrahedralisation + marking")
    mesh = surface_to_mesh(out_dir / "surf.vtk", out_dir=out_dir,
                           edge_length_fac=args.edge_length_fac, quiet=True)

    print("==> Step 3: extract CSF compartment")
    csf = extract_csf(mesh)
    csf.save(out_dir / "csf_compartment.vtk")

    print("==> Step 4: mark facets (interfaces + outer boundaries)")
    interfaces = mark_interface_facets(mesh)
    boundaries = mark_boundary_facets(mesh)
    interfaces.save(out_dir / "interface_facets.vtp")
    boundaries.save(out_dir / "boundary_facets.vtp")

    regions = sorted(int(m) for m in np.unique(mesh["marker"]))
    print()
    print("Summary")
    print(f"  Tetrahedra:        {mesh.n_cells}")
    print(f"  Region markers:    {regions}")
    print(f"  CSF compartment:   {csf.n_cells} tets")
    print(f"  Interface facets:  {interfaces.n_cells}")
    print(f"  Boundary facets:   {boundaries.n_cells}")


if __name__ == "__main__":
    main()
