"""Command-line entry points for brainmesh pipelines."""
import argparse


def surface_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Clean up a brain segmentation and extract a multi-boundary surface mesh."
    )
    parser.add_argument("seg", help="Input segmentation (.nii.gz)")
    parser.add_argument("-o", "--out-dir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of numba threads (default: 8)")
    args = parser.parse_args(argv)

    from brainmesh.pipeline import segmentation_to_surface
    segmentation_to_surface(args.seg, out_dir=args.out_dir, numba_threads=args.threads)


def mesh_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tetrahedralise a surface mesh and mark cells with anatomical labels."
    )
    parser.add_argument("surf", help="Input surface mesh (.vtk) with boundary_labels")
    parser.add_argument("-o", "--out-dir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--edge-length-fac", type=float, default=0.05,
                        help="Target edge length as fraction of bbox diagonal (default: 0.05)")
    parser.add_argument("--stop-energy", type=float, default=10.0,
                        help="fTetWild stop energy threshold (default: 10)")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="fTetWild envelope size (default: 1e-3)")
    parser.add_argument("--coarsen", action="store_true",
                        help="Enable fTetWild coarsening pass")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress fTetWild log output")
    args = parser.parse_args(argv)

    from brainmesh.pipeline import surface_to_mesh
    surface_to_mesh(
        args.surf,
        out_dir=args.out_dir,
        edge_length_fac=args.edge_length_fac,
        stop_energy=args.stop_energy,
        epsilon=args.epsilon,
        coarsen=args.coarsen,
        quiet=args.quiet,
    )

def curve_mesh_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert linear tetrahedra to quadratic and snap boundaries to a target surface."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to input linear tetrahedral mesh (.vtk, .vtu, ...)")
    parser.add_argument("-t", "--target", type=str, required=True,
                        help="Path to target high-res surface mesh (.vtk, .stl, .ply, .obj, ...)")
    parser.add_argument("-o", "--output", type=str, default="snapped_output.vtk",
                        help="Path to save the output mesh (default: snapped_output.vtk)")
    parser.add_argument("--min-quality-factor", type=float, default=0.8,
                        help="Minimum allowed quality as a fraction of the original mesh's"
                             " minimum quality (default: 0.8)")
    args = parser.parse_args(argv)

    import pyvista as pv
    from brainmesh.curved_mesh import (
        adaptive_snap_boundaries,
        convert_to_quadratic,
        print_quality_stats,
    )

    input_mesh = pv.read(args.input)
    target_surface = pv.read(args.target)

    print_quality_stats(input_mesh, "1. Original Linear Mesh")

    print("Converting to 2nd-order quadratic tetrahedra...")
    quad_mesh = convert_to_quadratic(input_mesh)
    orig_q = print_quality_stats(quad_mesh, "2. Unsnapped Quadratic Mesh")

    print("Snapping boundary nodes to target surface...")
    adaptive_snap_boundaries(quad_mesh, target_surface,
                              min_quality=orig_q.min() * args.min_quality_factor)
    print_quality_stats(quad_mesh, "3. Snapped Quadratic Mesh")

    quad_mesh.save(args.output)
    print(f"Success! Snapped mesh saved to: {args.output}")


def mark_facets_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract and save internal interface facets and external boundary facets of a marked tetrahedral mesh."
    )
    parser.add_argument("mesh", help="Input marked tetrahedral mesh (.vtk, .vtu, ...)")
    parser.add_argument("-o", "--output", default="facets.vtk",
                        help="Output file for combined facet mesh (default: facets.vtk)")
    parser.add_argument("--label-array", default="marker",
                        help="Cell data array used for region markers (default: marker)")
    parser.add_argument("--max-angle", type=float, default=10.0,
                        help="Max angle (degrees) from downward for spinal boundary detection (default: 10)")
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="Max z-distance from the lowest boundary face for spinal detection, in mesh units (default: 0.5)")
    args = parser.parse_args(argv)

    import pyvista as pv
    from brainmesh.mesh import mark_facets

    mesh = pv.read(args.mesh)
    combined = mark_facets(mesh, label_array=args.label_array,
                           max_angle=args.max_angle, max_distance=args.max_distance)
    combined.save(args.output)
    import numpy as np
    from brainmesh.labels import SPINAL_ID
    ids = combined.cell_data["interface_id"]
    print(f"Saved {(ids >= 100000).sum()} interface + {((ids > 0) & (ids < 100000) & (ids != SPINAL_ID)).sum()} boundary"
          f" + {(ids == SPINAL_ID).sum()} spinal facets → {args.output}")


def remark_sas_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-label CSF tetrahedra with subdivided SAS labels from a NIfTI parcellation."
    )
    parser.add_argument("mesh", help="Marked tetrahedral mesh (.vtk, .vtu, ...)")
    parser.add_argument("sas", help="SAS subdivision NIfTI (.nii.gz) from brainmesh-subdivide-sas")
    parser.add_argument("-o", "--output", default="mesh_marked_sas.vtk",
                        help="Output path for the re-labelled mesh (default: mesh_marked_sas.vtk)")
    parser.add_argument("--label-array", default="marker",
                        help="Cell data array used for region markers (default: marker)")
    args = parser.parse_args(argv)

    import numpy as np
    import pyvista as pv
    from brainmesh.labels import Label
    from brainmesh.mesh import remark_csf_with_sas

    mesh = pv.read(args.mesh)
    remark_csf_with_sas(mesh, args.sas, label_array=args.label_array)
    mesh.save(args.output)

    markers = mesh.cell_data[args.label_array]
    n_csf = (markers == Label.CSF).sum()
    n_sas = np.sum((markers > 0) & (markers != Label.CSF))
    print(f"Saved {args.output}: {n_sas} SAS-subdivided + {n_csf} unmarked CSF tets")


def subdivide_SAS(argv=None):
    parser = argparse.ArgumentParser(
        description="Subdivide the SAS by the nearest cortical parcellation label"
    )
    parser.add_argument("--segfile",help="Input segmentation (.nii.gz)")
    parser.add_argument("--parcfile",help="Input parcellation (.nii.gz)")
    parser.add_argument("-o", "--outfile", help="Output file")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of numba threads (default: 8)")
    args = parser.parse_args(argv)

    from brainmesh.pipeline import subdivide_SAS
    import nibabel as nib

    labeled_SAS = subdivide_SAS(args.segfile,args.parcfile, numba_threads=args.threads)
    nib.save(labeled_SAS, args.outfile)
