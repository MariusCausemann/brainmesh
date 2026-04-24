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
