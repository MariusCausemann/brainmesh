"""Command-line entry points for brainmesh pipelines."""
import argparse
from pathlib import Path

def surface_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Clean up a brain segmentation and extract a multi-boundary surface mesh."
    )
    parser.add_argument("seg", help="Input segmentation (.nii.gz)")

    parser.add_argument("--out-seg", default="results/seg.nii.gz",
                        help="Output path for the cleaned segmentation (default: results/seg.nii.gz)")
    parser.add_argument("--out-surf", default="results/surf.vtk",
                        help="Output path for the extracted surface (default: results/surf.vtk)")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to a TOML file with pipeline parameters "
                             "(see configs/default.toml for the full reference).")
    parser.add_argument("--hemisphere-gap", type=int, default=None,
                        help="Override falx.hemisphere_gap (voxel gap forced between hemispheres).")
    parser.add_argument("--cerebrum-cerebellum-gap", type=int, default=None,
                        help="Override tentorium.cerebrum_cerebellum_gap.")
    parser.add_argument("--brainstem-caudal-z-offset", type=int, default=None,
                        help="Override extend_brainstem_caudally.footprint_z_offset.")
    parser.add_argument("--ventricle-jacket-thickness", type=int, default=None,
                        help="Override tight_ventricles.surrounding_layer_thickness.")
    parser.add_argument("--decimation-ratio", type=float, default=None,
                        help="Override coarsen_surface.decimation_ratio.")

    parser.add_argument("--threads", type=int, default=1,
                        help="Number of numba threads (default: 1)")
    args = parser.parse_args(argv)

    from brainmesh.config import SegmentationConfig
    from brainmesh.pipeline import segmentation_to_surface

    cfg = SegmentationConfig.from_toml(args.config) if args.config else SegmentationConfig()
    if args.hemisphere_gap is not None:
        cfg.falx.hemisphere_gap = args.hemisphere_gap
    if args.cerebrum_cerebellum_gap is not None:
        cfg.tentorium.cerebrum_cerebellum_gap = args.cerebrum_cerebellum_gap
    if args.brainstem_caudal_z_offset is not None:
        cfg.extend_brainstem_caudally.footprint_z_offset = args.brainstem_caudal_z_offset
    if args.ventricle_jacket_thickness is not None:
        cfg.tight_ventricles.surrounding_layer_thickness = args.ventricle_jacket_thickness
    if args.decimation_ratio is not None:
        cfg.coarsen_surface.decimation_ratio = args.decimation_ratio

    segmentation_to_surface(
        args.seg,
        out_seg=args.out_seg,
        out_surf=args.out_surf,
        config=cfg,
        numba_threads=args.threads,
    )

def mesh_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tetrahedralise a surface mesh and mark cells with anatomical labels."
    )
    parser.add_argument("surf", help="Input surface mesh (.vtk) with boundary_labels")
    parser.add_argument("-o", "--out-file", 
                        help="Output mesh file")
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
    parser.add_argument("--num-threads", type=int, default=0,
                        help="fTetWild number of threads (default all)")
    args = parser.parse_args(argv)

    from brainmesh.pipeline import surface_to_mesh
    surface_to_mesh(
        args.surf,
        out_file=args.out_file,
        edge_length_fac=args.edge_length_fac,
        stop_energy=args.stop_energy,
        epsilon=args.epsilon,
        coarsen=args.coarsen,
        quiet=args.quiet,
        num_threads=args.num_threads,
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

    from brainmesh.curved_mesh import (
        adaptive_snap_boundaries,
        convert_to_quadratic,
        print_quality_stats,
    )
    from .mesh_optimizer import run_mesh_optimization
    from brainmesh.io import read_mesh, save_mesh

    input_mesh = read_mesh(args.input)
    target_surface = read_mesh(args.target)

    print_quality_stats(input_mesh, "1. Original Linear Mesh")

    print("Converting to 2nd-order quadratic tetrahedra...")
    quad_mesh = convert_to_quadratic(input_mesh)
    quad_mesh.field_data["grid_z_normal"] = quad_mesh.field_data["grid_z_normal"]

    orig_q = print_quality_stats(quad_mesh, "2. Unsnapped Quadratic Mesh")

    print("Snapping boundary nodes to target surface...")
    all_boundary_ids = adaptive_snap_boundaries(quad_mesh, target_surface,
                              min_quality=orig_q.min() * args.min_quality_factor,
                              decay_step=0.1)
    print_quality_stats(quad_mesh, "3. Snapped Quadratic Mesh")

    print("Optimizing Internal Nodes...")

    # Pass the mesh and the boundary IDs to freeze
    quad_mesh = run_mesh_optimization(quad_mesh, boundary_ids=all_boundary_ids, iters=20,
                                      target_quality=0.2, step_factor=0.1)

    print_quality_stats(quad_mesh, "4. Final Optimized Quadratic Mesh")

    #quad_mesh = run_mesh_optimization(quad_mesh, boundary_ids=[], iters=2,
    #                                  target_quality=0.08, step_factor=0.1)

    #print_quality_stats(quad_mesh, "4. Final Optimized Quadratic Mesh")

    save_mesh(quad_mesh, args.output)
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
    parser.add_argument("--max-angle", type=float, default=20.0,
                        help="Max angle (degrees) from downward for spinal boundary detection (default: 20)")
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="Max z-distance from the lowest boundary face for spinal detection, in mesh units (default: 0.5)")
    parser.add_argument("--no-smooth-sas-labels", action="store_true",
                        help="Disable majority-vote smoothing of SAS boundary labels")
    parser.add_argument("--keep-sas-interfaces", action="store_true",
                        help="Keep interfaces between SAS subdivision regions (dropped by default)")
    args = parser.parse_args(argv)

    from brainmesh.io import read_mesh, save_mesh
    from brainmesh.mesh import mark_facets

    mesh = read_mesh(args.mesh)
    combined = mark_facets(mesh, label_array=args.label_array,
                           max_angle=args.max_angle, max_distance=args.max_distance,
                           smooth_sas_labels=not args.no_smooth_sas_labels,
                           ignore_sas_interfaces=not args.keep_sas_interfaces)
    save_mesh(combined, args.output)
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
    from brainmesh.io import read_mesh, save_mesh
    from brainmesh.labels import Label
    from brainmesh.mesh import remark_csf_with_sas

    mesh = read_mesh(args.mesh)
    remark_csf_with_sas(mesh, args.sas, label_array=args.label_array)
    save_mesh(mesh, args.output)

    markers = mesh.cell_data[args.label_array]
    n_csf = (markers == Label.CSF).sum()
    n_sas = np.sum((markers > 0) & (markers != Label.CSF))
    print(f"Saved {args.output}: {n_sas} SAS-subdivided + {n_csf} unmarked CSF tets")


def extract_csf_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract the CSF compartment (SAS + ventricles) and its facets from a marked tet mesh."
    )
    parser.add_argument("mesh", help="Marked tetrahedral mesh (.vtk, .vtu, ...)")
    parser.add_argument("-o", "--output", default="csf_mesh.vtk",
                        help="Output path for the CSF submesh (default: csf_mesh.vtk)")
    parser.add_argument("--facets", default="csf_facets.vtk",
                        help="Output path for the CSF facet mesh (default: csf_facets.vtk)")
    parser.add_argument("--label-array", default="marker",
                        help="Cell data array used for region markers (default: marker)")
    parser.add_argument("--max-angle", type=float, default=10.0,
                        help="Max angle (degrees) from downward for spinal boundary detection (default: 10)")
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="Max z-distance from the lowest boundary face for spinal detection, in mesh units (default: 0.5)")
    parser.add_argument("--no-smooth-sas-labels", action="store_true",
                        help="Disable majority-vote smoothing of SAS boundary labels")
    parser.add_argument("--keep-sas-interfaces", action="store_true",
                        help="Keep interfaces between SAS subdivision regions (dropped by default)")
    args = parser.parse_args(argv)

    import numpy as np
    from brainmesh.io import read_mesh, save_mesh
    from brainmesh.labels import SPINAL_ID
    from brainmesh.mesh import extract_csf

    mesh = read_mesh(args.mesh)
    csf_mesh, facets = extract_csf(
        mesh,
        label_array=args.label_array,
        return_facets=True,
        max_angle=args.max_angle,
        max_distance=args.max_distance,
        smooth_sas_labels=not args.no_smooth_sas_labels,
        ignore_sas_interfaces=not args.keep_sas_interfaces,
    )

    assert np.allclose(csf_mesh.points, facets.points)
    save_mesh(csf_mesh, args.output)
    save_mesh(facets, args.facets)

    ids = facets.cell_data["interface_id"]
    print(f"CSF mesh: {csf_mesh.n_cells} tets → {args.output}")
    print(f"Facets:   {(ids >= 100000).sum()} interface + "
          f"{((ids > 0) & (ids < 100000) & (ids != SPINAL_ID)).sum()} boundary + "
          f"{(ids == SPINAL_ID).sum()} spinal → {args.facets}")


def group_regions_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Group CSF facets into anatomical regions (lobes, tentorium, sagittal sinus, …)."
    )
    parser.add_argument("mesh", help="Marked facet mesh (.vtk, .vtu, ...)")
    parser.add_argument("-o", "--output", default="csf_facets_regions.vtk", type=Path,
                        help="Output path (default: csf_facets_regions.vtk)")
    parser.add_argument("--label-array", default="marker",
                        help="Cell data array used for region markers (default: marker)")
    parser.add_argument("--no-smooth-sas-labels", action="store_true",
                        help="Disable majority-vote smoothing of SAS boundary labels")
    args = parser.parse_args(argv)

    from brainmesh.io import read_mesh, save_mesh
    from brainmesh.mesh import group_csf_facets_by_region
    import numpy as np
    import tomlkit
    
    facets = read_mesh(args.mesh)
    result, region_label_dict = group_csf_facets_by_region(facets)
    assert np.allclose(facets.points, result.points)
    save_mesh(result, args.output)
    label_path = args.output.with_name(f"{args.output.stem}_labels").with_suffix(".toml")
    with open(label_path, "w", encoding="utf-8") as f:
        tomlkit.dump(region_label_dict, f)

    print(f"Saved → {args.output}")


def plot_mesh_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render diagnostic views of a marked tet mesh OR its source surface mesh."
    )
    parser.add_argument("mesh", help="Marked tetrahedral mesh OR labelled surface mesh (.vtk, .vtu, ...)")
    parser.add_argument("-o", "--output", default="mesh_plot.png",
                        help="Output image path (default: mesh_plot.png; format inferred from extension)")
    parser.add_argument("--label-array", default=None,
                        help="Cell data array used for region markers "
                             "(default: 'marker' for tet, 'boundary_labels' for surface)")
    args = parser.parse_args(argv)

    import pyvista as pv
    from pyvista import CellType
    from brainmesh.io import read_mesh
    from brainmesh.plotting import plot_surface_mesh, plot_tet_mesh

    mesh = read_mesh(args.mesh)
    if isinstance(mesh, pv.ImageData):
        plot_tet_mesh(mesh, args.output,
                      label_array=args.label_array or "data")
    elif isinstance(mesh, pv.PolyData):
        plot_surface_mesh(mesh, args.output,
                          label_array=args.label_array or "boundary_labels")
    else:
        # UnstructuredGrid: dispatch by cell type
        is_tet = bool(set(mesh.celltypes.tolist())
                      & {int(CellType.TETRA), int(CellType.QUADRATIC_TETRA)})
        if is_tet:
            plot_tet_mesh(mesh, args.output,
                          label_array=args.label_array or "marker")
        else:
            plot_surface_mesh(mesh.extract_surface(), args.output,
                              label_array=args.label_array or "boundary_labels")
    print(f"Saved → {args.output}")


def plot_facets_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render diagnostic views of a facet mesh into a single image."
    )
    parser.add_argument("facets", type=Path,
                        help="Facet mesh from brainmesh-mark-facets / brainmesh-extract-csf")
    parser.add_argument("-o", "--output", default="facets_plot.png",
                        help="Output image path (default: facets_plot.png; format inferred from extension)")
    parser.add_argument("--no-group", action="store_true",
                        help="Skip grouping by anatomical region")
    parser.add_argument("--labels", type=Path, default=None,
                        help="TOML file mapping {region_name: id} for the legend. "
                             "Defaults to '<facets>_labels.toml' next to the facets file.")
    args = parser.parse_args(argv)

    import tomlkit
    from brainmesh.io import read_mesh
    from brainmesh.plotting import plot_facet_mesh

    labels_path = args.labels or args.facets.with_name(
        f"{args.facets.stem}_labels"
    ).with_suffix(".toml")
    region_labels = None
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            region_labels = dict(tomlkit.load(f))
    elif args.labels is not None:
        parser.error(f"Labels file not found: {labels_path}")

    plot_facet_mesh(read_mesh(args.facets), args.output,
                    group=not args.no_group, region_labels=region_labels)
    print(f"Saved → {args.output}")


def subdivide_SAS(argv=None):
    parser = argparse.ArgumentParser(
        description="Subdivide the SAS by the nearest cortical parcellation label"
    )
    parser.add_argument("--segfile",help="Input segmentation (.nii.gz)")
    parser.add_argument("--parcfile",help="Input parcellation (.nii.gz)")
    parser.add_argument("-o", "--outfile", help="Output file")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of numba threads (default: 1)")
    args = parser.parse_args(argv)

    from brainmesh.pipeline import subdivide_SAS
    import nibabel as nib

    labeled_SAS = subdivide_SAS(args.segfile,args.parcfile, numba_threads=args.threads)
    nib.save(labeled_SAS, args.outfile)
