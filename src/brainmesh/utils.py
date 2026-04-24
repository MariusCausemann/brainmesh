import pyvista as pv
import numpy as np
import igl
import nbmorph
from .synthseg_labels import Label, VENTRICLE_LABELS
from nbmorph import dilate_labels_spherical as dilate
from nbmorph import erode_labels_spherical as erode
from numba import njit
import nibabel.processing as nibp
import skimage.morphology as skim
import skimage
from edt import edt
from scipy.spatial import cKDTree
from .decorators import track_voxel_changes, plot_voxel_changes, time_func
from scipy import ndimage as ndi

def transfer_labels(source_mesh, target_mesh, label_name="boundary_labels"):
    """
    Transfers cell labels from a dense source mesh to a coarse target mesh
    using Nearest Neighbor interpolation.
    """
    tree = cKDTree(source_mesh.cell_centers().points)
    dist, idx = tree.query(target_mesh.cell_centers().points, k=1)
    target_mesh[label_name] = source_mesh[label_name][idx]
    return target_mesh

@time_func
def upsample_nib(img, factor=2, order=0):
    """
    Upsamples a nib image by a given factor while preserving the 
    physical bounding box and affine orientation.
    
    Parameters:
        img (nibabel.Nifti1Image): The input image.
        factor (float): The upsampling factor (e.g., 2 means double resolution).
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic).
        
    Returns:
        nibabel.Nifti1Image: The upsampled image.
    """
    target_shape = list(img.shape)
    for i in range(min(3, len(target_shape))):
        target_shape[i] = int(np.round(target_shape[i] * factor))
    
    target_affine = img.affine.copy()
    
    # Scale the voxel spacing down by the factor
    target_affine[:3, :3] /= factor
    
    # Calculate the exact origin shift to keep the outer bounding box locked.
    # Formula: (1/factor - 1) / 2
    offset_multiplier = (1.0 / factor - 1.0) / 2.0
    offset = img.affine[:3, :3] @ np.array([offset_multiplier, offset_multiplier, offset_multiplier])
    target_affine[:3, 3] += offset
    
    # 3. Resample and return
    return nibp.resample_from_to(
        img, 
        to_vox_map=(tuple(target_shape), target_affine), 
        order=order
    )

@time_func
def nibabel_to_pyvista(nib_img, scalar_name="data"):
    """
    Converts a nibabel image object into a pyvista ImageData (UniformGrid).
    
    Parameters:
    - nib_img: nibabel image object (e.g., loaded via nib.load())
    - scalar_name: string name for the data array in PyVista
    
    Returns:
    - grid: pyvista.ImageData object containing the volume
    """
    data = nib_img.get_fdata()
    if data.ndim > 3:
        data = data[..., 0]
        
    affine = nib_img.affine
    spacing = np.array(nib_img.header.get_zooms()[:3])
    
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1
    grid.spacing = spacing
    
    try:
        grid.direction_matrix = affine[:3, :3] / spacing
        grid.origin = affine[:3, 3] - (grid.direction_matrix @ spacing) / 2.0
    except AttributeError:
        grid.origin = affine[:3, 3] - spacing / 2.0
        
    grid.cell_data[scalar_name] = data.flatten(order='F')
    
    return grid

@time_func
def mark_mesh(mesh, surf):
    """
    Mark the mesh using a windingnumber of the input surfaces.
    
    Parameters:
    - mesh: tetrahedralized pyivsta.UnStructuredGrid
    - seg: string name for the data array in PyVista
    
    Returns:
    - grid: pyivsta.UnStructuredGrid with "marker" cell.
    """
    labels = np.unique(surf["boundary_labels"])
    marker = np.zeros(mesh.n_cells, dtype=np.int32)
    query_points = np.array(mesh.cell_centers().points)
    F_global = np.array(surf.faces.reshape(-1, 4)[:, 1:])
    V_global = np.array(surf.points)
    blabels = surf.cell_data["boundary_labels"]
    
    for i, cid in enumerate(labels):
        if i==0: continue

        #  triangles where the label is inside (normals point outwards)
        mask_out = blabels[:, 0] == cid
        F_out = F_global[mask_out]
        
        # 2. triangles where the label is outside (normals point inwards)
        mask_in = blabels[:, 1] == cid
        F_in = F_global[mask_in]
        
        # 3. flip the inward triangles
        if len(F_in) > 0:
            F_in_flipped = F_in[:, [0, 2, 1]]
            F_label = np.vstack((F_out, F_in_flipped))
        else:
            F_label = F_out
        fwn = igl.fast_winding_number(V_global, F_label, 
                                           query_points)

        marker = np.where(marker == 0, cid * np.isclose(fwn, -1, rtol=0.5), marker)
    #marker = np.argmax(-1 * fwn, axis=0)
    mesh["marker"] = marker
    mesh = mesh.extract_cells(marker>0)
    return mesh

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit
def enforce_csf_layer(data, thickness=1):
    mask = (data > 0) & (data!=Label.CSF) & (data!=Label.TENTORIUM) & (data!=Label.FALX) & (data!=Label.BRAIN_STEM)
    dilated_mask = dilate(mask, radius=thickness, struct_sequence="B")
    mask += (data == Label.BRAIN_STEM)
    return np.where(dilated_mask > mask, Label.CSF, data)

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit
def enforce_cortex_layer(data, thickness=1):

    # exclude corpus callosum
    cc_interface = (dilate(data==Label.LEFT_CEREBRAL_WHITE_MATTER, radius=2) & 
                    dilate(data==Label.RIGHT_CEREBRAL_WHITE_MATTER, radius=2))
    
    cc_exclusion_mask = dilate(cc_interface, radius=10)

    for cortex_id, wm_id in [(Label.LEFT_CEREBRAL_CORTEX,
                              Label.LEFT_CEREBRAL_WHITE_MATTER),
                             (Label.RIGHT_CEREBRAL_CORTEX,
                              Label.RIGHT_CEREBRAL_WHITE_MATTER)]:
        mask = data == wm_id
        dilated_mask = dilate(mask, radius=thickness, struct_sequence="BD")
        dilated_mask *= ~cc_exclusion_mask
        data = np.where(np.logical_and(dilated_mask,
                                       np.isin(data, [0, Label.CSF])),
                                       cortex_id, data)
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit
def enforce_wm_thickness(data, thickness=1):
    for wm_id in [Label.LEFT_CEREBRAL_WHITE_MATTER,
                  Label.RIGHT_CEREBRAL_WHITE_MATTER]:
        data = enforce_min_thickness(data, wm_id, thickness)
    return data

@njit
def enforce_min_thickness(data, label, radius, s="B"):
    mask = data==label
    opened = nbmorph.erode_labels_spherical(mask, 
                                            radius=radius,
                                              struct_sequence=s)
    opened = nbmorph.dilate_labels_spherical(opened, 
                                             radius=radius,
                                             struct_sequence=s)
    diff = mask ^ opened
    dil = nbmorph.dilate_labels_spherical(diff, radius=radius, 
                                          struct_sequence=s)
    data = np.where(dil, label, data)
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
@njit
def carve_gruves(data, radius):
    return enforce_min_thickness(data, Label.CSF, radius=radius)

@njit
def separate_labels(data, l1, l2, dist, newlabel=Label.CSF, except_labels=None,
                    except_region=None):
    m1 = nbmorph.dilate_labels_spherical(np.isin(data,l1), dist)
    m2 = nbmorph.dilate_labels_spherical(np.isin(data,l2), dist)
    if except_region is None: except_region = np.zeros(shape=data.shape, dtype=np.bool)
    if except_labels is None:
        return np.where(m1 & m2 & ~except_region, newlabel, data)
    return np.where(m1 & m2 &  ~except_region & ~np.isin(data, except_labels), newlabel, data)

@njit
def separate_hemispheres(data, distance=4):
    # exclude corpus callosum and ventricle regions
    cc_interface = (dilate(data==Label.LEFT_CEREBRAL_WHITE_MATTER, radius=2) & 
                    dilate(data==Label.RIGHT_CEREBRAL_WHITE_MATTER, radius=2))
    
    cc_exclusion_mask = dilate(cc_interface + data==Label.THIRD_VENTRICLE, radius=20)

    return separate_labels(data, [Label.RIGHT_CEREBRAL_CORTEX], 
                          [Label.LEFT_CEREBRAL_CORTEX], distance,
                           except_region=cc_exclusion_mask
                           )

@njit
def separate_cerebellum_and_cerebrum(data, distance=4):
    return separate_labels(data, 
                           [Label.RIGHT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_CORTEX], 
                            [Label.RIGHT_CEREBELLUM_CORTEX, Label.LEFT_CEREBELLUM_CORTEX],
                            distance, except_labels=[np.uint8(70)])

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def solidify_csf(data):
    mask = data > 0
    closed_mask = nbmorph.close_labels_spherical(mask, radius=5, iterations=1)
    seal = dilate(closed_mask) ^ closed_mask
    holes = ndi.binary_fill_holes(mask + seal) & ~(mask + dilate(seal, radius=1))
    data[holes] = Label.CSF
    return data

@njit
def diamond_mode_filter(data):
    """
    Applies a 3D diamond (von Neumann) mode filter.
    The footprint is 7 voxels: the center and its 6 face-sharing neighbors.
    Replaces each voxel with the majority label in its neighborhood.
    """
    # Create a copy so we don't bleed updated values into adjacent calculations
    out = data.copy()
    nx, ny, nz = data.shape
    
    # The 7 voxels: Center (0,0,0) + 6 face-neighbors
    offsets = ((0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))
    
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                
                # Gather the 7 labels in the diamond neighborhood
                neighbors = np.zeros(7, dtype=data.dtype)
                for idx in range(7):
                    dx, dy, dz = offsets[idx]
                    neighbors[idx] = data[i+dx, j+dy, k+dz]
                
                # Default the winning label to the center voxel's current label
                best_label = neighbors[0]
                max_count = 0
                
                # Tally votes for each unique candidate in the neighborhood
                for idx in range(7):
                    candidate_label = neighbors[idx]
                    
                    # Count how many times this candidate appears
                    count = 0
                    for jdx in range(7):
                        if neighbors[jdx] == candidate_label:
                            count += 1
                            
                    # Update if this candidate has strictly more votes.
                    # Using '>' ensures that in a tie, the first checked max 
                    # (which is biased toward the center voxel) wins.
                    if count > max_count:
                        max_count = count
                        best_label = candidate_label
                        
                out[i, j, k] = best_label
                
    return out

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def create_falx(data, hemisphere_distance=4, sigma=20):
    from skimage.filters import gaussian

    data = separate_hemispheres(data, distance=hemisphere_distance)

    right_mask = np.isin(data, [Label.RIGHT_CEREBRAL_CORTEX, 
                                Label.RIGHT_CEREBRAL_WHITE_MATTER])
    left_mask = np.isin(data, [Label.LEFT_CEREBRAL_CORTEX,
                               Label.LEFT_CEREBRAL_WHITE_MATTER])
    
    smooth_right = gaussian(right_mask.astype(np.float32), sigma=sigma)
    smooth_left = gaussian(left_mask.astype(np.float32), sigma=sigma)
    right_territory = smooth_right > smooth_left

    falx_mask = dilate(right_territory, radius=1) ^ right_territory
    falx_mask += dilate(~right_territory, radius=1) ^ (~right_territory)

    falx_mask[~dilate(right_mask + left_mask, radius=20)] = 0
    falx_mask[~nbmorph.close_labels_spherical(data>0, radius=1)] = 0

    cc_interface = (dilate(data==Label.LEFT_CEREBRAL_WHITE_MATTER, radius=2) & 
                    dilate(data==Label.RIGHT_CEREBRAL_WHITE_MATTER, radius=2))
    
    exl_mask = np.isin(data, [Label.RIGHT_VENTRAL_DC,
                             Label.LEFT_VENTRAL_DC,
                             Label.RIGHT_LATERAL_VENTRICLE, 
                             Label.LEFT_LATERAL_VENTRICLE])
    exl_mask += cc_interface
    
    # keep a gap above the corpus calossus, and exclude ventricles
    falx_mask[dilate(exl_mask, radius=4)] = 0

    cerebellum_mask = np.isin(data, [Label.LEFT_CEREBELLUM_CORTEX, Label.RIGHT_CEREBELLUM_CORTEX])
    # dont change the cerebellum
    falx_mask[dilate(cerebellum_mask, radius=2)] = 0
    # leave the CSF part around the "nose" of V3
    falx_mask[dilate(data==Label.THIRD_VENTRICLE, radius=30)] = 0
    data[falx_mask] = Label.FALX
    enforce_csf_around_falx(data, radius=1)
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def create_tentorium(data, distance=4, sigma=6):
    from skimage.filters import gaussian
    data = separate_cerebellum_and_cerebrum(data, distance=distance)
    cer_mask = np.isin(data, [Label.RIGHT_CEREBRAL_CORTEX, Label.LEFT_CEREBRAL_CORTEX,
                              Label.RIGHT_CEREBRAL_WHITE_MATTER, Label.LEFT_CEREBRAL_WHITE_MATTER])
    ceb_mask = np.isin(data, [Label.LEFT_CEREBELLUM_CORTEX, Label.RIGHT_CEREBELLUM_CORTEX])

    smooth_cer = gaussian(cer_mask.astype(np.float32), sigma=sigma)
    smooth_ceb = gaussian(ceb_mask.astype(np.float32), sigma=sigma)
    phantom_ceb = gaussian(ceb_mask.astype(np.float32), sigma=sigma * 3)
    print(smooth_cer.max())
    print(smooth_ceb.max())
    print(phantom_ceb.max())
    cer_territory = smooth_cer > np.maximum(smooth_ceb, phantom_ceb)

    tent_mask = dilate(cer_territory, radius=1) ^ cer_territory

    # keep tentorium away from frontal area of the brainstem
    tent_mask[~dilate(ceb_mask + cer_mask, radius=12)] = 0
    # keep tentorium away from brainstem
    tent_mask[dilate(data == Label.BRAIN_STEM, radius=10)] = 0
    tent_mask = dilate(tent_mask, radius=1)

    # keep it within the original mask
    tent_mask[~nbmorph.close_labels_spherical(data>0, radius=1)] = 0

    data[tent_mask] = Label.TENTORIUM
    enforce_csf_around_tentorium(data, radius=1)
    # remove falx if below tentorium
    data[(~cer_territory) & (data==Label.FALX)] = Label.CSF
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_csf_around_tentorium(data, radius=1):
    tent_mask = data==Label.TENTORIUM
    dil_tent_mask = dilate(tent_mask, radius=radius, struct_sequence="B")
    data[dil_tent_mask & ~tent_mask & (data > 0) & (data!=Label.FALX)] = Label.CSF
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_csf_around_falx(data, radius=1):
    falx_mask = data==Label.FALX
    dil_falx_mask = dilate(falx_mask, radius=radius, struct_sequence="B")
    data[dil_falx_mask & ~falx_mask & (data > 0) & (data!=Label.TENTORIUM)] = Label.CSF
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def build_inferior_lateral_ventricle_horns(data, radius=2):
    LV_INF = [Label.LEFT_INFERIOR_LATERAL_VENTRICLE,
              Label.RIGHT_INFERIOR_LATERAL_VENTRICLE]
    LV = [Label.LEFT_LATERAL_VENTRICLE, Label.RIGHT_LATERAL_VENTRICLE]
    CP = [Label.LEFT_CHOROID_PLEXUS, Label.RIGHT_CHOROID_PLEXUS]
    for LVINFID, LVID, CPID in zip(LV_INF, LV, CP):
        mask = data == LVINFID
        mask = dilate(nbmorph.close_labels_spherical(mask, radius=15), radius=1)
        mask = nbmorph.smooth_labels_spherical(mask, 2)
        #data[data==LVINFID] = 0
        mask[data==CPID] = 0
        data[mask] = LVINFID
        #data = dilate(data, radius=1)
    return data

def get_closest_point(a, b):
    dist = edt(a==False) # takes about 0.5 s on 0.5mm resolution
    dist[b==False] = np.inf
    minidx = np.unravel_index(np.argmin(dist), a.shape)
    return minidx

def connect_by_line(m1, m2, radius=2):
    # compute connection m1 and m2
    pointa = get_closest_point(m1, m2)
    pointb = get_closest_point(m2, m1)

    # add a line between the shortest points
    line = np.array(skimage.draw.line_nd(pointa, pointb, endpoint=True))
    conn = np.zeros_like(m1)
    i,j,k = line
    conn[i,j,k] = 1
    return dilate(conn, radius=radius, struct_sequence="B")

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_connected_ventricles(data, min_thickness=2):
    V4_mask = nbmorph.smooth_labels_spherical(data==Label.FOURTH_VENTRICLE, radius=2)
    V3_mask = nbmorph.smooth_labels_spherical(data==Label.THIRD_VENTRICLE, radius=2)
    aq_conn = connect_by_line(V3_mask, V4_mask, radius=min_thickness)
    
    data[aq_conn] = Label.FOURTH_VENTRICLE

    RLV_mask = nbmorph.smooth_labels_spherical(data==Label.RIGHT_LATERAL_VENTRICLE, radius=2)
    LLV_mask = nbmorph.smooth_labels_spherical(data==Label.LEFT_LATERAL_VENTRICLE, radius=2)

    # enforce foramen of monroe connection
    fm_conn = connect_by_line(V3_mask, RLV_mask, radius=min_thickness)
    fm_conn += connect_by_line(V3_mask, LLV_mask, radius=min_thickness)
    data[fm_conn] = Label.THIRD_VENTRICLE
    return data

def get_lowest_point(mask):
    idx = np.argwhere(mask)
    ind = np.argsort(idx[:, 2])
    return idx[ind[0]]

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def enforce_tight_ventricles(data, thickness=2):
    ventricle_mask = np.isin(data, VENTRICLE_LABELS)
    v_layer = dilate(ventricle_mask, radius=thickness) ^ ventricle_mask
    v_lowest = get_lowest_point(v_layer)
    v_layer[:, :, :v_lowest[2] + 20] = 0
    tissue = data.copy()
    tissue[np.isin(tissue, VENTRICLE_LABELS + [Label.CSF])] = 0
    tissue = dilate(tissue, radius=10)
    data = np.where(v_layer, tissue, data)
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def extend_brainstem_caudally(data, offset=12):
    """
    Extends the brainstem downwards through the CSF to the bottom of the image.
    """
    
    lowest_brain_stem = get_lowest_point(data==Label.BRAIN_STEM)[2]
    lowest_csf = get_lowest_point(data==Label.CSF)[2] 
    
    extend_down = 0
    # Extract the 2D footprint of the brainstem at its lowest point
    footprint = data[:, :, lowest_brain_stem+offset:lowest_brain_stem+offset+1] == Label.BRAIN_STEM
    footprint = nbmorph.close_labels_spherical(footprint, radius=4)
    footprint_csf = nbmorph.dilate_labels_spherical(footprint, radius=4)

    #from IPython import embed; embed()  
    z_min = max(0, min(lowest_brain_stem, lowest_csf) - extend_down)
    z_max = max(lowest_brain_stem, lowest_csf) + offset
    
    target_block = data[:, :, z_min:z_max]

    for fp, l in [(footprint_csf, Label.CSF), (footprint, Label.BRAIN_STEM)]:
        mask_to_replace = fp & np.logical_or(target_block == Label.CSF, target_block==0)
        target_block[mask_to_replace] = l
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def extend_brainstem(data):
    dil_stem_mask = data == Label.BRAIN_STEM
    lowest_csf_z = get_lowest_point(data==Label.CSF)[2]
    lowest_bs_z = get_lowest_point(data==Label.BRAIN_STEM)[2]

    while lowest_bs_z > lowest_csf_z + 2:
        dil_stem_mask = dilate(dil_stem_mask, 2)
        dil_stem_mask[data > 0] = 0
        lowest_bs_z = get_lowest_point(dil_stem_mask)[2]

    dil_stem_mask[:,:, :lowest_csf_z] = 0
    data[dil_stem_mask] = Label.BRAIN_STEM
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def cut_bottom(data, offset=10):
    lowest_z = get_lowest_point(data>0)[2]
    data[:,:, :lowest_z + offset] = 0
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def close_csf_space(data, radius=1,iter=1, brainstem_area_radius=0):
    closed_mask = nbmorph.close_labels_spherical(data > 0, radius=radius, iterations=iter)
    if brainstem_area_radius:
        brainstem_mask = dilate(data==Label.BRAIN_STEM, radius=brainstem_area_radius)
    else: 
        brainstem_mask = True
    data[closed_mask & (data==0) & brainstem_mask] = Label.CSF
    return data

def smooth_1d_lines_taubin(line_mesh, iterations=50, lambda_val=0.5, mu_val=-0.5):
    """
    Smooths a 1D line mesh using Taubin (volume-preserving) smoothing.
    """
    lines = line_mesh.lines
    edges = []
    i = 0
    while i < len(lines):
        n_pts = lines[i]
        for j in range(n_pts - 1):
            edges.append([lines[i + 1 + j], lines[i + 2 + j]])
        i += n_pts + 1
    edges = np.array(edges)
    
    # 2. Count the neighbors for each point
    counts = np.zeros((len(line_mesh.points), 1))
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(counts, edges[:, 1], 1)
    counts[counts == 0] = 1 
    
    pts = line_mesh.points.copy()
    
    for _ in range(iterations):

        avg_pts = np.zeros_like(pts)
        np.add.at(avg_pts, edges[:, 0], pts[edges[:, 1]])
        np.add.at(avg_pts, edges[:, 1], pts[edges[:, 0]])
        
        # Calculate the direction and distance to the average neighbor
        laplacian = (avg_pts / counts) - pts
            
        pts += lambda_val * laplacian # Move partially inward
        
        # expand for volume preservations
        avg_pts = np.zeros_like(pts)
        np.add.at(avg_pts, edges[:, 0], pts[edges[:, 1]])
        np.add.at(avg_pts, edges[:, 1], pts[edges[:, 0]])
        
        laplacian = (avg_pts / counts) - pts

        pts += mu_val * laplacian # Push back outward slightly
            
    line_mesh.points = pts
    return line_mesh


@time_func
def straighten_spinal_interface(surf, orig_grid):
    grid = orig_grid.copy()
    grid.direction_matrix = np.round(grid.direction_matrix, 0)
    surf_rough = grid.contour_labels("all", smoothing=False)
    bottom_points = np.isclose(surf_rough.points[:,2], 
                               surf_rough.points[:,2].min())
    surf.points[bottom_points, 2] = surf.points[:,2].min()
    bottom_patch = surf.extract_cells(
        np.isclose(surf.cell_centers().points[:,2], 
        surf.points[:,2].min())
    ).extract_surface(algorithm=None).extract_feature_edges()

    tree = cKDTree(surf.points)
    _, original_surf_indices = tree.query(bottom_patch.points)

    bottom_patch = smooth_1d_lines_taubin(bottom_patch)
    surf.points[original_surf_indices] = bottom_patch.points
    return surf


@time_func
def coarsen_surface(surf, decimation_ratio=0.5):
    surf_dec = surf.decimate(decimation_ratio)
    return transfer_labels(surf, surf_dec, "boundary_labels")

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def fill_holes_csf(data):
    holes = ndi.binary_fill_holes(data > 0) != (data > 0)
    data[holes] = Label.CSF
    return data

@plot_voxel_changes(num_samples=4, window_radius=12)
@track_voxel_changes
@time_func
def fill_wm_hyperintensities(data):
    wm_data = data.copy()
    wm_data[~np.isin(data, [Label.LEFT_CEREBRAL_WHITE_MATTER, Label.RIGHT_CEREBRAL_WHITE_MATTER])] = 0
    data[data==Label.WM_HYPOINTENSITIES] = dilate(wm_data, radius=6)[data==Label.WM_HYPOINTENSITIES]
    return data

