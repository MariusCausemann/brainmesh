import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from .synthseg_labels import Label
from datetime import datetime
from contextlib import contextmanager
import os
import time

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{name}] finished in {end - start:.4f} seconds")

# A simple helper to wrap the timer around any function
def time_func(func):
    def wrapper(*args, **kwargs):
        with timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper

def track_voxel_changes(func):
    """
    Decorator that tracks and prints the exact voxel label changes 
    made by a function. Assumes the first argument is a numpy array 
    (the image data) and that the function returns the modified numpy array.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Ensure there is a first argument and it's a numpy array
        if not args or not isinstance(args[0], np.ndarray):
            return func(*args, **kwargs)
            
        # 1. Store a copy of the input array
        original_data = args[0].copy()
        
        # 2. Execute the original function
        result = func(*args, **kwargs)
        
        # Ensure the output is also a numpy array to allow comparison
        if not isinstance(result, np.ndarray):
            return result
            
        # 3. Compare the arrays
        changed_mask = original_data != result
        total_changed = np.sum(changed_mask)
        
        if total_changed == 0:
            print("  No voxels were altered.\n")
            return result
        
        reverse_map = {
            getattr(Label, attr): attr 
            for attr in dir(Label) 
            if not attr.startswith('_') and isinstance(getattr(Label, attr), (int, np.integer))
        }
        reverse_map[0] = "BACKGROUND (0)"
        print(f"  Total voxels changed: {total_changed} ({total_changed / original_data.size *100:.3f}%)")
        
        # Extract the old and new values using the boolean mask
        old_vals = original_data[changed_mask]
        new_vals = result[changed_mask]
        
        # Stack into pairs: [[old1, new1], [old2, new2], ...]
        change_pairs = np.column_stack((old_vals, new_vals))
        
        # Find unique transition pairs and count them
        unique_pairs, counts = np.unique(change_pairs, axis=0, return_counts=True)
        
        # Sort by count (descending) so the biggest changes print first
        sort_idx = np.argsort(-counts)
        
        for (old_val, new_val), count in zip(unique_pairs[sort_idx], counts[sort_idx]):
            old_name = reverse_map.get(old_val, f"UNKNOWN_LABEL_{old_val}")
            new_name = reverse_map.get(new_val, f"UNKNOWN_LABEL_{new_val}")
            
            print(f"  - {count:<8} voxels: {old_name} -> {new_name}")
        print() # Blank line for readability
        
        return result
    return wrapper

import numpy as np
import matplotlib.pyplot as plt
import os
import fastremap
from functools import wraps

# Assuming Label is already imported at the top of utils.py
# from .synthseg_labels import Label

def get_real_function_name(f):
    """Recursively unpacks closures to find the true function name."""
    name = getattr(f, '__name__', 'unknown')
    if name == 'wrapper' and getattr(f, '__closure__', None):
        for cell in f.__closure__:
            if callable(cell.cell_contents):
                return get_real_function_name(cell.cell_contents)
    return name



def plot_voxel_changes(num_samples=3, window_radius=15, output_dir=None):
    """
    Decorator that plots a 2x3 before-and-after view of changed voxels.
    Uses fastremap and a discrete colormap for maximum categorical contrast.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        output_dir = f"./change_plots_{timestamp}"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args or not isinstance(args[0], np.ndarray):
                return func(*args, **kwargs)
                
            original_data = args[0].copy()
            result = func(*args, **kwargs)
            return result
            if not isinstance(result, np.ndarray):
                return result
                
            changed_mask = original_data != result
            changed_indices = np.argwhere(changed_mask)
            
            if len(changed_indices) == 0:
                return result
                
            os.makedirs(output_dir, exist_ok=True)
                
            n_plots = min(num_samples, len(changed_indices))
            chosen_idx = np.random.choice(len(changed_indices), n_plots, replace=False)
            chosen_coords = changed_indices[chosen_idx]
            
            real_func_name = get_real_function_name(func)
            
            reverse_map = {
                getattr(Label, attr): attr for attr in dir(Label) 
                if not attr.startswith('_') and isinstance(getattr(Label, attr), (int, np.integer))
            }
            reverse_map[0] = "BACKGROUND (0)"
            
            for i, (x, y, z) in enumerate(chosen_coords):
                
                x_min, x_max = max(0, x - window_radius), min(original_data.shape[0], x + window_radius + 1)
                y_min, y_max = max(0, y - window_radius), min(original_data.shape[1], y + window_radius + 1)
                z_min, z_max = max(0, z - window_radius), min(original_data.shape[2], z + window_radius + 1)
                
                old_val = int(original_data[x, y, z])
                new_val = int(result[x, y, z])
                
                # Extract original patches
                patches = {
                    'sag_b': original_data[x, y_min:y_max, z_min:z_max],
                    'sag_a': result[x, y_min:y_max, z_min:z_max],
                    'cor_b': original_data[x_min:x_max, y, z_min:z_max],
                    'cor_a': result[x_min:x_max, y, z_min:z_max],
                    'ax_b': original_data[x_min:x_max, y_min:y_max, z],
                    'ax_a': result[x_min:x_max, y_min:y_max, z]
                }
                
                # Find unique labels across all patches using fastremap
                all_visible = np.concatenate([p.flatten() for p in patches.values()])
                unique_vals = set(fastremap.unique(all_visible))
                unique_vals.add(0)
                n_colors = len(unique_vals)
                
                # --- DISCRETE COLORMAP FIX ---
                # Force matplotlib to chunk the continuous map into exactly 'n_colors' blocks
                cmap = plt.get_cmap('nipy_spectral', n_colors)
                
                # Shift the min and max by 0.5 so integers fall dead-center in the color blocks
                global_vmin = -0.5
                global_vmax = n_colors - 0.5
                
                # Create a mapping from sparse IDs to dense sequence (0, 1, 2...)
                mapping = {int(val): idx for idx, val in enumerate(unique_vals)}
                
                # Remap all patches in-place to the new dense integers
                for key in patches:
                    patches[key] = fastremap.remap(patches[key], mapping)
                
                # Helper to draw a 1x1 voxel bounding box
                def add_red_box(ax, hz, vt):
                    xs = [hz - 0.5, hz + 0.5, hz + 0.5, hz - 0.5, hz - 0.5]
                    ys = [vt - 0.5, vt - 0.5, vt + 0.5, vt + 0.5, vt - 0.5]
                    ax.plot(xs, ys, color='red', linewidth=2)
                
                # Setup 2x3 Grid
                fig, axes = plt.subplots(2, 3, figsize=(14, 8))
                fig.suptitle(f"[{real_func_name}] Change: {reverse_map[old_val]} $\\rightarrow$ {reverse_map[new_val]}\nCoord: ({x}, {y}, {z})", fontsize=16)
                
                # --- COLUMN 0: SAGITTAL ---
                axes[0, 0].imshow(patches['sag_b'], cmap=cmap, vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
                axes[0, 0].set_title("Sagittal Before")
                im = axes[1, 0].imshow(patches['sag_a'], cmap=cmap, vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
                axes[1, 0].set_title("Sagittal After")
                
                add_red_box(axes[0, 0], z - z_min, y - y_min)
                add_red_box(axes[1, 0], z - z_min, y - y_min)

                # --- COLUMN 1: CORONAL ---
                axes[0, 1].imshow(patches['cor_b'], cmap=cmap, vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
                axes[0, 1].set_title("Coronal Before")
                axes[1, 1].imshow(patches['cor_a'], cmap=cmap, vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
                axes[1, 1].set_title("Coronal After")
                
                add_red_box(axes[0, 1], z - z_min, x - x_min)
                add_red_box(axes[1, 1], z - z_min, x - x_min)

                # --- COLUMN 2: AXIAL ---
                axes[0, 2].imshow(patches['ax_b'], cmap=cmap, vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
                axes[0, 2].set_title("Axial Before")
                axes[1, 2].imshow(patches['ax_a'], cmap=cmap, vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
                axes[1, 2].set_title("Axial After")
                
                add_red_box(axes[0, 2], y - y_min, x - x_min)
                add_red_box(axes[1, 2], y - y_min, x - x_min)
                # Clean up axes ticks
                for ax in axes.flat:
                    ax.axis('off')
                
                # --- COLORBAR CREATION ---
                # Place vertical colorbar on the right side. 'drawedges=True' adds physical lines between colors!
                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.03, pad=0.04, drawedges=True)
                
                # Set ticks at the exact integers (which are the centers of the color blocks now)
                cbar.set_ticks(range(n_colors))
                
                # Use the ORIGINAL integer values to fetch the names for the tick labels
                tick_labels = [reverse_map.get(int(v), f"UNKNOWN ({int(v)})") for v in unique_vals]
                cbar.set_ticklabels(tick_labels, fontsize=10)
                
                # Save out
                filename = f"{real_func_name}_change_{old_val}_to_{new_val}_{x}_{y}_{z}.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig) 
                
            print(f"[{real_func_name}] Saved {n_plots} debug plot(s) to {output_dir}/")
            return result
        return wrapper
    return decorator