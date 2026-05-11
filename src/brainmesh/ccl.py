import numpy as np
from numba import njit

# ---------------------------------------------------------
# Union-Find Helper Functions
# ---------------------------------------------------------
@njit(cache=True)
def find(parent, i):
    """Finds the root label of a given component with path compression."""
    root = i
    while root != parent[root]:
        root = parent[root]
    
    # Path compression: make all nodes on the path point directly to root
    curr = i
    while curr != root:
        nxt = parent[curr]
        parent[curr] = root
        curr = nxt
    return root

@njit(cache=True)
def union(parent, i, j):
    """Merges two labels into the same component."""
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i != root_j:
        # Attach the larger root label to the smaller one
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

# ---------------------------------------------------------
# Main CCL Algorithm
# ---------------------------------------------------------
@njit(cache=True)
def ccl_3d_26conn(image):
    Z, Y, X = image.shape
    out = np.zeros((Z, Y, X), dtype=np.int32)
    
    # Pre-allocate parent array. 
    # Max possible independent components in a 3D grid is roughly (Z*Y*X)//8
    max_labels = (Z * Y * X) // 8 + 2
    parent = np.arange(max_labels, dtype=np.int32)
    next_label = 1

    # The 13 backward-looking neighbors for 26-connectivity
    offsets = np.array([
        [-1, -1, -1], [-1, -1,  0], [-1, -1,  1],
        [-1,  0, -1], [-1,  0,  0], [-1,  0,  1],
        [-1,  1, -1], [-1,  1,  0], [-1,  1,  1],
        [ 0, -1, -1], [ 0, -1,  0], [ 0, -1,  1],
        [ 0,  0, -1]
    ], dtype=np.int32)

    # --- PASS 1: Assign initial labels and record equivalences ---
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if not image[z, y, x]:
                    continue

                min_label = next_label
                has_neighbor = False

                # Check all 13 previously visited neighbors
                for i in range(13):
                    dz, dy, dx = offsets[i]
                    nz, ny, nx = z + dz, y + dy, x + dx
                    
                    if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                        lbl = out[nz, ny, nx]
                        if lbl > 0:
                            has_neighbor = True
                            root = find(parent, lbl)
                            if root < min_label:
                                min_label = root
                
                if not has_neighbor:
                    # New isolated component found
                    out[z, y, x] = next_label
                    next_label += 1
                else:
                    # Assign the minimum root label found
                    out[z, y, x] = min_label
                    
                    # Record equivalences (union) for all surrounding foreground pixels
                    for i in range(13):
                        dz, dy, dx = offsets[i]
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                            lbl = out[nz, ny, nx]
                            if lbl > 0:
                                union(parent, min_label, lbl)

    # --- PASS 2: Resolve labels and make them contiguous ---
    new_labels = np.zeros(next_label, dtype=np.int32)
    current_new_label = 1

    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if out[z, y, x] > 0:
                    # Find the ultimate root of the current label
                    root = find(parent, out[z, y, x])
                    
                    # Map the root to a dense, contiguous label set
                    if new_labels[root] == 0:
                        new_labels[root] = current_new_label
                        current_new_label += 1
                        
                    out[z, y, x] = new_labels[root]

    return out, current_new_label - 1

@njit(cache=True)
def ccl_3d_6conn(image):
    Z, Y, X = image.shape
    out = np.zeros((Z, Y, X), dtype=np.int32)
    
    max_labels = (Z * Y * X) // 8 + 2
    parent = np.arange(max_labels, dtype=np.int32)
    next_label = 1

    # Only 3 backward-looking neighbors for 6-connectivity (Z-face, Y-face, X-face)
    offsets = np.array([
        [-1,  0,  0], 
        [ 0, -1,  0], 
        [ 0,  0, -1]
    ], dtype=np.int32)

    # --- PASS 1: Assign initial labels and record equivalences ---
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if not image[z, y, x]:
                    continue

                min_label = next_label
                has_neighbor = False

                # Check the 3 previously visited neighbors
                for i in range(3):
                    dz, dy, dx = offsets[i]
                    nz, ny, nx = z + dz, y + dy, x + dx
                    
                    if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                        lbl = out[nz, ny, nx]
                        if lbl > 0:
                            has_neighbor = True
                            root = find(parent, lbl)
                            if root < min_label:
                                min_label = root
                
                if not has_neighbor:
                    out[z, y, x] = next_label
                    next_label += 1
                else:
                    out[z, y, x] = min_label
                    for i in range(3):
                        dz, dy, dx = offsets[i]
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                            lbl = out[nz, ny, nx]
                            if lbl > 0:
                                union(parent, min_label, lbl)

    # --- PASS 2: Resolve labels and count N ---
    new_labels = np.zeros(next_label, dtype=np.int32)
    current_new_label = 1

    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if out[z, y, x] > 0:
                    root = find(parent, out[z, y, x])
                    
                    if new_labels[root] == 0:
                        new_labels[root] = current_new_label
                        current_new_label += 1
                        
                    out[z, y, x] = new_labels[root]

    # Calculate total number of components
    N = current_new_label - 1

    return out, N

@njit(cache=True)
def remove_small_objects(image, max_size, connectivity=6):
    # 1. Run the fast CCL
    if connectivity==26:
        labels, N = ccl_3d_26conn(image)
    elif connectivity==6:
        labels, N = ccl_3d_6conn(image)
    else:
        raise ValueError("connectivity has to be 6 or 26!")
    # 2. Count the pixels in each label
    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels)
    
    # 3. Create the filtered boolean array
    Z, Y, X = image.shape
    out = np.zeros((Z, Y, X), dtype=np.bool_)
    
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                lbl = labels[z, y, x]
                # If it's foreground AND the component size >= min_size, keep it
                if lbl > 0 and counts[lbl] >= max_size:
                    out[z, y, x] = True
                    
    return out

@njit
def label(img, connectivity=6):
    if connectivity==26:
        return ccl_3d_26conn(img)
    elif connectivity==6:
        return ccl_3d_6conn(img)
    else:
        raise ValueError("connectivity has to be 6 or 26!")