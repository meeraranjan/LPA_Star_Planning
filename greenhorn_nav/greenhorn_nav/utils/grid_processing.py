#utils/grid_processing.py
import numpy as np
from scipy.ndimage import binary_dilation

def inflate_obstacles(occ_grid, inflation_radius, resolution):
    """
    Inflate occupied cells in occ_grid by inflation_radius (in meters).
    occ_grid: 2D numpy array (0=free, 1=occupied)
    resolution: meters per cell
    """
    inflate_cells = int(np.ceil(inflation_radius / resolution))
    if inflate_cells <= 0:
        return occ_grid.copy()

    structure = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1))
    inflated = binary_dilation(occ_grid > 0.5, structure=structure).astype(np.uint8)
    return inflated

def detect_changed_cells(old_grid, new_grid):
    """
    Returns list of (i,j) cells that changed.
    """
    if old_grid is None:
        return []
    diff = (old_grid != new_grid)
    changed = np.argwhere(diff)
    return [tuple(idx) for idx in changed]
