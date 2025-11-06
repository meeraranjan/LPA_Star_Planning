#utils/path_utils.py
import numpy as np
from scipy.interpolate import splprep, splev

def smooth_path(waypoints, smooth_factor=0.0, num_points=100):
    """Smooth a 2D path using B-spline interpolation.

    This function takes a list of waypoints (x, y) and generates a smooth
    path by fitting a B-spline through the points. The resulting path
    can have a different number of points than the original.

    Args:
        waypoints (list of tuple[float, float]): List of (x, y) waypoints.
        smooth_factor (float, optional): Smoothing factor for the spline.
            Higher values create smoother paths. Defaults to 0.0 (interpolation).
        num_points (int, optional): Number of points in the output smoothed path.
            Defaults to 100.

    Returns:
        np.ndarray: Smoothed path of shape (num_points, 2), with columns [x, y].
                    If the input has fewer than 3 waypoints, returns the input as-is.
    """
    if len(waypoints) < 3:
        return waypoints
    x, y = zip(*waypoints)
    tck, _ = splprep([x, y], s=smooth_factor)
    u = np.linspace(0, 1, num_points)
    x_s, y_s = splev(u, tck)
    return np.vstack((x_s, y_s)).T
