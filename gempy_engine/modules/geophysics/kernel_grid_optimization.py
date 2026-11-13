"""Kernel (centered) grid spacing that spreads gravity sensitivity evenly over the cells.

The kernel grid around each device is a tensor grid whose cells grow away from the device. With the default
geometric spacing the cells next to the device still carry a much larger share of the gravity signal than the rest,
so a contact that happens to cross them weighs too much (aliasing). Following Liang, de la Varga and Wellmann (2023,
Geophysics 88(3), Algorithm 1), the cell spacing of each axis is parameterised as
``unit_radius = cumsum(softmax(u))`` - positive steps that always end at the kernel radius - and ``u`` is chosen to
minimise the standard deviation of the cells' sensitivity ``tz``.

The optimum depends only on the resolution and the ratios of the radii, so it is computed once and cached.
"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize


def optimized_unit_radii(grid_resolution, scaling_factor, base_spacing=0.01, z_axis_shift=0.05, z_axis_scale=1.2,
                         max_iterations=200) -> list[np.ndarray]:
    """Per-axis cell-center distances (in units of the radius) that minimise the spread of tz.

    Arguments match ``CenteredGrid.update_kernels``. Returns a list of three arrays to pass as
    ``CenteredGrid.create_irregular_grid_kernel(..., unit_radii=...)``.
    """
    resolution = tuple(int(r) for r in np.atleast_1d(grid_resolution))
    radius = np.atleast_1d(np.asarray(scaling_factor, dtype="float64"))
    if radius.size == 1:
        radius = np.repeat(radius, 3)
    ratios = tuple(np.round(radius / radius.max(), 12))  # tz scales with length: std(tz) only rescales
    cached = _optimize(resolution, ratios, float(base_spacing), float(z_axis_shift), float(z_axis_scale),
                       int(max_iterations))
    return [axis.copy() for axis in cached]


def kernel_tz_spread(grid_resolution, scaling_factor, unit_radii=None, base_spacing=0.01, z_axis_shift=0.05,
                     z_axis_scale=1.2) -> float:
    """Standard deviation of tz over the kernel cells, the quantity the optimisation minimises."""
    from ...core.data.centered_grid import CenteredGrid
    from .gravity_gradient import calculate_gravity_gradient

    centers, left, right = CenteredGrid.create_irregular_grid_kernel(
        grid_resolution, np.asarray(scaling_factor, dtype="float64"), base_spacing=base_spacing,
        z_axis_shift=z_axis_shift, z_axis_scale=z_axis_scale, unit_radii=unit_radii)
    kernel = SimpleNamespace(kernel_grid_centers=centers, left_voxel_edges=left, right_voxel_edges=right)
    return float(np.std(calculate_gravity_gradient(kernel)))


def _points_per_axis(resolution) -> list[int]:
    return [resolution[axis] // (1 if axis == 2 else 2) for axis in range(3)]


def _unit_radius(u: np.ndarray) -> np.ndarray:
    steps = np.exp(u - u.max())
    return np.cumsum(steps / steps.sum())


@lru_cache(maxsize=32)
def _optimize(resolution, ratios, base_spacing, z_axis_shift, z_axis_scale, max_iterations):
    counts = _points_per_axis(resolution)
    splits = np.cumsum(counts)[:-1]

    def unit_radii(theta):
        return [_unit_radius(u) if u.size else u for u in np.split(theta, splits)]

    # Start from the default geometric spacing, so the result is never worse than it.
    start = []
    for n in counts:
        geometric = np.geomspace(base_spacing, 1, n) if n else np.zeros(0)
        start.append(np.log(np.diff(np.concatenate(([0.0], geometric)))) if n else geometric)
    theta0 = np.concatenate(start)

    def raw_spread(theta):
        return kernel_tz_spread(resolution, np.asarray(ratios), unit_radii(theta), base_spacing, z_axis_shift,
                                z_axis_scale)

    start_spread = raw_spread(theta0)

    def spread(theta):  # relative to the start, so the gradient is O(1) whatever the units of tz
        return raw_spread(theta) / start_spread

    result = minimize(spread, theta0, method="L-BFGS-B", options={"maxiter": max_iterations})
    best = result.x if result.fun <= 1.0 else theta0
    return tuple(unit_radii(best))
