"""Optimized kernel (centered) grid spacing: CenteredGrid(kernel_spacing="optimized")."""
import numpy as np
import pytest

from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.modules.geophysics.gravity_gradient import calculate_gravity_gradient
from gempy_engine.modules.geophysics.kernel_grid_optimization import optimized_unit_radii

RESOLUTION = [10, 10, 15]
RADIUS = [1000.0, 1000.0, 1000.0]


def _grid(spacing):
    return CenteredGrid(centers=[[500.0, 500.0, 0.0]], resolution=RESOLUTION, radius=RADIUS, kernel_spacing=spacing)


def test_default_spacing_is_unchanged():
    geometric = [np.geomspace(0.01, 1, n) for n in (5, 5, 15)]
    explicit = CenteredGrid.create_irregular_grid_kernel(RESOLUTION, np.array(RADIUS), unit_radii=geometric)
    default = _grid("exponential")
    np.testing.assert_array_equal(default.kernel_grid_centers, explicit[0])
    np.testing.assert_array_equal(default.left_voxel_edges, explicit[1])
    np.testing.assert_array_equal(default.right_voxel_edges, explicit[2])


def test_optimized_spacing_spreads_tz_more_evenly():
    tz_exponential = calculate_gravity_gradient(_grid("exponential"))
    tz_optimized = calculate_gravity_gradient(_grid("optimized"))
    assert tz_optimized.shape == tz_exponential.shape
    assert np.std(tz_optimized) < 0.5 * np.std(tz_exponential)
    assert np.abs(tz_optimized).max() < 0.5 * np.abs(tz_exponential).max()


def test_optimized_unit_radii_are_increasing_and_end_at_the_radius():
    radii = optimized_unit_radii(RESOLUTION, RADIUS)
    assert [r.shape for r in radii] == [(5,), (5,), (15,)]
    for r in radii:
        assert np.all(np.diff(r) > 0) and r[0] > 0
        assert r[-1] == pytest.approx(1.0)


def test_optimized_spacing_is_cached_and_depends_only_on_radius_ratios():
    a = optimized_unit_radii(RESOLUTION, RADIUS)
    b = optimized_unit_radii(RESOLUTION, [2 * r for r in RADIUS])
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_unknown_spacing_raises():
    with pytest.raises(ValueError, match="kernel_spacing"):
        _grid("regular")
