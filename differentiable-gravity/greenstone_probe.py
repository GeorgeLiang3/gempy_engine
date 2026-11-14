"""Probe: differentiable gravity on gempy's Greenstone model (3 series + basement, real geometry).

    PYTHONPATH=<engine checkout of feat/differentiable-gravity> python greenstone_probe.py
"""

import os
import sys
import time
from pathlib import Path

import gempy as gp
import gempy_engine
import numpy as np
import torch

DATA = Path(os.environ.get("GEMPY_REPO", Path.home() / "Documents/github/gempy_dev/gempy")) / \
    "examples/data/input_data/tut_SandStone"
DENSITIES = np.array([2.61, 2.92, 3.1, 2.92, 2.61])  # EarlyGranite, SimpleMafic2, SimpleBIF, SimpleMafic1, basement
EXTENT = [696000, 747000, 6863000, 6930000, -20000, 200]


def receivers(n=10):
    xs = np.linspace(EXTENT[0] + 5000, EXTENT[1] - 5000, n)
    ys = np.linspace(EXTENT[2] + 5000, EXTENT[3] - 5000, n)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, 200.0)]


def build(blend, slope, kernel_res=(10, 10, 40), kernel_radius=(5000, 5000, 6000), shifts=None):
    m = gp.create_geomodel(
        project_name="Greenstone", extent=EXTENT, refinement=1,
        importer_helper=gp.data.ImporterHelper(path_to_orientations=str(DATA / "SandStone_Foliations.csv"),
                                               path_to_surface_points=str(DATA / "SandStone_Points.csv")))
    gp.map_stack_to_surfaces(m, {"EarlyGranite_Series": "EarlyGranite", "BIF_Series": ("SimpleMafic2", "SimpleBIF"),
                                 "SimpleMafic_Series": "SimpleMafic1", "Basement": "basement"})
    gp.set_centered_grid(m.grid, centers=receivers(), resolution=np.array(kernel_res), radius=np.array(kernel_radius))
    tz = gp.calculate_gravity_gradient(m.grid.centered_grid)
    m.geophysics_input = gp.data.GeophysicsInput(tz=tz, densities=DENSITIES, blend_properties=blend)
    m.interpolation_options.mesh_extraction = False
    m.interpolation_options.cache_mode = gp.data.InterpolationOptions.CacheMode.NO_CACHE
    if slope is not None:
        m.interpolation_options.sigmoid_slope = slope
        m.interpolation_options.erosion_sigmoid_slope = slope
    for rows, dz in (shifts or []):
        z0 = m.surface_points_copy.df["Z"].values.copy()
        gp.modify_surface_points(m, slice=rows, Z=z0[rows] + dz)
    return m


def element_names(m):
    return {e.name: i for i, e in enumerate(m.structural_frame.structural_elements)}


def point_formation(m):
    ids = m.surface_points_copy.df["id"].values
    id_to_name = {e.id: e.name for e in m.structural_frame.structural_elements}
    return np.array([id_to_name[i] for i in ids])


def main():
    m = build(False, None)
    forms = point_formation(m)
    print("surface points per formation:", {f: int((forms == f).sum()) for f in np.unique(forms)})
    print("transform scale:", m.input_transform.scale)
    picks = [int(np.where(forms == f)[0][len(np.where(forms == f)[0]) // 2]) for f in np.unique(forms)]
    print("probe points:", picks, forms[picks])

    t0 = time.time()
    g_hard = np.asarray(gp.compute_model(m, engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy)).gravity)
    print(f"numpy forward {time.time() - t0:.1f}s, gravity range {g_hard.min():.1f} .. {g_hard.max():.1f} mGal")

    h = 150.0
    fd = {}
    for p in picks:
        gp_ = [np.asarray(gp.compute_model(build(False, None, shifts=[([p], s)]), engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.numpy)).gravity) for s in (h, -h)]
        fd[p] = (gp_[0] - gp_[1]) / (2 * h)

    for slope in [float(s) for s in sys.argv[1:]] or [300.0, 1000.0]:
        m = build(True, slope)
        t0 = time.time()
        sol = gp.compute_model(m, engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH, dtype="float64", compute_grads=True))
        g = sol.gravity
        t_f = time.time() - t0
        sp = m.taped_interpolation_input.surface_points.sp_coords
        t0 = time.time()
        jac = np.zeros((len(g), len(picks)))
        v = torch.ones_like(g)  # cheap check: one backward of sum(g) for a gradient per point
        (grad,) = torch.autograd.grad((g * v).sum(), sp, retain_graph=True)
        t_b = time.time() - t0
        for j, p in enumerate(picks):
            ad_sum = grad[p, 2].item() * m.input_transform.scale[2]
            print(f"slope {slope:g}: point {p} ({forms[p]}): d(sum g)/dz AD={ad_sum:.4g} FD={fd[p].sum():.4g} "
                  f"ratio={ad_sum / fd[p].sum():.3f}")
        g_np = g.detach().numpy()
        print(f"slope {slope:g}: forward {t_f:.1f}s backward {t_b:.1f}s, max |g_soft - g_hard| = "
              f"{np.abs(g_np - g_hard).max():.3f} mGal ({np.abs(g_np - g_hard).max() / np.abs(g_hard).max():.2%})")


if __name__ == "__main__":
    main()
