"""Differentiable gravity on gempy's Greenstone model (fork PRs 1 + 2): data for the demo report.

Greenstone belt (Western Australia), gempy example data: EarlyGranite / BIF (SimpleMafic2, SimpleBIF) / SimpleMafic1
series over basement, 70 surface points, 41 orientations, 51 x 67 x 20 km. Synthetic gravity at 100 receivers.

Writes greenstone_demo.json with model sections, gravity maps, the Jacobian of gravity with respect to the depth of
every surface point (current engine vs PRs 1 + 2, finite-difference checks), an inversion of the four horizon depths,
and the Hessian of the misfit with its Laplace posterior.

Run on a VM, from a directory that has no gempy/ folder in it:
    GEMPY_REPO=<gempy checkout> python greenstone_demo.py <slope> out.json
"""

import json
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
EXTENT = [696000, 747000, 6863000, 6930000, -20000, 200]
DENSITIES = np.array([2.61, 2.92, 3.1, 2.92, 2.61])  # EarlyGranite, SimpleMafic2, SimpleBIF, SimpleMafic1, basement
STACKS = {"EarlyGranite_Series": "EarlyGranite", "BIF_Series": ("SimpleMafic2", "SimpleBIF"),
          "SimpleMafic_Series": "SimpleMafic1", "Basement": "basement"}
FORMATIONS = ["EarlyGranite", "SimpleMafic2", "SimpleBIF", "SimpleMafic1"]
KERNEL_RES, KERNEL_RADIUS = (10, 10, 40), (5000.0, 5000.0, 6000.0)
FD_H = 150.0
NOISE_MGAL = 0.5  # likelihood standard deviation for the misfit and the Laplace posterior
ENGINE_NUMPY = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy)
ENGINE_TORCH = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH, dtype="float64",
                                         compute_grads=True)
ENGINE_TORCH_NOGRAD = gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH, dtype="float64",
                                                compute_grads=False)  # ~4x faster than numpy for the FD runs


def receivers(n=10):
    xs = np.linspace(EXTENT[0] + 5000, EXTENT[1] - 5000, n)
    ys = np.linspace(EXTENT[2] + 5000, EXTENT[3] - 5000, n)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, 200.0)]


def new_model(blend, slope, shifts=None):
    m = gp.create_geomodel(
        project_name="Greenstone", extent=EXTENT, refinement=1,
        importer_helper=gp.data.ImporterHelper(path_to_orientations=str(DATA / "SandStone_Foliations.csv"),
                                               path_to_surface_points=str(DATA / "SandStone_Points.csv")))
    gp.map_stack_to_surfaces(m, STACKS)
    m.interpolation_options.mesh_extraction = False
    m.interpolation_options.cache_mode = gp.data.InterpolationOptions.CacheMode.NO_CACHE
    if slope is not None:
        m.interpolation_options.sigmoid_slope = slope
        m.interpolation_options.erosion_sigmoid_slope = slope
    for rows, dz in (shifts or []):
        z0 = m.surface_points_copy.df["Z"].values.copy()
        gp.modify_surface_points(m, slice=rows, Z=z0[rows] + dz)
    return m


def with_gravity(m, blend):
    gp.set_centered_grid(m.grid, centers=receivers(), resolution=np.array(KERNEL_RES), radius=np.array(KERNEL_RADIUS))
    tz = gp.calculate_gravity_gradient(m.grid.centered_grid)
    m.geophysics_input = gp.data.GeophysicsInput(tz=tz, densities=DENSITIES, blend_properties=blend)
    return m


def formations_of_points(m):
    id_to_name = {e.id: e.name for e in m.structural_frame.structural_elements}
    return np.array([id_to_name[i] for i in m.surface_points_copy.df["id"].values])


def sections(slope):
    """Unit id on a map at z = -1500 m and on an E-W section through the middle (custom grid, no gravity)."""
    xs, ys, zs = np.linspace(*EXTENT[0:2], 121), np.linspace(*EXTENT[2:4], 121), np.linspace(-8000, 200, 83)
    X, Y = np.meshgrid(xs, ys)
    plan = np.c_[X.ravel(), Y.ravel(), np.full(X.size, -1500.0)]
    Xs, Zs = np.meshgrid(xs, zs)
    ew = np.c_[Xs.ravel(), np.full(Xs.size, 0.5 * (EXTENT[2] + EXTENT[3])), Zs.ravel()]
    m = new_model(slope is not None, slope)
    gp.set_custom_grid(m.grid, xyz_coord=np.r_[plan, ew])
    m.grid.active_grids = gp.data.Grid.GridTypes.CUSTOM
    out = gp.compute_model(m, engine_config=ENGINE_NUMPY).octrees_output[0].outputs_centers[-1]
    ids = np.asarray(out.final_block[out.grid.custom_grid_slice])
    return {"x": xs.tolist(), "y": ys.tolist(), "z": zs.tolist(),
            "plan": ids[:len(plan)].reshape(len(ys), len(xs)).tolist(),
            "ew": ids[len(plan):].reshape(len(zs), len(xs)).tolist()}


class Forward:
    """gravity(theta) with theta = depth shift (m) of each formation's surface points, differentiable."""

    def __init__(self, blend, slope):
        self.m = with_gravity(new_model(blend, slope), blend)
        gp.compute_model(self.m, engine_config=ENGINE_TORCH)
        self.ii = self.m.taped_interpolation_input
        self.base = self.ii.surface_points.sp_coords.detach().clone()
        self.scale = float(self.m.input_transform.scale[2])
        forms = formations_of_points(self.m)
        self.rows = [torch.as_tensor(np.where(forms == f)[0]) for f in FORMATIONS]

    def __call__(self, theta=None, point_shift=None):
        sp = self.base.clone()
        if theta is not None:
            shift = torch.zeros_like(self.base)
            for k, rows in enumerate(self.rows):
                shift[rows, 2] = theta[k] * self.scale
            sp = sp + shift
        if point_shift is not None:
            sp = sp + point_shift
        self.ii.surface_points.sp_coords = sp
        return gempy_engine.compute_model(self.ii, self.m.interpolation_options, self.m.input_data_descriptor,
                                          geophysics_input=self.m.geophysics_input).gravity


def jacobian_all_points(forward):
    """d g_k / d z_i for every receiver k and surface point i (reverse mode, one pass per receiver)."""
    leaf = torch.zeros_like(forward.base, requires_grad=True)
    g = forward(point_shift=leaf)
    if not g.requires_grad:
        return np.zeros((len(g), len(forward.base)))
    rows = []
    for k in range(len(g)):
        (grad,) = torch.autograd.grad(g[k], leaf, retain_graph=True, allow_unused=True)
        rows.append(np.zeros(len(forward.base)) if grad is None else grad[:, 2].numpy() * forward.scale)
    return np.array(rows)


def fd_column(point, slope=None):
    """Central finite difference of the current (hard) model for one surface point."""
    g = [np.asarray(gp.compute_model(with_gravity(new_model(False, None, [([point], s)]), False),
                                     engine_config=ENGINE_TORCH_NOGRAD).gravity, dtype=float) for s in (FD_H, -FD_H)]
    return (g[0] - g[1]) / (2 * FD_H)


def misfit(forward, g_obs):
    obs = torch.as_tensor(g_obs)
    return lambda theta: 0.5 * (((forward(theta) - obs) / NOISE_MGAL) ** 2).sum()


def invert(forward, g_obs, start, steps, lr):
    loss_fn = misfit(forward, g_obs)
    theta = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    path, losses = [], []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(theta)
        if loss.requires_grad:
            loss.backward()
        path.append(theta.detach().numpy().tolist())
        losses.append(float(loss))
        opt.step()
    return path, losses


def main(slope, out):
    t_start = time.time()
    result = {}

    def log(msg):  # save after every stage, so a crash keeps what is done
        Path(out).write_text(json.dumps(result))
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    result |= {"slope": slope, "formations": FORMATIONS, "densities": DENSITIES.tolist(), "noise_mgal": NOISE_MGAL,
              "receivers": receivers().tolist(), "extent": EXTENT, "fd_h": FD_H}

    result["sections"] = {"current": sections(None), "pr1_pr2": sections(slope)}
    log("sections")

    base = new_model(False, None)
    result["surface_points"] = base.surface_points_copy.df[["X", "Y", "Z"]].values.tolist()
    result["point_formation"] = formations_of_points(base).tolist()

    forms = np.array(result["point_formation"])
    picks = [int(np.where(forms == f)[0][len(np.where(forms == f)[0]) // 2]) for f in FORMATIONS]
    result["fd_points"] = picks
    result["fd"] = {str(p): fd_column(p).tolist() for p in picks}
    log("finite differences")

    fwd = {"current": Forward(False, None), "pr1_pr2": Forward(True, slope)}
    result["gravity"] = {k: f().detach().numpy().tolist() for k, f in fwd.items()}
    log("forward gravity")

    result["jacobian"] = {}
    for k, f in fwd.items():
        result["jacobian"][k] = jacobian_all_points(f).tolist()
        log(f"jacobian {k}")

    start, truth = [400.0, -300.0, 300.0, -400.0], [0.0, 0.0, 0.0, 0.0]
    result["inversion"] = {"start": start, "truth": truth}
    for k, steps in (("current", 5), ("pr1_pr2", 50)):
        g_obs = np.array(result["gravity"][k])  # each model inverts its own synthetic data at the truth
        t0 = time.time()
        path, losses = invert(fwd[k], g_obs, start, steps, lr=60.0)
        result["inversion"][k] = {"path": path, "loss": losses, "seconds": time.time() - t0}
        log(f"inversion {k}: end {np.round(path[-1], 1)}")

    loss_fn = misfit(fwd["pr1_pr2"], np.array(result["gravity"]["pr1_pr2"]))
    t0 = time.time()
    hess = torch.autograd.functional.hessian(loss_fn, torch.tensor(truth, dtype=torch.float64)).numpy()
    t_hess = time.time() - t0

    def grad_at(theta):
        t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        (gr,) = torch.autograd.grad(loss_fn(t), t)
        return gr.numpy()

    h = 20.0
    hess_fd = np.array([(grad_at(np.eye(4)[i] * h) - grad_at(-np.eye(4)[i] * h)) / (2 * h) for i in range(4)])
    cov = np.linalg.inv(hess)  # Laplace approximation with a flat prior
    result["hessian"] = {"ad": hess.tolist(), "fd": hess_fd.tolist(), "seconds": t_hess,
                         "laplace_std": np.sqrt(np.diag(cov)).tolist(),
                         "laplace_corr": (cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))).tolist()}
    log("hessian")
    Path(out).write_text(json.dumps(result))
    log(f"wrote {out}")


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
