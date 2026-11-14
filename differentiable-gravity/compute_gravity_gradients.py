"""Data for the report: differentiable gravity (fork PRs 1 + 2, branch feat/differentiable-gravity).

Two stacks: a flat erosion surface (Young, ERODE) at z=600 m cutting a 30-degree dipping surface (Old) over basement;
19 gravity receivers on top. Writes a JSON with:
  - forward gravity, current engine vs PRs 1+2;
  - d(gravity)/dz of each surface: finite difference of the current model (truth) vs autograd for
    current / PR 2 only / PRs 1+2;
  - a two-parameter inversion (depth of each surface) from synthetic gravity by Adam, current vs PRs 1+2;
  - the 2x2 Hessian of the misfit at the truth, autograd (double backward) vs finite difference of the gradient.

Run with PYTHONPATH pointing at a checkout of feat/differentiable-gravity:
    PYTHONPATH=<engine checkout> python compute_gravity_gradients.py out.json
"""

import json
import sys
import time
from pathlib import Path

import gempy as gp
import gempy_engine
import numpy as np
import torch

HERE = Path(__file__).parent
DATA = HERE / "data"
RECEIVERS_X = np.arange(50, 1000, 50.0)
KERNEL_RESOLUTION = np.array([10, 10, 100])
DENSITIES = np.array([2.0, 2.6, 3.1])
SLOPE = 100.0
FD_H = 30.0
GROUPS = {"erosion": slice(0, 4), "old": slice(4, 8)}
CONFIGS = {  # name: (blend_properties, erosion_sigmoid_slope, sigmoid_slope)
    "current": (False, None, 5_000_000),
    "pr2_only": (True, None, SLOPE),
    "pr1_pr2": (True, SLOPE, SLOPE),
}

SP_CSV = """X,Y,Z,formation
200,500,600,unconf
800,500,600,unconf
500,200,600,unconf
500,800,600,unconf
200,500,327,old
800,500,673,old
500,200,500,old
500,800,500,old
"""
ORI_CSV = """X,Y,Z,azimuth,dip,polarity,formation
500,500,600,90,0,1,unconf
500,500,500,270,30,1,old
"""


def build(config, shifts=None):
    blend, erosion_slope, sigmoid_slope = CONFIGS[config]
    m = gp.create_geomodel(
        project_name="grav", extent=[0, 1000, 0, 1000, 0, 1000], refinement=1,
        importer_helper=gp.data.ImporterHelper(path_to_orientations=str(DATA / "ori.csv"),
                                               path_to_surface_points=str(DATA / "sp.csv")))
    gp.map_stack_to_surfaces(m, {"Young": ("unconf",), "Old": ("old",)})
    receivers = np.c_[RECEIVERS_X, np.full_like(RECEIVERS_X, 500.0), np.full_like(RECEIVERS_X, 1000.0)]
    gp.set_centered_grid(m.grid, centers=receivers, resolution=KERNEL_RESOLUTION, radius=np.full(3, 1000.0))
    tz = gp.calculate_gravity_gradient(m.grid.centered_grid)
    m.geophysics_input = gp.data.GeophysicsInput(tz=tz, densities=DENSITIES, blend_properties=blend)
    m.interpolation_options.mesh_extraction = False
    m.interpolation_options.cache_mode = gp.data.InterpolationOptions.CacheMode.NO_CACHE
    m.interpolation_options.erosion_sigmoid_slope = erosion_slope
    m.interpolation_options.sigmoid_slope = sigmoid_slope
    for group, dz in (shifts or {}).items():
        rows = np.arange(8)[GROUPS[group]]
        z0 = m.surface_points_copy.df["Z"].values.copy()
        gp.modify_surface_points(m, slice=rows, Z=z0[rows] + dz)
    return m


def gravity_numpy(config, shifts=None):
    m = build(config, shifts)
    sol = gp.compute_model(m, engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy))
    return np.asarray(sol.gravity, dtype=float)


class TorchForward:
    """gravity(theta), theta = (dz erosion surface, dz old surface) in metres, differentiable."""

    def __init__(self, config):
        self.m = build(config)
        gp.compute_model(self.m, engine_config=gp.data.GemPyEngineConfig(
            backend=gp.data.AvailableBackends.PYTORCH, dtype="float64", compute_grads=True))
        self.ii = self.m.taped_interpolation_input
        self.base = self.ii.surface_points.sp_coords.detach().clone()
        self.scale = float(self.m.input_transform.scale[2])

    def __call__(self, theta):
        shift = torch.zeros_like(self.base)
        shift[GROUPS["erosion"], 2] = theta[0] * self.scale
        shift[GROUPS["old"], 2] = theta[1] * self.scale
        self.ii.surface_points.sp_coords = self.base + shift
        sol = gempy_engine.compute_model(self.ii, self.m.interpolation_options, self.m.input_data_descriptor,
                                         geophysics_input=self.m.geophysics_input)
        return sol.gravity


def jacobian(forward):
    theta = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    g = forward(theta)
    if not g.requires_grad:
        return np.zeros((len(RECEIVERS_X), 2)), g.detach().numpy()
    rows = [torch.autograd.grad(g[k], theta, retain_graph=True, allow_unused=True)[0] for k in range(len(g))]
    jac = np.array([[0.0, 0.0] if r is None else r.detach().numpy() for r in rows])
    return jac, g.detach().numpy()


def invert(forward, g_obs, start, steps, lr):
    theta = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    obs = torch.as_tensor(g_obs)
    path, losses = [], []
    for _ in range(steps):
        opt.zero_grad()
        g = forward(theta)
        loss = ((g - obs) ** 2).mean() / 1e-4  # misfit in units of (0.01 mGal)^2
        if loss.requires_grad:
            loss.backward()
        path.append(theta.detach().numpy().tolist())
        losses.append(float(loss))
        opt.step()
    return path, losses


def misfit_hessian(forward, g_obs, h=2.0):
    obs = torch.as_tensor(g_obs)

    def loss(theta):
        return ((forward(theta) - obs) ** 2).mean() / 1e-4

    def gradient(theta_np):
        t = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
        (gr,) = torch.autograd.grad(loss(t), t)
        return gr.numpy()

    t0 = time.time()
    hess_ad = torch.autograd.functional.hessian(loss, torch.tensor([0.0, 0.0], dtype=torch.float64)).numpy()
    t_ad = time.time() - t0
    hess_fd = np.array([(gradient(np.eye(2)[i] * h) - gradient(-np.eye(2)[i] * h)) / (2 * h) for i in range(2)])
    return hess_ad, hess_fd, t_ad


def main(out):
    DATA.mkdir(exist_ok=True)
    (DATA / "sp.csv").write_text(SP_CSV)
    (DATA / "ori.csv").write_text(ORI_CSV)
    result = {"receivers_x": RECEIVERS_X.tolist(), "fd_h": FD_H, "slope": SLOPE}

    result["fd"] = {}
    for group in GROUPS:
        plus = gravity_numpy("current", {group: FD_H})
        minus = gravity_numpy("current", {group: -FD_H})
        result["fd"][group] = ((plus - minus) / (2 * FD_H)).tolist()

    result["ad"], result["forward"] = {}, {}
    for config in CONFIGS:
        jac, g = jacobian(TorchForward(config))
        result["ad"][config] = {group: jac[:, i].tolist() for i, group in enumerate(GROUPS)}
        result["forward"][config] = g.tolist()

    # Synthetic data from each configuration's own forward model at the truth (as in a synthetic inversion test),
    # so the truth is the misfit minimum of the model being inverted.
    start, steps, lr = [-80.0, 60.0], 150, 6.0
    result["inversion"] = {"start": start, "truth": [0.0, 0.0], "steps": steps, "lr": lr}
    for config in ("current", "pr1_pr2"):
        g_obs = np.array(result["forward"][config])
        t0 = time.time()
        path, losses = invert(TorchForward(config), g_obs, start, steps, lr)
        result["inversion"][config] = {"path": path, "loss": losses, "seconds": time.time() - t0}

    hess_ad, hess_fd, t_ad = misfit_hessian(TorchForward("pr1_pr2"), np.array(result["forward"]["pr1_pr2"]))
    result["hessian"] = {"ad": hess_ad.tolist(), "fd": hess_fd.tolist(), "seconds_ad": t_ad}
    Path(out).write_text(json.dumps(result))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
