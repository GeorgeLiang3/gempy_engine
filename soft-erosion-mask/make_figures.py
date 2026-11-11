"""Hard vs soft erosion mask in gempy_engine (branch feat/soft-erosion-mask), thesis-style figures.

Model: a flat erosion surface (Young, ERODE) at z=600 m cutting a 30-degree dipping surface (Old) over basement.
Densities are linear in unit id (Young 2.0, Old 2.5, basement 3.0 g/cc): the soft mask blends unit ids, so only a
linear id->density map turns the blend into the right density blend. Gravity is computed here from final_block with
gempy's centered-grid kernel (sum of density contrast x tz); gempy_engine's own compute_gravity indexes densities by
integer id and is not differentiable in geometry.

    fig1_gravity.png  - Fig 3.8 style: gravity profile / cross-section / d(gravity)/dz_erosion at the receivers
    fig2_sections.png - Fig 3.11 style: hard / soft / reference columns, unit-density and sensitivity maps

Run with an environment that has gempy + the fork's gempy_engine installed:
    python make_figures.py
"""

from pathlib import Path

import gempy as gp
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

HERE = Path(__file__).parent
DATA = HERE / "data"
SOFT_SLOPE = 100.0  # erosion_sigmoid_slope (blend ~40 m wide in this model)
Z_EROSION = 600.0
RHO = {1: 2.0, 2: 2.5, 3: 3.0}  # g/cc
RECEIVERS_X = np.arange(50, 1000, 50.0)
KERNEL_RESOLUTION = np.array([10, 10, 200])
KERNEL_RADIUS = np.array([1000.0, 1000.0, 1000.0])
FD_H_GRAV = 30.0  # m, central difference of the hard model at the receivers
FD_H_SECTION = 10.0  # m, central difference of the hard model per section cell

C_HARD, C_SOFT, C_REF = "tab:blue", "tab:red", "0.25"
UNIT_COLORS = ["#f3e6b5", "#cfe6c4", "#c9d9ec"]  # Young, Old, basement
C_EROSION, C_OLD = "crimson", "teal"

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


# --------------------------------------------------------------------------------------------- model
def write_inputs():
    DATA.mkdir(exist_ok=True)
    (DATA / "sp.csv").write_text(SP_CSV)
    (DATA / "ori.csv").write_text(ORI_CSV)


def erosion_rows(m):
    return np.where(np.isclose(m.surface_points_copy.df["Z"].values, Z_EROSION))[0]


def build(slope, grid, erosion_dz=0.0):
    """grid: ("section", xyz) for a custom grid, or ("gravity", receivers) for the centered kernel grid."""
    m = gp.create_geomodel(
        project_name="soft_erosion",
        extent=[0, 1000, 0, 1000, 0, 1000],
        refinement=1,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=str(DATA / "ori.csv"), path_to_surface_points=str(DATA / "sp.csv")
        ),
    )
    gp.map_stack_to_surfaces(m, {"Young": ("unconf",), "Old": ("old",)})
    kind, values = grid
    if kind == "section":
        gp.set_custom_grid(m.grid, xyz_coord=values)
        m.grid.active_grids = gp.data.Grid.GridTypes.CUSTOM
    else:
        gp.set_centered_grid(m.grid, centers=values, resolution=KERNEL_RESOLUTION, radius=KERNEL_RADIUS)
    m.interpolation_options.mesh_extraction = False
    m.interpolation_options.erosion_sigmoid_slope = slope
    if erosion_dz:
        rows = erosion_rows(m)
        z0 = m.surface_points_copy.df["Z"].values.copy()
        gp.modify_surface_points(m, slice=rows, Z=z0[rows] + erosion_dz)
    return m


def compute(m, torch_backend=False):
    cfg = (
        gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.PYTORCH, dtype="float64", compute_grads=True)
        if torch_backend
        else gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy)
    )
    gp.compute_model(m, engine_config=cfg)
    return m.solutions.octrees_output[0].outputs_centers


def grid_slice(outs, kind):
    g = outs[0].grid
    return g.custom_grid_slice if kind == "section" else g.geophysics_grid_slice


def density(ids):
    return RHO[1] + (ids - 1) * (RHO[3] - RHO[1]) / 2


def gravity(ids, tz):
    """Delta g (mGal) at each receiver from the density contrast to the Young unit."""
    drho = density(ids) - RHO[1]
    tz = torch.as_tensor(tz) if torch.is_tensor(drho) else tz
    return (drho.reshape(len(RECEIVERS_X), -1) * tz.reshape(1, -1)).sum(1)


def forward(slope, grid, erosion_dz=0.0, tz=None):
    m = build(slope, grid, erosion_dz)
    outs = compute(m)
    s = grid_slice(outs, grid[0])
    ids = np.asarray(outs[-1].final_block[s])
    res = {"ids": ids, "fields": [np.asarray(o.exported_fields.scalar_field[s]) for o in outs],
           "sfai": [np.asarray(o.scalar_field_at_sp) for o in outs]}
    if tz is not None:
        res["g"] = gravity(ids, tz)
    return res


def ad_directional(slope, grid, quantity):
    """d(quantity)/d(z of the erosion surface) by autograd: forward-mode via double backward, one pass."""
    m = build(slope, grid)
    outs = compute(m, torch_backend=True)
    y = quantity(outs[-1].final_block[grid_slice(outs, grid[0])])
    sp = m.taped_interpolation_input.surface_points.sp_coords
    u = torch.zeros_like(sp)
    u[erosion_rows(m), 2] = 1.0  # all erosion-surface points move up together
    lam = torch.ones_like(y, requires_grad=True)
    (g,) = torch.autograd.grad(y, sp, grad_outputs=lam, create_graph=True, allow_unused=True)
    if g is None or not g.requires_grad:  # hard mask: the graph does not reach the contact at all
        return np.zeros(tuple(y.shape))
    (jv,) = torch.autograd.grad((g * u).sum(), lam, allow_unused=True)
    jv = torch.zeros_like(y) if jv is None else jv
    return jv.detach().numpy() * m.input_transform.scale[2]  # per metre of shift


# --------------------------------------------------------------------------------------------- style
def set_style():
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
        "axes.edgecolor": "0.3", "figure.dpi": 150, "savefig.dpi": 200,
    })


def caption(fig, text, y=0.01):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8, wrap=True)


def draw_contacts(ax, xs, zs, res, lw=1.2):
    shape = (len(zs), len(xs))
    young, old = res["fields"][0].reshape(shape), res["fields"][1].reshape(shape)
    e = np.min(res["sfai"][0])
    ax.contour(xs, zs, young, levels=[e], colors=C_EROSION, linewidths=lw)
    old_below = np.ma.masked_where(young > e, old)
    ax.contour(xs, zs, old_below, levels=sorted(np.atleast_1d(res["sfai"][1])), colors=C_OLD, linewidths=lw)


# --------------------------------------------------------------------------------------------- figure 1
def figure_gravity(xs, zs, section_hard):
    receivers = np.c_[RECEIVERS_X, np.full_like(RECEIVERS_X, 500.0), np.full_like(RECEIVERS_X, 1000.0)]
    grav = ("gravity", receivers)
    tz = gp.calculate_gravity_gradient(build(None, grav).grid.centered_grid)

    g_hard = forward(None, grav, tz=tz)["g"]
    g_soft = forward(SOFT_SLOPE, grav, tz=tz)["g"]
    fd = (forward(None, grav, +FD_H_GRAV, tz=tz)["g"] - forward(None, grav, -FD_H_GRAV, tz=tz)["g"]) / (2 * FD_H_GRAV)
    ad_hard = ad_directional(None, grav, lambda b: gravity(b, tz))
    ad_soft = ad_directional(SOFT_SLOPE, grav, lambda b: gravity(b, tz))
    rmse = lambda a: np.sqrt(np.mean((a - fd) ** 2))

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.15, 1.0], "hspace": 0.12})
    ax = axes[0]
    ax.plot(RECEIVERS_X, g_hard, "--", color="0.7", lw=0.8, zorder=0)
    ax.plot(RECEIVERS_X, g_hard, "o", mfc="none", mec=C_HARD, ms=6, mew=1.0, label="hard mask (current)")
    ax.plot(RECEIVERS_X, g_soft, "o", color=C_SOFT, ms=2.5, label=f"soft erosion mask (slope {SOFT_SLOPE:g})")
    ax.set_ylabel(r"$\Delta g$ (mGal)")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    ax.imshow(np.rint(section_hard["ids"]).reshape(len(zs), len(xs)), origin="lower",
              extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap=ListedColormap(UNIT_COLORS), vmin=0.5, vmax=3.5,
              aspect="auto", interpolation="nearest")
    draw_contacts(ax, xs, zs, section_hard, lw=1.4)
    ax.plot(RECEIVERS_X, np.full_like(RECEIVERS_X, 985), "v", color="red", ms=5, clip_on=False)
    ax.annotate("erosion surface (parameter $z_e$)", xy=(120, Z_EROSION), xytext=(120, 780), fontsize=7.5,
                color=C_EROSION, arrowprops=dict(arrowstyle="->", color=C_EROSION, lw=0.8))
    for x, z, t in [(860, 820, "Young  2.0 g/cc"), (150, 450, "Old  2.5 g/cc"), (760, 250, "basement  3.0 g/cc")]:
        ax.text(x, z, t, fontsize=7.5, ha="center", color="0.25")
    ax.set_ylabel("z (m)")
    ax.set_ylim(0, 1000)

    ax = axes[2]
    ax.plot(RECEIVERS_X, fd, "--", color=C_REF, lw=1.2, label=fr"finite difference, hard model ($\pm${FD_H_GRAV:g} m)")
    ax.plot(RECEIVERS_X, ad_hard, "o", color=C_HARD, ms=3.5, label=f"autograd, hard mask  (RMSE {rmse(ad_hard):.2e})")
    ax.plot(RECEIVERS_X, ad_soft, "o", color=C_SOFT, ms=3.5, label=f"autograd, soft mask  (RMSE {rmse(ad_soft):.2e})")
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.set_ylabel(r"$\partial\Delta g / \partial z_e$ (mGal/m)")
    ax.set_xlabel("x (m)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), frameon=False)
    ax.set_xlim(0, 1000)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.98, bottom=0.17)
    caption(fig, "Figure 1: Upper row: forward gravity at 19 receivers with the hard mask (current gempy_engine) and the "
                 "soft erosion mask;\nmiddle row: cross-section at y = 500 m, receivers in red; lower row: derivative "
                 "of gravity with respect to the height of the erosion surface,\nby autograd for both masks and by "
                 "finite difference of the hard model. The hard mask gives no gradient at all.")
    out = HERE / "fig1_gravity.png"
    fig.savefig(out)
    print(f"wrote {out}  rmse hard={rmse(ad_hard):.3e} soft={rmse(ad_soft):.3e}  fd max={np.abs(fd).max():.3e}")


# --------------------------------------------------------------------------------------------- figure 2
def figure_sections():
    xs, zs = np.linspace(0, 1000, 201), np.linspace(450, 750, 151)
    X, Z = np.meshgrid(xs, zs)
    sec = ("section", np.c_[X.ravel(), np.full(X.size, 500.0), Z.ravel()])
    shape = (len(zs), len(xs))
    ext = [xs[0], xs[-1], zs[0], zs[-1]]

    hard, soft = forward(None, sec), forward(SOFT_SLOPE, sec)
    ad_hard = ad_directional(None, sec, density)
    ad_soft = ad_directional(SOFT_SLOPE, sec, density)
    fd = (density(forward(None, sec, +FD_H_SECTION)["ids"]) - density(forward(None, sec, -FD_H_SECTION)["ids"])) / (
        2 * FD_H_SECTION)

    dz = zs[1] - zs[0]
    col = lambda a: a.reshape(shape).sum(0) * dz  # column integral: what a gravity receiver above sees
    col_err = lambda a: np.mean(np.abs(col(a) - col(fd))) / np.mean(np.abs(col(fd)))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
    rho_kw = dict(origin="lower", extent=ext, aspect="auto", interpolation="nearest", cmap="cividis", vmin=2.0, vmax=3.0)
    for ax, res, title in [(axes[0, 0], hard, "Hard mask (current)"),
                           (axes[0, 1], soft, f"Soft erosion mask, slope {SOFT_SLOPE:g}")]:
        im_rho = ax.imshow(density(res["ids"]).reshape(shape), **rho_kw)
        draw_contacts(ax, xs, zs, hard, lw=0.9)
        ax.set_title(f"{title}\ndensity (g/cc)")
    diff = (density(soft["ids"]) - density(hard["ids"])).reshape(shape)
    im_diff = axes[0, 2].imshow(diff, origin="lower", extent=ext, aspect="auto", interpolation="nearest",
                                cmap="twilight_shifted", vmin=-0.5, vmax=0.5)
    draw_contacts(axes[0, 2], xs, zs, hard, lw=0.9)
    axes[0, 2].set_title(f"Soft $-$ hard\nmax |difference| {np.abs(diff).max():.2f} g/cc, only at the erosion contact")

    vmax = np.percentile(np.abs(np.r_[ad_soft, fd]), 99.5)
    sens_kw = dict(origin="lower", extent=ext, aspect="auto", interpolation="nearest", cmap="twilight_shifted",
                   vmin=-vmax, vmax=vmax)
    for ax, s, title in [
        (axes[1, 0], ad_hard, f"Hard mask, autograd\ncolumn error vs FD: {col_err(ad_hard):.0%}"),
        (axes[1, 1], ad_soft, f"Soft erosion mask, autograd\ncolumn error vs FD: {col_err(ad_soft):.1%}"),
        (axes[1, 2], fd, fr"Finite difference, hard model ($\pm${FD_H_SECTION:g} m)" + "\nreference"),
    ]:
        im_s = ax.imshow(s.reshape(shape), **sens_kw)
        draw_contacts(ax, xs, zs, hard, lw=0.6)
        ax.set_title(title)
    for ax in axes[:, 0]:
        ax.set_ylabel("z (m)")
    for ax in axes[1]:
        ax.set_xlabel("x (m)")

    fig.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.2, wspace=0.2, hspace=0.32)
    for ax, im, label in [(axes[0, 1], im_rho, "density (g/cc)"), (axes[0, 2], im_diff, "difference (g/cc)"),
                          (axes[1, 2], im_s, r"$\partial\rho/\partial z_e$ (g/cc per m)")]:
        cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
        fig.colorbar(im, cax=cax).set_label(label, fontsize=7.5)
    caption(fig, "Figure 2: Cross-sections at y = 500 m around the erosion contact (crimson; teal: the eroded Old "
                 "surface). Upper row: density from the hard and the soft mask and their difference.\nLower row: "
                 "per-cell derivative of density with respect to the height of the erosion surface; column error is "
                 "the mean relative error of its vertical integral against the finite-difference reference.",
            y=0.02)
    out = HERE / "fig2_sections.png"
    fig.savefig(out)
    print(f"wrote {out}  column error hard={col_err(ad_hard):.3f} soft={col_err(ad_soft):.3f}")


def main():
    write_inputs()
    set_style()
    figure_sections()
    xs, zs = np.linspace(0, 1000, 201), np.linspace(0, 1000, 201)
    X, Z = np.meshgrid(xs, zs)
    section_full = forward(None, ("section", np.c_[X.ravel(), np.full(X.size, 500.0), Z.ravel()]))
    figure_gravity(xs, zs, section_full)


if __name__ == "__main__":
    main()
