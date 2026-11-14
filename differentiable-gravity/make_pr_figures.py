"""Static PNG figures for the upstream PR descriptions, from the JSON written by compute_gravity_gradients.py and
greenstone_demo.py (plotting only, no model evaluation).

    python make_pr_figures.py <out_dir>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

HERE = Path(__file__).parent
C = {"fd": "0.15", "current": "0.55", "pr2_only": "#e69f00", "pr1_pr2": "#d62728"}
LABEL = {"current": "main (rounded ids, hard masks)", "pr2_only": "blend_properties only",
         "pr1_pr2": "blend_properties + erosion_sigmoid_slope"}
UNIT_NAMES = ["EarlyGranite", "SimpleMafic2", "SimpleBIF", "SimpleMafic1", "basement"]
UNIT_COLORS = ["#e9a3a3", "#7fb77e", "#4b6cb7", "#b5d99c", "#d9d2c3"]
MARKERS = {"EarlyGranite": "o", "SimpleMafic2": "s", "SimpleBIF": "D", "SimpleMafic1": "^"}


def style():
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 8, "axes.linewidth": 0.8,
                         "axes.edgecolor": "0.3", "savefig.dpi": 160, "figure.dpi": 110})


def rel_rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2)) / np.abs(b).max()


def synthetic(out):
    d = json.loads((HERE / "data/gravity_gradients.json").read_text())
    x = np.array(d["receivers_x"])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharex=True)
    short = {"current": "main", "pr2_only": "blend only", "pr1_pr2": "both"}
    for ax, group, title in zip(axes, ("erosion", "old"),
                                ("height of the erosion surface (between stacks)",
                                 "height of the dipping surface (inside a stack)")):
        ax.plot(x, d["fd"][group], "--", color=C["fd"], lw=1.3, label=f"finite difference of main (±{d['fd_h']:g} m)")
        for k in ("current", "pr2_only", "pr1_pr2"):
            ax.plot(x, d["ad"][k][group], "o", color=C[k], ms=4, label=f"autograd, {LABEL[k]}")
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        errors = ", ".join(f"{short[k]} {rel_rmse(d['ad'][k][group], d['fd'][group]):.0%}"
                           for k in ("current", "pr2_only", "pr1_pr2"))
        ax.set_title(f"∂Δg/∂z, {title}\nrelative RMSE vs FD: {errors}")
        ax.set_xlabel("receiver x (m)")
    axes[0].set_ylabel("mGal per m")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "pr2_synthetic_gradients.png")
    plt.close(fig)

    inv = d["inversion"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for k in ("current", "pr1_pr2"):
        axes[0].semilogy(inv[k]["loss"], color=C[k], lw=1.6, label=LABEL[k])
        p = np.array(inv[k]["path"])
        axes[1].plot(p[:, 0], p[:, 1], "-o", color=C[k], ms=2.5, lw=1)
    axes[1].plot(*inv["truth"], "*", ms=14, color="gold", mec="k", label="truth")
    axes[1].plot(*inv["start"], "x", ms=9, color="k", label="start")
    axes[0].set(xlabel="Adam step", ylabel="misfit", title="misfit")
    axes[1].set(xlabel="shift of the erosion surface (m)", ylabel="shift of the dipping surface (m)",
                title="parameter path")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pr2_synthetic_inversion.png")
    plt.close(fig)
    return d


def greenstone(out):
    d = json.loads((HERE / "data/greenstone_demo.json").read_text())
    pts, forms = np.array(d["surface_points"]), np.array(d["point_formation"])
    rec = np.array(d["receivers"])
    j_cur, j_pr = np.array(d["jacobian"]["current"]), np.array(d["jacobian"]["pr1_pr2"])
    s_cur, s_pr = np.linalg.norm(j_cur, axis=0), np.linalg.norm(j_pr, axis=0)
    ext = [v / 1000 for v in d["extent"][:4]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    s = d["sections"]["current"]
    axes[0].imshow(np.array(s["plan"]), origin="lower", extent=ext, cmap=ListedColormap(UNIT_COLORS),
                   vmin=0.5, vmax=5.5, interpolation="nearest")
    axes[0].plot(rec[:, 0] / 1000, rec[:, 1] / 1000, "v", color="k", ms=3)
    for name, c in zip(UNIT_NAMES, UNIT_COLORS):
        axes[0].plot([], [], "s", color=c, ms=8, label=name)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7, frameon=False)
    axes[0].set_title("Greenstone model, map at z = −1.5 km, receivers")
    vmax = s_pr.max()
    for ax, sens, title in ((axes[1], s_cur, "main"), (axes[2], s_pr, "blend_properties + erosion_sigmoid_slope")):
        for f, mk in MARKERS.items():
            sel = forms == f
            sc = ax.scatter(pts[sel, 0] / 1000, pts[sel, 1] / 1000, c=sens[sel], marker=mk, s=42, cmap="plasma",
                            vmin=0, vmax=vmax, edgecolors="k", linewidths=0.4, label=f)
        ax.set_title(f"‖∂Δg/∂z‖ per surface point: {title}\n{int((sens > 0).sum())} of {len(sens)} non-zero")
        ax.set_xlim(ext[0], ext[1])
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=7, frameon=False)
    fig.colorbar(sc, ax=axes[2], shrink=0.8, label="mGal per m")
    for ax in axes:
        ax.set_xlabel("Easting (km)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Northing (km)")
    fig.tight_layout()
    fig.savefig(out / "pr2_greenstone_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    fd_stats = []
    for ax, p in zip(axes, d["fd_points"]):
        fd, ad = np.array(d["fd"][str(p)]), j_pr[:, p]
        lim = np.abs(np.r_[fd, ad]).max() * 1.1
        ax.plot([-lim, lim], [-lim, lim], ":", color="0.6")
        ax.plot(fd, j_cur[:, p], "o", color=C["current"], ms=3, label="main")
        ax.plot(fd, ad, "o", color=C["pr1_pr2"], ms=3, label="this PR")
        ax.set_title(f"{forms[p]} (point {p})\nrel. RMSE {rel_rmse(ad, fd):.0%}, r = {np.corrcoef(ad, fd)[0, 1]:.2f}")
        ax.set_xlabel("finite difference (mGal/m)")
        ax.ticklabel_format(style="sci", scilimits=(-2, 2))
        fd_stats.append((forms[p], p, rel_rmse(ad, fd), np.corrcoef(ad, fd)[0, 1]))
    axes[0].set_ylabel("autograd (mGal/m)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pr2_greenstone_fd_check.png")
    plt.close(fig)

    inv = d["inversion"]
    std = np.array(d["hessian"]["laplace_std"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for k in ("current", "pr1_pr2"):
        axes[0].semilogy(inv[k]["loss"], color=C[k], lw=1.6, label=LABEL[k])
    path = np.array(inv["pr1_pr2"]["path"])
    for i, f in enumerate(d["formations"]):
        axes[1].plot(path[:, i], color=UNIT_COLORS[i], lw=2,
                     label=f"{f}: {path[-1, i]:+.0f} m ({abs(path[-1, i]) / std[i]:.1f}σ)")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set(xlabel="Adam step", ylabel="misfit", title="misfit")
    axes[1].set(xlabel="Adam step", ylabel="depth shift (m)", title="horizon depth shifts (truth 0)")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pr2_greenstone_inversion.png")
    plt.close(fig)
    return d, fd_stats


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    synthetic(out)
    greenstone(out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
