"""Results-chapter figures for the NOR cohort (CardioSDF model vs derived mesh).

Reads:
  * representative_nor.npz          (scripts/fig_results_nor_data.py)
  * cohort_aha17_thickening.csv     (CardioSDF cohort AHA-17, NOR filter)
  * derived_aha17_nor.csv           (derived cohort AHA-17)

Writes three figures to images/ :
  results_nor_recon.png       -- CardioSDF vs derived endo/epi surfaces, ED & ES.
  results_nor_thickness.png   -- endocardium coloured by Laplace wall thickness,
                                 CardioSDF vs derived, shared colour bar.
  results_nor_thickening.png  -- CardioSDF surface coloured by ED->ES regional
                                 thickening (%) beside the AHA-17 bullseye.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pyvista as pv  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from generate_patient002_methodology_figures import (  # noqa: E402
    draw_aha17,
    save_rgb_figure,
)

pv.OFF_SCREEN = True

OUT_IMG = ROOT / "images"
OUT_IMG.mkdir(exist_ok=True)
DATA = ROOT / "scripts" / "webapp" / "notebooks" / "outputs" / "cohort_single"
NPZ = DATA / "representative_nor.npz"

ENDO_COLOR = "#C1443B"      # warm red
EPI_COLOR = "#9FB2C6"       # cool grey-blue
THICK_CMAP = "turbo"
THICKEN_CMAP = "turbo"
AHA_NAMES = [
    "Basal Anterior", "Basal Anteroseptal", "Basal Inferoseptal", "Basal Inferior",
    "Basal Inferolateral", "Basal Anterolateral", "Mid Anterior", "Mid Anteroseptal",
    "Mid Inferoseptal", "Mid Inferior", "Mid Inferolateral", "Mid Anterolateral",
    "Apical Anterior", "Apical Septal", "Apical Inferior", "Apical Lateral", "Apex",
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _load_npz() -> dict:
    d = np.load(NPZ, allow_pickle=True)
    return {k: d[k] for k in d.files}


def orient_pair(endo_v: np.ndarray, epi_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre both surfaces and flip the long axis so the apex points down."""
    c = np.vstack([endo_v, epi_v]).mean(0)
    e = (endo_v - c).astype(float)
    p = (epi_v - c).astype(float)
    zt = e[:, 2]
    hi = e[zt > np.percentile(zt, 75)]
    lo = e[zt < np.percentile(zt, 25)]
    r_hi = np.hypot(hi[:, 0], hi[:, 1]).mean() if len(hi) else 1.0
    r_lo = np.hypot(lo[:, 0], lo[:, 1]).mean() if len(lo) else 1.0
    if r_hi < r_lo:          # apex (small radius) currently on top -> flip
        e[:, 2] *= -1.0
        p[:, 2] *= -1.0
    return e, p


def orient_single(endo_v: np.ndarray) -> np.ndarray:
    e = (endo_v - endo_v.mean(0)).astype(float)
    zt = e[:, 2]
    hi = e[zt > np.percentile(zt, 75)]
    lo = e[zt < np.percentile(zt, 25)]
    if (np.hypot(hi[:, 0], hi[:, 1]).mean() if len(hi) else 1) < \
       (np.hypot(lo[:, 0], lo[:, 1]).mean() if len(lo) else 1):
        e[:, 2] *= -1.0
    return e


# ──────────────────────────────────────────────────────────────────
# Figure 1 : reconstruction, CardioSDF vs derived, ED & ES
# ──────────────────────────────────────────────────────────────────
MAT = dict(smooth_shading=True, specular=0.05, specular_power=8,
           ambient=0.34, diffuse=0.92, show_scalar_bar=False)


def _pv_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, np.int64)
    return np.hstack([np.full((len(faces), 1), 3, np.int64), faces]).ravel()


def _smooth(V: np.ndarray, faces: np.ndarray, n_iter: int = 25) -> np.ndarray:
    """Volume-preserving Taubin smoothing for display (keeps vertex order)."""
    if n_iter <= 0:
        return np.asarray(V, np.float32)
    m = pv.PolyData(np.asarray(V, np.float32), _pv_faces(faces))
    s = m.smooth_taubin(n_iter=n_iter, pass_band=0.02,
                        normalize_coordinates=True, boundary_smoothing=True,
                        feature_smoothing=False)
    return np.asarray(s.points, np.float32)


def _aim(pl, center: np.ndarray, radius: float, azim: float, elev: float) -> None:
    az, el = np.radians(azim), np.radians(elev)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pl.camera.position = tuple(center + d * radius * 3.4)
    pl.camera.focal_point = tuple(center)
    pl.camera.up = (0.0, 0.0, 1.0)


def _autocrop(img: np.ndarray, pad: int = 10) -> np.ndarray:
    a = np.asarray(img)
    mask = (a[:, :, :3] < 248).any(2)
    if not mask.any():
        return a
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad)
    return a[y0:y1, x0:x1]


def render_surfaces(surfaces: list, azim: float = -62.0, elev: float = 12.0,
                    window: tuple = (900, 1120)) -> np.ndarray:
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    pl.set_background("white")
    try:
        pl.enable_depth_peeling(12)
    except Exception:
        pass
    allV = []
    for s in surfaces:
        mesh = pv.PolyData(np.asarray(s["V"], np.float32), _pv_faces(s["F"]))
        if s.get("scalars") is not None:
            mesh["v"] = np.asarray(s["scalars"], np.float32)
            pl.add_mesh(mesh, scalars="v", cmap=s.get("cmap", "turbo"),
                        clim=s.get("clim"), opacity=s.get("opacity", 1.0),
                        nan_color=s.get("nan_color", "#c9ccd1"), **MAT)
        else:
            pl.add_mesh(mesh, color=s.get("color", "#cccccc"),
                        opacity=s.get("opacity", 1.0), **MAT)
        allV.append(np.asarray(s["V"]))
    allV = np.vstack(allV)
    center = allV.mean(0)
    radius = float(np.linalg.norm(allV - center, axis=1).max())
    _aim(pl, center, radius, azim, elev)
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    img = pl.screenshot(return_img=True)
    pl.close()
    return _autocrop(img)


def fig_recon(D: dict) -> None:
    cols = [("cardio", "CardioSDF (model)"), ("derived", "Derived mesh (no model)")]
    rows = [("ED", "End-diastole"), ("ES", "End-systole")]
    imgs = {}
    for phase, _ in rows:
        for geom, _ in cols:
            ev, ef = D[f"{phase}_{geom}_endo_v"], D[f"{phase}_{geom}_endo_f"]
            pvv, pf = D[f"{phase}_{geom}_epi_v"], D[f"{phase}_{geom}_epi_f"]
            e, p = orient_pair(ev, pvv)
            n_it = 45 if geom == "derived" else 12
            e, p = _smooth(e, ef, n_it), _smooth(p, pf, n_it)
            imgs[(phase, geom)] = render_surfaces([
                dict(V=p, F=pf, color=EPI_COLOR, opacity=0.28),
                dict(V=e, F=ef, color=ENDO_COLOR, opacity=1.0),
            ])
    fig = plt.figure(figsize=(7.0, 7.3), facecolor="white")
    for r, (phase, rlab) in enumerate(rows):
        for c, (geom, clab) in enumerate(cols):
            ax = fig.add_axes([0.06 + 0.475 * c, 0.50 - 0.47 * r, 0.45, 0.45])
            ax.imshow(imgs[(phase, geom)])
            ax.axis("off")
            if r == 0:
                ax.set_title(clab, fontsize=10.5, weight="bold", pad=6)
            if c == 0:
                ax.text(-0.05, 0.5, rlab, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10.5, weight="bold")
    fig.legend(handles=[
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=ENDO_COLOR,
                   markersize=10, label="Endocardium"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=EPI_COLOR,
                   markersize=10, label="Epicardium (translucent)"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.01))
    save_rgb_figure(fig, OUT_IMG / "results_nor_recon.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("  wrote results_nor_recon.png")


# ──────────────────────────────────────────────────────────────────
# Figure 2 : wall-thickness heatmap, CardioSDF vs derived (ED)
# ──────────────────────────────────────────────────────────────────
def fig_thickness(D: dict) -> None:
    pairs = [("cardio", "CardioSDF (model)"), ("derived", "Derived mesh (no model)")]
    vals = []
    for geom, _ in pairs:
        t = np.asarray(D[f"ED_{geom}_thick"], float)[D[f"ED_{geom}_wall"]]
        vals.append(t[np.isfinite(t)])
    allv = np.concatenate(vals)
    vmin = float(np.floor(np.percentile(allv, 3)))
    vmax = float(np.ceil(np.percentile(allv, 97)))
    imgs = {}
    for geom, _ in pairs:
        ev, ef = D[f"ED_{geom}_endo_v"], D[f"ED_{geom}_endo_f"]
        thick = np.asarray(D[f"ED_{geom}_thick"], float).copy()
        thick[~D[f"ED_{geom}_wall"]] = np.nan
        e = orient_single(ev)
        e = _smooth(e, ef, 45 if geom == "derived" else 12)
        imgs[geom] = render_surfaces([dict(V=e, F=ef, scalars=thick,
                                           cmap=THICK_CMAP, clim=(vmin, vmax))])
    fig = plt.figure(figsize=(7.0, 4.3), facecolor="white")
    for c, (geom, lab) in enumerate(pairs):
        ax = fig.add_axes([0.02 + 0.49 * c, 0.15, 0.47, 0.80])
        ax.imshow(imgs[geom])
        ax.axis("off")
        ax.set_title(lab, fontsize=10.5, weight="bold", pad=2)
    cax = fig.add_axes([0.31, 0.09, 0.38, 0.032])
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin, vmax), cmap=THICK_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Wall thickness (mm)", fontsize=9)
    cb.ax.tick_params(labelsize=8, length=2)
    save_rgb_figure(fig, OUT_IMG / "results_nor_thickness.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("  wrote results_nor_thickness.png")


# ──────────────────────────────────────────────────────────────────
# Figure 3 : ED->ES regional thickening (%) — surface + bullseye
# ──────────────────────────────────────────────────────────────────
def cohort_thickening(geom: str) -> np.ndarray:
    """Cohort-mean AHA-17 ED->ES thickening (%) for 'cardiosdf' or 'derived'."""
    if geom == "cardiosdf":
        df = pd.read_csv(DATA / "cohort_aha17_thickening.csv")
        df = df[df["group"].str.upper() == "NOR"] if "group" in df else df
    else:
        df = pd.read_csv(DATA / "derived_aha17_nor.csv")
    pct = np.full(17, np.nan)
    for sid in range(1, 18):
        s = df[df["segment_id"] == sid]
        ed = np.nanmean(s[s["phase"] == "ED"]["mean_mm"])
        es = np.nanmean(s[s["phase"] == "ES"]["mean_mm"])
        pct[sid - 1] = 100.0 * (es - ed) / ed if ed and np.isfinite(ed) else np.nan
    return pct


def _annotate_aha(ax, seg_vals: dict) -> None:
    outline = [pe.withStroke(linewidth=1.7, foreground="white")]
    rings = [(0.72, 1.00, 6, 1, 90.0), (0.45, 0.72, 6, 7, 90.0), (0.20, 0.45, 4, 13, 45.0)]
    for inner, outer, n, first, off in rings:
        rmid = 0.5 * (inner + outer)
        for i in range(n):
            ang = np.radians(off - 360.0 * (i + 0.5) / n)
            ax.text(rmid * np.cos(ang), rmid * np.sin(ang), f"{seg_vals[first + i]:+.0f}",
                    ha="center", va="center", fontsize=6.8, weight="bold",
                    color="#111111", path_effects=outline)
    ax.text(0, 0, f"{seg_vals[17]:+.0f}", ha="center", va="center",
            fontsize=6.8, weight="bold", color="#111111", path_effects=outline)


def fig_thickening(D: dict) -> None:
    pct = cohort_thickening("cardiosdf")
    seg_vals = {i + 1: float(pct[i]) for i in range(17)}
    vmax = float(np.ceil(np.nanmax(pct) / 10) * 10)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)

    endo_v, endo_f = D["ED_cardio_endo_v"], D["ED_cardio_endo_f"]
    aha = D["ED_cardio_aha"]
    vert = np.array([seg_vals.get(int(a), np.nan) for a in aha], float)
    e = orient_single(endo_v)
    e = _smooth(e, endo_f, 12)
    img = render_surfaces([dict(V=e, F=endo_f, scalars=vert,
                                cmap=THICKEN_CMAP, clim=(0.0, vmax))])

    fig = plt.figure(figsize=(7.4, 4.1), facecolor="white")
    axm = fig.add_axes([0.005, 0.13, 0.47, 0.83])
    axm.imshow(img)
    axm.axis("off")
    axm.set_title("(a) CardioSDF surface", fontsize=10.5, weight="bold")

    axb = fig.add_axes([0.50, 0.13, 0.45, 0.83])
    draw_aha17(axb, seg_vals, norm)
    _annotate_aha(axb, seg_vals)
    axb.set_xlim(-1.15, 1.15)
    axb.set_ylim(-1.15, 1.15)
    axb.set_aspect("equal")
    axb.set_axis_off()
    axb.set_title("(b) AHA-17 bullseye", fontsize=10.5, weight="bold")

    cax = fig.add_axes([0.31, 0.06, 0.38, 0.032])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=THICKEN_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("ED\u2192ES wall thickening (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8, length=2)
    save_rgb_figure(fig, OUT_IMG / "results_nor_thickening.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("  wrote results_nor_thickening.png")


def main() -> None:
    D = _load_npz()
    print(f"Representative patient: {D['pid']}")
    fig_recon(D)
    fig_thickness(D)
    if (DATA / "derived_aha17_nor.csv").exists():
        fig_thickening(D)
    else:
        print("  (skipping thickening figure: derived_aha17_nor.csv not ready)")


if __name__ == "__main__":
    main()
