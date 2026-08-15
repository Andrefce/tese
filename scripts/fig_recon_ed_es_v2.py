"""Render the reconstruction figure for the Results chapter (new v2 cohort).

Writes ``images/results_recon_ed_es.png``: the watertight reconstructed surfaces
beside the segmentation-derived ("true") meshes, at both cardiac phases. Meshes
are read from the cohort cache written by the evaluation run, so nothing is
recomputed here.

    C:/Python313/python.exe scripts/fig_recon_ed_es_v2.py [patientID]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

THESIS = Path(__file__).resolve().parents[1]
CACHE = THESIS / "test-new-model" / "cohort_full_nor_hcm10" / "cache"
OUT = THESIS / "images" / "results_recon_ed_es.png"

PATIENT = sys.argv[1] if len(sys.argv) > 1 else "patient061"
ENDO_COLOUR = "#c1121f"
EPI_COLOUR = "#adb5bd"

pv.global_theme.font.family = "times"


def load(tag: str) -> pv.PolyData:
    return pv.read(CACHE / f"{PATIENT}_{tag}.ply")


def panel(plotter, row: int, col: int, endo, epi, title: str, frame) -> None:
    plotter.subplot(row, col)
    plotter.add_text(title, font_size=11, position="upper_edge", color="black")
    plotter.add_mesh(frame, opacity=0.0, show_scalar_bar=False)  # common framing
    plotter.add_mesh(epi, color=EPI_COLOUR, opacity=0.28, smooth_shading=True,
                     specular=0.2)
    plotter.add_mesh(endo, color=ENDO_COLOUR, opacity=1.0, smooth_shading=True,
                     specular=0.3)


def main() -> None:
    surfaces = {}
    for phase in ("ED", "ES"):
        for kind in ("model", "voxel"):
            for wall in ("endo", "epi"):
                surfaces[(phase, kind, wall)] = load(f"{phase}_{kind}_{wall}")

    # Shared framing: an invisible box spanning every surface, added to each
    # panel so all four are drawn at the same scale.
    bounds = np.array([s.bounds for s in surfaces.values()])
    focus = (bounds[:, 0].min(), bounds[:, 1].max(),
             bounds[:, 2].min(), bounds[:, 3].max(),
             bounds[:, 4].min(), bounds[:, 5].max())
    frame = pv.Box(bounds=focus)

    plotter = pv.Plotter(shape=(2, 2), window_size=(1500, 1250), off_screen=True,
                         border=False)
    plotter.set_background("white")

    layout = [
        (0, 0, "ED", "model", "(a) Reconstruction, end-diastole"),
        (0, 1, "ED", "voxel", "(b) Segmentation-derived, end-diastole"),
        (1, 0, "ES", "model", "(c) Reconstruction, end-systole"),
        (1, 1, "ES", "voxel", "(d) Segmentation-derived, end-systole"),
    ]
    for row, col, phase, kind, title in layout:
        panel(plotter, row, col,
              surfaces[(phase, kind, "endo")], surfaces[(phase, kind, "epi")],
              title, frame)
        plotter.view_vector((1.6, -1.6, 0.0), viewup=(0.0, 0.0, 1.0))
        plotter.reset_camera()
        plotter.camera.zoom(0.9)

    plotter.link_views()
    plotter.screenshot(str(OUT), transparent_background=False)
    plotter.close()
    print(f"Wrote {OUT.relative_to(THESIS)} from {PATIENT}")
    for key, mesh in surfaces.items():
        print(f"  {key}: watertight={mesh.is_manifold} "
              f"n_faces={mesh.n_faces_strict}")


if __name__ == "__main__":
    main()
