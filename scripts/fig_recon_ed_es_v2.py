"""Render patient002 reconstruction and segmentation-derived surfaces.

The four-panel layout matches the original Results figure. Only the model
surfaces are radially snapped to the observed SAX contours using the same
topology-preserving operation as the cohort evaluator. The comparator meshes
are rendered exactly as cached, without smoothing or deformation.

Output: ``images/results_recon_ed_es.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

THESIS = Path(__file__).resolve().parents[1]
DEMO_OUT = THESIS / "scripts" / "eval_demo" / "outputs"
OUT = THESIS / "images" / "results_recon_ed_es.png"
PATIENT = "patient002"

sys.path.insert(0, str(THESIS / "scripts" / "webapp"))
from core.sdf_model import _snap_mesh_to_contours  # noqa: E402

ENDO_COLOUR = "#c1121f"
EPI_COLOUR = "#adb5bd"

pv.OFF_SCREEN = True
pv.global_theme.font.family = "times"


def _polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.hstack([
        np.full((len(faces), 1), 3, dtype=np.int64),
        np.asarray(faces, dtype=np.int64),
    ]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=np.float32), cells)


def load_surfaces() -> tuple[dict, dict]:
    """Load patient002 meshes and snap only the model to the input contours."""
    surfaces = {}
    diagnostics = {}
    for phase in ("ED", "ES"):
        with np.load(DEMO_OUT / f"demo_{PATIENT}_{phase}.npz") as data:
            contours = np.asarray(data["contours_xyz_mm"], dtype=np.float32)
            tissue = np.asarray(data["contours_tissue"], dtype=np.float32)
            for kind in ("model", "voxel"):
                for wall, label in (("endo", 0), ("epi", 1)):
                    vertices = np.asarray(data[f"{kind}_{wall}_v"], dtype=np.float32)
                    faces = np.asarray(data[f"{kind}_{wall}_f"], dtype=np.int64)
                    if kind == "model":
                        vertices = _snap_mesh_to_contours(
                            vertices, contours, tissue, surface=wall)

                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                                           process=False)
                    if not mesh.is_watertight:
                        raise RuntimeError(f"{phase} {kind} {wall} is not watertight")

                    contour = contours[np.abs(tissue - label) < 0.5]
                    _, distances, _ = trimesh.proximity.closest_point(mesh, contour)
                    key = (phase, kind, wall)
                    surfaces[key] = _polydata(vertices, faces)
                    diagnostics[key] = (
                        float(np.mean(distances)),
                        float(np.percentile(distances, 95)),
                    )
    return surfaces, diagnostics


def panel(plotter, row: int, col: int, endo, epi, title: str, frame) -> None:
    plotter.subplot(row, col)
    plotter.add_text(title, font_size=11, position="upper_edge", color="black")
    plotter.add_mesh(frame, opacity=0.0, show_scalar_bar=False)
    plotter.add_mesh(epi, color=EPI_COLOUR, opacity=0.28, smooth_shading=True,
                     specular=0.2)
    plotter.add_mesh(endo, color=ENDO_COLOUR, opacity=1.0, smooth_shading=True,
                     specular=0.3)


def main() -> None:
    surfaces, diagnostics = load_surfaces()

    bounds = np.array([surface.bounds for surface in surfaces.values()])
    focus = (bounds[:, 0].min(), bounds[:, 1].max(),
             bounds[:, 2].min(), bounds[:, 3].max(),
             bounds[:, 4].min(), bounds[:, 5].max())
    frame = pv.Box(bounds=focus)

    plotter = pv.Plotter(shape=(2, 2), window_size=(1500, 1250),
                         off_screen=True, border=False)
    plotter.set_background("white")
    layout = [
        (0, 0, "ED", "model", "(a) Reconstruction, end-diastole"),
        (0, 1, "ED", "voxel", "(b) Segmentation-derived, end-diastole"),
        (1, 0, "ES", "model", "(c) Reconstruction, end-systole"),
        (1, 1, "ES", "voxel", "(d) Segmentation-derived, end-systole"),
    ]
    for row, col, phase, kind, title in layout:
        panel(plotter, row, col,
              surfaces[(phase, kind, "endo")],
              surfaces[(phase, kind, "epi")], title, frame)
        plotter.view_vector((1.6, -1.6, 0.0), viewup=(0.0, 0.0, 1.0))
        plotter.reset_camera()
        plotter.camera.zoom(0.9)

    plotter.link_views()
    plotter.screenshot(str(OUT), transparent_background=False)
    plotter.close()

    print(f"wrote {OUT.relative_to(THESIS)} from {PATIENT}")
    for key, surface in surfaces.items():
        mean, p95 = diagnostics[key]
        print(f"  {key}: watertight={surface.is_manifold}, "
              f"contour distance={mean:.3f} mm mean/{p95:.3f} mm p95")


if __name__ == "__main__":
    main()
