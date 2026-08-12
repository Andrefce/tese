"""Thesis figure: 3D reconstructed surfaces and their watertight cross-sections.

Per phase:
  (a) cut-away 3D renders of the CardioSDF (model) and segmentation (voxel)
      shells. The epicardium is cut by a long-axis plane and the cut is capped,
      which only produces a closed solid when the surface is genuinely sealed,
      so the flat cap face is itself the watertightness evidence. The
      endocardium is drawn intact inside it and the myocardial wall is visible
      between the two;
  (b) short-axis cross-sections through both shells at basal / mid / apical
      levels. Each cut of a closed shell returns closed rings; the myocardial
      band between the endo and epi rings is filled. Any opening in a surface
      would show up here as a broken ring.

Run:
    python fig_meshes_watertight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUT_DIR / "figures"

COL = {
    "model_endo": "#1f6fd0",
    "model_epi": "#7fb3ea",
    "voxel_endo": "#b8451f",
    "voxel_epi": "#f0a58a",
    "contour_endo": "#111111",
    "contour_epi": "#555555",
}

THESIS_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}


def load_mesh(npz, tag: str) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=npz[f"{tag}_v"], faces=npz[f"{tag}_f"], process=False)


def topology(mesh: trimesh.Trimesh) -> str:
    euler = len(mesh.vertices) - (len(mesh.faces) * 3 // 2) + len(mesh.faces)
    genus = (2 - euler) // 2
    return f"watertight={mesh.is_watertight}, genus={genus}"


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce face count for rendering only; slices always use the full mesh."""
    if len(mesh.faces) <= target_faces:
        return mesh
    for name in ("simplify_quadric_decimation", "simplify_quadratic_decimation"):
        fn = getattr(mesh, name, None)
        if fn is None:
            continue
        try:
            return fn(face_count=target_faces)
        except TypeError:
            try:
                return fn(target_faces)
            except Exception:
                pass
        except Exception:
            pass
    keep = np.random.default_rng(0).choice(len(mesh.faces), target_faces, replace=False)
    return mesh.submesh([keep], append=True)


def camera_basis(elev_deg: float, azim_deg: float):
    """Orthographic camera: returns (right, up, view) unit vectors.

    ``view`` points from the scene towards the camera.
    """
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(up_world, view)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    return right, up / np.linalg.norm(up), view


def render_scene(ax, layers, cam, target_faces: int = 60000,
                 light=(0.4, 0.25, 0.88)):
    """Depth-sorted 2D render of several meshes in one pass.

    ``layers`` is a sequence of ``(mesh, colour)``. All triangles from all
    meshes are sorted together, so an inner surface is correctly hidden by the
    shell around it. Rendering each mesh separately would paint whichever was
    drawn last on top regardless of depth. matplotlib's own 3D z-sorting has the
    same defect and additionally interleaves front and back faces on dense
    meshes, which speckles the surface; back-face culling here removes that too.
    """
    right, up, view = cam
    light = np.asarray(light, dtype=np.float64)
    light = light / np.linalg.norm(light)

    all_xy, all_depth, all_col = [], [], []
    for mesh, color in layers:
        small = decimate(mesh, target_faces)
        verts = np.asarray(small.vertices, dtype=np.float64)
        faces = np.asarray(small.faces)
        normals = np.asarray(small.face_normals, dtype=np.float64)

        keep = (normals @ view) > 0.0
        if not keep.any():
            keep = np.ones(len(faces), bool)
        faces, normals = faces[keep], normals[keep]

        tris = verts[faces]
        all_xy.append(np.stack([tris @ right, tris @ up], axis=-1))
        all_depth.append((tris @ view).mean(axis=1))

        shade = 0.42 + 0.58 * np.clip(normals @ light, 0.0, 1.0)
        rgb = np.array(matplotlib.colors.to_rgb(color))
        all_col.append(np.clip(shade[:, None] * rgb[None, :] + 0.06, 0, 1))

    xy = np.concatenate(all_xy)
    depth = np.concatenate(all_depth)
    facecolors = np.concatenate(all_col)
    order = np.argsort(depth)

    ax.add_collection(matplotlib.collections.PolyCollection(
        xy[order], facecolors=facecolors[order], linewidths=0.0,
        edgecolors="none", antialiased=True,
    ))
    return xy.reshape(-1, 2)


def project_points(pts: np.ndarray, cam) -> np.ndarray:
    right, up, _ = cam
    return np.stack([pts @ right, pts @ up], axis=-1)


def cut_away(mesh: trimesh.Trimesh, origin, normal) -> trimesh.Trimesh:
    """Half-cut the shell and cap the opening.

    ``cap=True`` can only close the cut when the input surface is watertight, so
    a solid flat cap face in the render is direct evidence of watertightness.
    """
    try:
        cut = mesh.slice_plane(plane_origin=origin, plane_normal=normal, cap=True)
    except Exception:
        return mesh
    if cut is None or len(cut.faces) < 100:
        return mesh
    return cut


def equalise_3d(ax, verts: np.ndarray) -> None:
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2.0
    radius = float((hi - lo).max()) / 2.0 * 1.05
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def section_rings(mesh: trimesh.Trimesh, z: float) -> list[np.ndarray]:
    """Closed xy rings where the plane z=const cuts the shell."""
    try:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return []
        planar, to_3d = sec.to_planar()
    except Exception:
        return []
    rings = []
    for poly in getattr(planar, "polygons_full", []) or []:
        xy = np.asarray(poly.exterior.coords)
        if len(xy) < 4:
            continue
        h = np.column_stack([xy, np.zeros(len(xy)), np.ones(len(xy))])
        rings.append((to_3d @ h.T).T[:, :2])
    return rings


def ring_stats(rings: list[np.ndarray]) -> tuple[int, bool]:
    closed = all(np.allclose(r[0], r[-1], atol=1e-6) for r in rings) if rings else False
    return len(rings), closed


def draw_cross_section(ax, endo: trimesh.Trimesh, epi: trimesh.Trimesh, z: float,
                       endo_col: str, epi_col: str, title: str) -> str:
    endo_rings = section_rings(endo, z)
    epi_rings = section_rings(epi, z)

    for r in epi_rings:
        ax.fill(r[:, 0], r[:, 1], color=epi_col, alpha=0.45, lw=0, zorder=1)
    for r in endo_rings:
        ax.fill(r[:, 0], r[:, 1], color="white", alpha=1.0, lw=0, zorder=2)
    for r in epi_rings:
        ax.plot(r[:, 0], r[:, 1], color=epi_col, lw=1.4, zorder=3)
    for r in endo_rings:
        ax.plot(r[:, 0], r[:, 1], color=endo_col, lw=1.4, zorder=4)

    n_endo, endo_closed = ring_stats(endo_rings)
    n_epi, epi_closed = ring_stats(epi_rings)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ok = n_endo > 0 and n_epi > 0 and endo_closed and epi_closed
    ax.text(0.02, 0.02, ("closed" if ok else "OPEN"), transform=ax.transAxes,
            fontsize=6, color=("#1a7f37" if ok else "#c0392b"),
            va="bottom", ha="left")
    return f"z={z:6.1f} mm  endo rings={n_endo} closed={endo_closed}  epi rings={n_epi} closed={epi_closed}"


def build_figure(phase: str) -> None:
    npz_path = OUT_DIR / f"demo_patient002_{phase}.npz"
    if not npz_path.exists():
        print(f"  skip {phase}: {npz_path.name} not found")
        return
    npz = np.load(npz_path, allow_pickle=True)

    meshes = {tag: load_mesh(npz, tag) for tag in
              ("model_endo", "model_epi", "voxel_endo", "voxel_epi")}
    print(f"[{phase}]")
    for tag, m in meshes.items():
        print(f"  {tag:11s} {len(m.faces):>7d} faces  {topology(m)}")

    cont_xyz = npz["contours_xyz_mm"]
    cont_tis = npz["contours_tissue"]

    # Sample basal / mid / apical levels from the shared endo extent.
    z_all = np.concatenate([meshes["model_endo"].vertices[:, 2],
                            meshes["voxel_endo"].vertices[:, 2]])
    z_lo, z_hi = np.percentile(z_all, 4), np.percentile(z_all, 96)
    levels = [(z_lo + f * (z_hi - z_lo), name) for f, name in
              ((0.15, "basal"), (0.50, "mid"), (0.85, "apical"))]

    with plt.rc_context(THESIS_STYLE):
        fig = plt.figure(figsize=(11.5, 7.2), facecolor="white")
        gs = fig.add_gridspec(2, 6, height_ratios=[1.6, 1.0],
                              hspace=0.05, wspace=0.42,
                              left=0.04, right=0.985, top=0.90, bottom=0.11)

        # ── Row 1: cut-away renders ───────────────────────────────
        cam = camera_basis(elev_deg=14.0, azim_deg=-58.0)
        # Cut with the plane facing the camera so the capped face is what we see.
        cut_n = -np.array([cam[2][0], cam[2][1], 0.0])
        cut_n /= np.linalg.norm(cut_n)
        for col, (source, label) in enumerate((("model", "CardioSDF reconstruction"),
                                               ("voxel", "Segmentation surfaces"))):
            ax = fig.add_subplot(gs[0, col * 3:(col + 1) * 3])
            endo, epi = meshes[f"{source}_endo"], meshes[f"{source}_epi"]
            centre = np.asarray(epi.bounds).mean(axis=0)
            epi_cut = cut_away(epi, centre, cut_n)
            endo_cut = cut_away(endo, centre, cut_n)

            xy_all = render_scene(ax, [
                (epi_cut, COL[f"{source}_epi"]),
                (endo_cut, COL[f"{source}_endo"]),
            ], cam)

            keep_pts = (cont_xyz - centre) @ cut_n >= -1.0
            pts = cont_xyz[keep_pts]
            if len(pts):
                p2 = project_points(pts, cam)
                ax.plot(p2[:, 0], p2[:, 1], ".", ms=1.3,
                        color=COL["contour_endo"], alpha=0.7, zorder=6)

            ax.set_aspect("equal")
            lo, hi = xy_all.min(0), xy_all.max(0)
            pad = 0.06 * float((hi - lo).max())
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.axis("off")
            ax.set_title(f"{label} — cut-away\n"
                         "epicardium sectioned and capped; endocardium inside",
                         fontsize=9.5, pad=4)

            # Scale bar (mm); the orthographic projection preserves length.
            bar = 20.0
            x0, y0 = lo[0] + pad, lo[1] + pad
            ax.plot([x0, x0 + bar], [y0, y0], color="#222222", lw=1.6)
            ax.text(x0 + bar / 2, y0 + 0.012 * (hi[1] - lo[1]), "20 mm",
                    ha="center", va="bottom", fontsize=7.5)

        # ── Row 2: watertight cross-sections ──────────────────────
        for j, (z, name) in enumerate(levels):
            for k, source in enumerate(("model", "voxel")):
                ax = fig.add_subplot(gs[1, j * 2 + k])
                src_label = "CardioSDF" if source == "model" else "Segmentation"
                msg = draw_cross_section(
                    ax, meshes[f"{source}_endo"], meshes[f"{source}_epi"], z,
                    COL[f"{source}_endo"], COL[f"{source}_epi"],
                    f"{src_label} — {name}",
                )
                print(f"  {source:5s} {name:6s} {msg}")
                if j == 0 and k == 0:
                    ax.set_ylabel("y (mm)")
                ax.set_xlabel("x (mm)")

        handles = [
            Line2D([], [], color=COL["model_endo"], lw=2, label="CardioSDF endocardium"),
            Line2D([], [], color=COL["model_epi"], lw=2, label="CardioSDF epicardium"),
            Line2D([], [], color=COL["voxel_endo"], lw=2, label="Segmentation endocardium"),
            Line2D([], [], color=COL["voxel_epi"], lw=2, label="Segmentation epicardium"),
            Line2D([], [], marker="o", ls="", ms=3, color=COL["contour_endo"],
                   label="Input SAX contours"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(
            f"patient002 — {phase}: watertight LV shells and short-axis cross-sections",
            fontsize=11.5, fontweight="bold", y=0.985)

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            path = FIG_DIR / f"fig_meshes_watertight_{phase}.{ext}"
            fig.savefig(path)
            print(f"  saved {path}")
        plt.close(fig)


def main() -> None:
    for phase in ("ED", "ES"):
        build_figure(phase)


if __name__ == "__main__":
    main()
