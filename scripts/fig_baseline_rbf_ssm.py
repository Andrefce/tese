"""Exploratory reconstruction baselines: RBF implicit fitting and SSM fitting.

Both baselines consume the *same* SAX contour rings as the model and are pushed
through the *same* watertight-repair and metric pipeline used for the reported
results, so the numbers are directly comparable with the contour-lofting row.

``rbf``  Thin-plate-spline implicit surface fitted to the contour points plus
         off-surface constraints along in-plane contour normals.
``ssm``  UK Digital Heart Project left-ventricular shape model (mean + 100 PCA
         modes, separate ED and ES models, each carrying an endocardial and an
         epicardial sheet) fitted to the contour points by alternating
         similarity registration and regularised mode-coefficient estimation.

Outputs
    images/results_baseline_rbf_ssm.png
    scripts/eval_demo/outputs/baseline_rbf_ssm_metrics.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

THESIS = Path(__file__).resolve().parents[1]
EVAL_DIR = THESIS / "scripts" / "eval_demo"
DEMO_OUT = EVAL_DIR / "outputs"
IMAGES = THESIS / "images"
SSM_REPO_URL = "https://github.com/UK-Digital-Heart-Project/Statistical-Shape-Model.git"
SSM_DIR = Path("/tmp/cardiosdf-ssm")

sys.path.insert(0, str(EVAL_DIR))

from geometry import (  # noqa: E402
    _clean_inside,
    isotropic_grid,
    make_watertight,
    marching_cubes_mesh,
    repair_if_invalid,
    signed_distance_from_mask,
)
from recon_metrics import reconstruction_quality  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Contour helpers
# ──────────────────────────────────────────────────────────────────────────
def rings_of(points: np.ndarray) -> list[np.ndarray]:
    """Split a contour point cloud into its per-slice rings, apex-last."""
    return [points[np.isclose(points[:, 2], z)] for z in np.unique(points[:, 2])]


def ring_normals(ring: np.ndarray) -> np.ndarray:
    """In-plane outward normals from the ring tangent, disambiguated radially."""
    tangent = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0], np.zeros(len(ring))])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    radial = ring - ring.mean(axis=0)
    radial[:, 2] = 0.0
    flip = np.sign(np.sum(normal * radial, axis=1))
    flip[flip == 0] = 1.0
    return normal * flip[:, None]


def surface_points(contours: np.ndarray, tissue: np.ndarray,
                   surface: str) -> np.ndarray:
    label = 0.0 if surface == "endo" else 1.0
    return np.asarray(contours[np.abs(tissue - label) < 0.5], dtype=np.float64)


def _evaluate_rbf(interpolator: RBFInterpolator, points: np.ndarray,
                  chunk: int = 8192) -> np.ndarray:
    """Evaluate a fitted thin-plate-spline interpolator through BLAS.

    SciPy walks the kernel point by point, which dominates the runtime of the
    whole baseline. Forming the squared-distance matrix with a matrix product
    computes the same values an order of magnitude faster. The fast path is
    accepted only after it reproduces SciPy on a random sample; otherwise the
    SciPy evaluation is used unchanged.
    """
    def scipy_path() -> np.ndarray:
        values = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), 50_000):
            stop = start + 50_000
            values[start:stop] = interpolator(points[start:stop])
        return values

    try:
        if interpolator.kernel != "thin_plate_spline" or interpolator.neighbors:
            return scipy_path()
        centres = np.asarray(interpolator.y, dtype=np.float64)
        coefficients = np.asarray(interpolator._coeffs, dtype=np.float64)
        shift = np.asarray(interpolator._shift, dtype=np.float64)
        scale = np.asarray(interpolator._scale, dtype=np.float64)
        powers = np.asarray(interpolator.powers, dtype=np.int64)
        epsilon = float(interpolator.epsilon)
    except AttributeError:
        return scipy_path()

    scaled_centres = centres * epsilon
    centre_norms = np.einsum("ij,ij->i", scaled_centres, scaled_centres)
    kernel_weights = coefficients[:len(centres)]
    polynomial_weights = coefficients[len(centres):]

    def block(query: np.ndarray) -> np.ndarray:
        scaled = query * epsilon
        squared = (np.einsum("ij,ij->i", scaled, scaled)[:, None]
                   + centre_norms[None, :]
                   - 2.0 * (scaled @ scaled_centres.T))
        # The floor keeps log finite; at that magnitude r^2 log r underflows to 0,
        # which is the defined value of the kernel at coincident points.
        np.maximum(squared, 1e-300, out=squared)
        logarithm = np.log(squared)
        squared *= logarithm
        squared *= 0.5  # r^2 log r
        monomials = (query - shift) / scale
        polynomial = np.prod(monomials[:, None, :] ** powers[None, :, :], axis=2)
        return (squared @ kernel_weights + polynomial @ polynomial_weights)[:, 0]

    sample = points[::max(1, len(points) // 2048)]
    if not np.allclose(block(sample), interpolator(sample), rtol=1e-8, atol=1e-8):
        return scipy_path()

    values = np.empty(len(points), dtype=np.float64)

    def fill(span: tuple[int, int]) -> None:
        values[slice(*span)] = block(points[span[0]:span[1]])

    # Each block materialises a (chunk x centres) matrix, so the peak scales with
    # chunk size times the worker count. Halving both on MemoryError keeps the
    # values identical -- blocks are row-independent -- and only trades speed.
    workers = min(4, os.cpu_count() or 1)
    while True:
        blocks = [(start, min(start + chunk, len(points)))
                  for start in range(0, len(points), chunk)]
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # NumPy releases the GIL inside these element-wise kernels. The
                # results must be consumed: ThreadPoolExecutor.map is lazy, so an
                # exception in a block would otherwise be discarded and leave that
                # slice of `values` holding uninitialised np.empty data.
                for _ in pool.map(fill, blocks):
                    pass
            return values
        except MemoryError:
            if chunk <= 256 and workers == 1:
                return scipy_path()
            workers = max(1, workers // 2)
            chunk = max(256, chunk // 2)


# ──────────────────────────────────────────────────────────────────────────
# Baseline 1 — RBF implicit surface
# ──────────────────────────────────────────────────────────────────────────
def build_rbf_geometry(contours: np.ndarray, tissue: np.ndarray,
                       offset_mm: float = 2.5, pitch: float = 1.0,
                       smoothing: float = 0.5) -> dict:
    """Thin-plate-spline implicit reconstruction from the SAX contour rings."""
    meshes: dict[str, trimesh.Trimesh] = {}
    reports: list[dict] = []

    for surface in ("endo", "epi"):
        points = surface_points(contours, tissue, surface)
        rings = [ring for ring in rings_of(points) if len(ring) >= 3]
        if len(rings) < 2:
            raise ValueError(f"RBF fitting needs at least two {surface} rings.")

        on = np.vstack(rings)
        normals = np.vstack([ring_normals(ring) for ring in rings])
        centres = np.vstack([on, on + offset_mm * normals, on - offset_mm * normals])
        values = np.concatenate([
            np.zeros(len(on)),
            np.full(len(on), offset_mm),
            np.full(len(on), -offset_mm),
        ])

        interpolator = RBFInterpolator(
            centres, values, kernel="thin_plate_spline", degree=1,
            smoothing=smoothing)

        # The grid is padded generously in-plane but barely along the long axis,
        # so the surface is capped just past the outermost observed rings —
        # the same support the lofting baseline is given.
        lo = on.min(axis=0) - np.array([8.0, 8.0, 1.5])
        hi = on.max(axis=0) + np.array([8.0, 8.0, 1.5])
        shape = tuple(int(np.ceil((hi[d] - lo[d]) / pitch)) + 1 for d in range(3))
        axes = [lo[d] + np.arange(shape[d]) * pitch for d in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        field = _evaluate_rbf(interpolator, grid)
        field = field.reshape(shape)

        inside = _clean_inside(field <= 0.0)
        clean = signed_distance_from_mask(inside, np.full(3, pitch), smooth_sigma=0.6)
        raw = marching_cubes_mesh(clean, lo, np.full(3, pitch))
        mesh, report = repair_if_invalid(raw, f"rbf-{surface}")
        meshes[surface] = mesh
        reports.append(report)

    return {**meshes, "reports": reports, "source": "RBF implicit surface"}


# ──────────────────────────────────────────────────────────────────────────
# Baseline 2 — statistical shape model fitting
# ──────────────────────────────────────────────────────────────────────────
def ensure_ssm_dir() -> Path:
    if (SSM_DIR / "LV_ED_mean.vtk").exists():
        return SSM_DIR
    subprocess.run(["git", "clone", "--depth", "1", SSM_REPO_URL, str(SSM_DIR)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return SSM_DIR


def read_vtk_polydata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    tokens = path.read_text().split()
    index = tokens.index("POINTS")
    count = int(tokens[index + 1])
    start = index + 3
    vertices = np.asarray(tokens[start:start + 3 * count], dtype=float).reshape(count, 3)

    index = tokens.index("POLYGONS")
    polygons = int(tokens[index + 1])
    cursor = index + 3
    faces: list[list[int]] = []
    for _ in range(polygons):
        size = int(tokens[cursor])
        ids = [int(value) for value in tokens[cursor + 1:cursor + 1 + size]]
        if size == 3:
            faces.append(ids)
        elif size > 3:
            faces.extend([ids[0], ids[k], ids[k + 1]] for k in range(1, size - 1))
        cursor += size + 1
    return vertices, np.asarray(faces, dtype=np.int64)


def _load_csv_matrix(path: Path) -> np.ndarray:
    """Read a numeric csv.gz into an array, caching the parse as .npy.

    ``np.genfromtxt`` materialises every field as a Python object before
    building the array, which needs gigabytes for the 66129x100 mode matrix and
    raises MemoryError on a small machine. ``pandas.read_csv`` streams the same
    values into a float64 buffer, and the .npy sidecar makes later loads a
    single mmap-able read instead of re-parsing 31 MB of gzip.
    """
    cache = path.with_suffix(path.suffix + ".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)

    import pandas as pd

    values = pd.read_csv(path, header=None, dtype=np.float64).to_numpy()
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    try:
        np.save(cache, values)
    except OSError:                              # a read-only cache is not fatal
        pass
    return values


def load_shape_model(phase: str, n_modes: int = 25) -> dict:
    """Mean LV shape, PCA basis, and the endo/epi vertex split of the sheets."""
    ssm = ensure_ssm_dir()
    tag = "ED" if phase.upper() == "ED" else "ES"
    mean, faces = read_vtk_polydata(ssm / f"LV_{tag}_mean.vtk")
    modes = _load_csv_matrix(ssm / f"LV_{tag}_pc_100_modes.csv.gz")[:, :n_modes]
    variance = _load_csv_matrix(ssm / f"LV_{tag}_var_100_modes.csv.gz")[:n_modes]
    basis = modes.reshape(len(mean), 3, n_modes) * np.sqrt(variance)  # b in std units

    template = trimesh.Trimesh(mean, faces, process=False)
    components = template.split(only_watertight=False)
    if len(components) != 2:
        raise RuntimeError(f"Expected an endo and an epi sheet, got {len(components)}.")

    axis = principal_axis(mean)
    labels = np.full(len(mean), -1, dtype=np.int64)
    radii = []
    for order, component in enumerate(components):
        offsets = component.vertices - mean.mean(axis=0)
        radial = offsets - np.outer(offsets @ axis, axis)
        radii.append(float(np.linalg.norm(radial, axis=1).mean()))
        labels[cKDTree(mean).query(component.vertices)[1]] = order
    epi_order = int(np.argmax(radii))

    return {
        "mean": mean,
        "basis": basis,
        "faces": faces,
        "sheets": {
            "endo": components[1 - epi_order],
            "epi": components[epi_order],
        },
        "vertex_surface": np.where(labels == epi_order, 1, 0),
    }


def principal_axis(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    return right[0] / np.linalg.norm(right[0])


def apex_direction(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Orient ``axis`` so that it points from the base towards the apex.

    The apex end of a ventricular point cloud is the end where the cross-section
    shrinks, so the two halves are compared by their mean radial spread.
    """
    projection = (points - points.mean(axis=0)) @ axis
    offsets = points - points.mean(axis=0)
    radial = np.linalg.norm(offsets - np.outer(projection, axis), axis=1)
    low = radial[projection < np.percentile(projection, 25)].mean()
    high = radial[projection > np.percentile(projection, 75)].mean()
    return axis if high < low else -axis


def similarity_from_pairs(source: np.ndarray, target: np.ndarray) -> tuple:
    """Umeyama similarity (scale, rotation, translation) mapping source→target."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    a, b = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd((b.T @ a) / len(a))
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular) @ correction) /
                  max(np.mean(np.sum(a ** 2, axis=1)), 1e-12))
    return scale, rotation, target_mean - scale * rotation @ source_mean


def stack_axis(points: np.ndarray) -> np.ndarray:
    """Long axis of a SAX stack: the line through the per-slice ring centroids.

    Taking the principal axis of the whole contour cloud fails on short stacks,
    where the in-plane extent exceeds the axial one and the fit ends up lying on
    its side.
    """
    heights = np.unique(points[:, 2])
    if len(heights) < 3:
        return principal_axis(points)
    centroids = np.array([points[points[:, 2] == height].mean(axis=0)
                          for height in heights])
    return principal_axis(centroids)


def initial_alignment(mean: np.ndarray, targets: dict) -> tuple:
    """Long-axis + scale + best yaw alignment of the mean shape to the contours."""
    contour_points = np.vstack(list(targets.values()))
    source_axis = apex_direction(mean, principal_axis(mean))
    target_axis = apex_direction(contour_points, stack_axis(contour_points))

    v = np.cross(source_axis, target_axis)
    c = float(np.dot(source_axis, target_axis))
    if np.linalg.norm(v) < 1e-8:
        rotation = np.eye(3) if c > 0 else -np.eye(3)
    else:
        skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + c))

    def spread(points: np.ndarray, axis: np.ndarray) -> float:
        offsets = points - points.mean(axis=0)
        return float(np.linalg.norm(offsets - np.outer(offsets @ axis, axis),
                                    axis=1).mean())

    scale = spread(contour_points, target_axis) / spread(mean, source_axis)
    translation = contour_points.mean(axis=0) - scale * rotation @ mean.mean(axis=0)

    tree = cKDTree(contour_points)
    probe = mean[::10]
    best = None
    for angle in np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False):
        cos, sin = np.cos(angle), np.sin(angle)
        k = np.array([[0, -target_axis[2], target_axis[1]],
                      [target_axis[2], 0, -target_axis[0]],
                      [-target_axis[1], target_axis[0], 0]])
        yaw = np.eye(3) + sin * k + (1 - cos) * (k @ k)
        candidate = yaw @ rotation
        offset = contour_points.mean(axis=0) - scale * candidate @ mean.mean(axis=0)
        moved = probe @ (scale * candidate).T + offset
        cost = float(tree.query(moved, workers=-1)[0].mean())
        if best is None or cost < best[0]:
            best = (cost, scale, candidate, offset)
    return best[1], best[2], best[3]


def fit_shape_model(model: dict, contours: np.ndarray, tissue: np.ndarray,
                    iterations: int = 12, regularisation: float = 12.0) -> dict:
    """Alternate correspondence, similarity registration and mode estimation."""
    mean, basis = model["mean"], model["basis"]
    n_modes = basis.shape[2]
    targets = {surface: surface_points(contours, tissue, surface)
               for surface in ("endo", "epi")}
    trees = {key: cKDTree(value) for key, value in targets.items()}
    is_epi = model["vertex_surface"] == 1
    masks = {"endo": ~is_epi, "epi": is_epi}

    scale, rotation, translation = initial_alignment(mean, targets)
    coefficients = np.zeros(n_modes)

    for _ in range(iterations):
        shape = mean + basis @ coefficients
        moved = shape @ (scale * rotation).T + translation

        pairs_source, pairs_target = [], []
        for surface, mask in masks.items():
            distance, index = trees[surface].query(moved[mask], workers=-1)
            keep = distance <= np.percentile(distance, 60)  # observed rings only
            pairs_source.append(np.flatnonzero(mask)[keep])
            pairs_target.append(targets[surface][index[keep]])
        selected = np.concatenate(pairs_source)
        matched = np.vstack(pairs_target)

        scale, rotation, translation = similarity_from_pairs(shape[selected], matched)

        transform = scale * rotation
        design = np.einsum("ij,njk->nik", transform, basis[selected]).reshape(-1, n_modes)
        residual = (matched - shape[selected] @ transform.T - translation
                    + (basis[selected] @ coefficients) @ transform.T).reshape(-1)
        gram = design.T @ design + regularisation * np.eye(n_modes)
        coefficients = np.linalg.solve(gram, design.T @ residual)
        coefficients = np.clip(coefficients, -3.0, 3.0)  # b is in std units

    shape = mean + basis @ coefficients
    return {
        "vertices": shape @ (scale * rotation).T + translation,
        "coefficients": coefficients,
        "scale": scale,
    }


def build_ssm_geometry(contours: np.ndarray, tissue: np.ndarray, phase: str,
                       n_modes: int = 25, taubin_iters: int = 12) -> dict:
    model = load_shape_model(phase, n_modes)
    fit = fit_shape_model(model, contours, tissue)

    meshes: dict[str, trimesh.Trimesh] = {}
    reports: list[dict] = []
    is_epi = model["vertex_surface"] == 1
    for surface, mask in (("endo", ~is_epi), ("epi", is_epi)):
        index = np.full(len(fit["vertices"]), -1, dtype=np.int64)
        index[mask] = np.arange(int(mask.sum()))
        faces = model["faces"][mask[model["faces"]].all(axis=1)]
        sheet = trimesh.Trimesh(fit["vertices"][mask], index[faces], process=False)
        mesh, report = make_watertight(close_sheet(sheet), f"ssm-{surface}", taubin_iters)
        meshes[surface] = mesh
        reports.append(report)

    return {**meshes, "reports": reports, "coefficients": fit["coefficients"],
            "source": f"UK Biobank LV shape model ({phase}, {n_modes} modes)"}


def close_sheet(sheet: trimesh.Trimesh, pitch: float = 1.0) -> trimesh.Trimesh:
    """Cap the open basal rim of an SSM sheet by re-meshing its filled interior."""
    capped = sheet.copy()
    trimesh.repair.fill_holes(capped)
    trimesh.repair.fix_normals(capped)
    if capped.is_watertight:
        return capped

    from geometry import voxelise_surface  # local import keeps the module import light

    origin, shape = isotropic_grid([sheet], pitch, pad_mm=4.0)
    inside = _clean_inside(voxelise_surface(capped, origin, pitch, shape))
    field = signed_distance_from_mask(inside, np.full(3, pitch), smooth_sigma=0.6)
    return marching_cubes_mesh(field, origin, np.full(3, pitch))


# ──────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────
ENDO_COLOUR = "#c1121f"
EPI_COLOUR = "#adb5bd"


def render(panels: list[tuple[str, dict]], contours: np.ndarray,
           tissue: np.ndarray, out: Path) -> None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.global_theme.font.family = "times"

    def polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
        return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float32), cells)

    bounds = np.array([geometry[surface].bounds
                       for _, geometry in panels for surface in ("endo", "epi")])
    frame = pv.Box(bounds=(bounds[:, 0, 0].min(), bounds[:, 1, 0].max(),
                           bounds[:, 0, 1].min(), bounds[:, 1, 1].max(),
                           bounds[:, 0, 2].min(), bounds[:, 1, 2].max()))

    plotter = pv.Plotter(shape=(1, len(panels) + 1),
                         window_size=(340 * (len(panels) + 1), 400),
                         off_screen=True, border=False)
    plotter.set_background("white")

    def frame_view() -> None:
        # viewup -z renders the base upwards, as in the Results figure.
        plotter.view_vector((1.6, -1.6, 0.0), viewup=(0.0, 0.0, -1.0))
        plotter.reset_camera()
        plotter.camera.zoom(1.15)

    plotter.subplot(0, 0)
    plotter.add_text("(a) Input SAX contours", font_size=11, position="upper_edge",
                     color="black")
    plotter.add_mesh(frame, opacity=0.0, show_scalar_bar=False)
    for label, colour in ((0.0, ENDO_COLOUR), (1.0, EPI_COLOUR)):
        cloud = pv.PolyData(contours[np.abs(tissue - label) < 0.5].astype(np.float32))
        plotter.add_mesh(cloud, color=colour, point_size=4, render_points_as_spheres=True)
    frame_view()

    for column, (title, geometry) in enumerate(panels, start=1):
        plotter.subplot(0, column)
        plotter.add_text(title, font_size=11, position="upper_edge", color="black")
        plotter.add_mesh(frame, opacity=0.0, show_scalar_bar=False)
        plotter.add_mesh(polydata(geometry["epi"]), color=EPI_COLOUR, opacity=0.28,
                         smooth_shading=True, specular=0.2)
        plotter.add_mesh(polydata(geometry["endo"]), color=ENDO_COLOUR, opacity=1.0,
                         smooth_shading=True, specular=0.3)
        frame_view()

    plotter.link_views()
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()


# ──────────────────────────────────────────────────────────────────────────
def load_demo(patient: str, phase: str) -> tuple:
    """Cached demo meshes; the model is snapped to the rings as in the Results figure."""
    sys.path.insert(0, str(THESIS / "scripts" / "webapp"))
    from core.sdf_model import _snap_mesh_to_contours

    with np.load(DEMO_OUT / f"demo_{patient}_{phase}.npz") as data:
        contours = np.asarray(data["contours_xyz_mm"], dtype=np.float64)
        tissue = np.asarray(data["contours_tissue"], dtype=np.float64)
        cached: dict[str, dict[str, trimesh.Trimesh]] = {}
        for kind in ("model", "voxel"):
            cached[kind] = {}
            for surface in ("endo", "epi"):
                vertices = np.asarray(data[f"{kind}_{surface}_v"], dtype=np.float32)
                if kind == "model":
                    vertices = _snap_mesh_to_contours(
                        vertices, contours.astype(np.float32),
                        tissue.astype(np.float32), surface=surface)
                cached[kind][surface] = trimesh.Trimesh(
                    vertices, data[f"{kind}_{surface}_f"], process=False)
    return contours, tissue, cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="patient002")
    parser.add_argument("--phase", default="ED", choices=("ED", "ES"))
    parser.add_argument("--modes", type=int, default=25)
    args = parser.parse_args()

    contours, tissue, cached = load_demo(args.patient, args.phase)

    print("fitting RBF implicit surface ...")
    rbf = build_rbf_geometry(contours, tissue)
    print("fitting statistical shape model ...")
    ssm = build_ssm_geometry(contours, tissue, args.phase, args.modes)
    print("  mode coefficients (std units):",
          np.array2string(ssm["coefficients"][:8], precision=2))

    reference = cached["voxel"]
    rows = []
    for name, geometry in (("model", cached["model"]), ("rbf", rbf), ("ssm", ssm)):
        metrics = reconstruction_quality(geometry, reference, pitch=1.0)
        rows.append({"method": name, **metrics})

    keys = ("endo_chamfer_mm", "epi_chamfer_mm", "endo_hd95_mm", "epi_hd95_mm",
            "endo_dice", "myo_dice", "vol_ratio_endo", "vol_ratio_epi")
    header = f"{'method':<8}" + "".join(f"{key:>18}" for key in keys)
    print(f"\n{args.patient} {args.phase} versus segmentation-derived reference")
    print(header)
    for row in rows:
        print(f"{row['method']:<8}" + "".join(f"{row[key]:>18.3f}" for key in keys))

    import pandas as pd

    csv = DEMO_OUT / "baseline_rbf_ssm_metrics.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    out = IMAGES / "results_baseline_rbf_ssm.png"
    render([("(b) RBF implicit", rbf),
            ("(c) SSM fit", ssm),
            ("(d) Proposed model", cached["model"]),
            ("(e) Segmentation-derived", reference)],
           contours, tissue, out)
    print(f"\nwrote {out.relative_to(THESIS)}")
    print(f"wrote {csv.relative_to(THESIS)}")


if __name__ == "__main__":
    main()
