import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib import colors, patches
from matplotlib import patheffects
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "notebooks" / "patient002"
OUT_DIR = ROOT / "images"
SSM_REPO_URL = "https://github.com/UK-Digital-Heart-Project/Statistical-Shape-Model.git"
SSM_FILE = "LV_ED_mean.vtk"

ED_MRI = CASE_DIR / "patient002_frame01.nii" / "DCM04Gate1.nii"
ED_SEG = CASE_DIR / "patient002_frame01_gt.nii" / "DCM04-OH-AL_V2_1.nii"
ES_MRI = CASE_DIR / "patient002_frame12.nii" / "DCM04Gate12.nii"
ES_SEG = CASE_DIR / "patient002_frame12_gt.nii" / "DCM04-OH-AL_V2_12.nii"

LABELS = {
    3: ("LV", "#0072B2"),
    2: ("MYO", "#D55E00"),
    1: ("RV", "#009E73"),
}

DRAW_ORDER = (2, 3, 1)
LEGEND_ORDER = (3, 2, 1)

FLOW_BLUE = "#0072B2"
FLOW_ORANGE = "#D55E00"
FLOW_GREEN = "#009E73"
FLOW_DARK = "#243447"
FLOW_LIGHT = "#F6F8FA"

# Monochrome plate style, shared with scripts/fig_model_architecture.py.
PLATE_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.0,
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.6,
}
PLATE_INK = "#000000"
PLATE_GREY = "#555555"
PLATE_RULE = "#B5B5B5"
PLATE_SURFACE = "#9B9B9B"
PLATE_CMAP = "viridis"
PLATE_DASH = (0, (3.0, 1.8))
PLATE_DOT = (0, (1.0, 1.6))

# Figure-fraction height of the horizontal segment linking the two panel rows.
WRAP_Y = 0.500
MESH_ELEV = 16.0
MESH_AZIM = -58.0


def ensure_ssm_dir() -> Path:
    candidates = [
        Path(os.environ["CARDIOSDF_SSM_DIR"]) if "CARDIOSDF_SSM_DIR" in os.environ else None,
        ROOT / "Statistical-Shape-Model",
        ROOT / "notebooks" / "Statistical-Shape-Model",
        Path("/tmp/cardiosdf-ssm"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / SSM_FILE).exists():
            return candidate

    target = Path("/tmp/cardiosdf-ssm")
    subprocess.run(
        ["git", "clone", "--depth", "1", SSM_REPO_URL, str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return target


def load_legacy_vtk_polydata(vtk_path: Path) -> tuple[np.ndarray, np.ndarray]:
    tokens = vtk_path.read_text().split()

    point_index = tokens.index("POINTS")
    point_count = int(tokens[point_index + 1])
    point_start = point_index + 3
    point_stop = point_start + 3 * point_count
    vertices = np.asarray(tokens[point_start:point_stop], dtype=float).reshape(point_count, 3)

    polygon_index = tokens.index("POLYGONS")
    polygon_count = int(tokens[polygon_index + 1])
    cursor = polygon_index + 3
    faces = []
    for _ in range(polygon_count):
        vertex_count = int(tokens[cursor])
        ids = [int(value) for value in tokens[cursor + 1 : cursor + 1 + vertex_count]]
        if vertex_count == 3:
            faces.append(ids)
        elif vertex_count > 3:
            faces.extend([ids[0], ids[index], ids[index + 1]] for index in range(1, vertex_count - 1))
        cursor += vertex_count + 1

    vertices = vertices - vertices.mean(axis=0, keepdims=True)
    return vertices, np.asarray(faces, dtype=np.int32)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(lengths, 1e-8)

    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

    outward_score = np.mean(np.sum(normals * vertices, axis=1))
    if outward_score < 0:
        normals *= -1.0
    return normals


def ssm_wall_surfaces() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ssm_dir = ensure_ssm_dir()
    endocardium, faces = load_legacy_vtk_polydata(ssm_dir / SSM_FILE)
    normals = vertex_normals(endocardium, faces)

    z_norm = (endocardium[:, 2] - endocardium[:, 2].min()) / max(np.ptp(endocardium[:, 2]), 1e-8)
    angles = np.arctan2(endocardium[:, 1], endocardium[:, 0])
    thickness = 5.5 + 4.2 * z_norm + 1.1 * np.sin(2.0 * angles + 0.6) * (0.4 + 0.6 * z_norm)
    thickness = np.clip(thickness, 4.5, 11.5)
    epicardium = endocardium + normals * thickness[:, None]

    return endocardium, epicardium, faces, thickness


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data, spacing


def save_rgb_figure(fig: plt.Figure, output: Path, **kwargs: object) -> None:
    fig.savefig(output, facecolor=fig.get_facecolor(), **kwargs)
    with Image.open(output) as image:
        image.convert("RGB").save(output)


def union_crop(*segmentations: np.ndarray, margin: int = 24) -> tuple[slice, slice]:
    mask = np.zeros(segmentations[0].shape[:2], dtype=bool)
    for segmentation in segmentations:
        mask |= np.any(segmentation > 0, axis=2)

    rows, cols = np.where(mask)
    row_start = max(int(rows.min()) - margin, 0)
    row_stop = min(int(rows.max()) + margin + 1, mask.shape[0])
    col_start = max(int(cols.min()) - margin, 0)
    col_stop = min(int(cols.max()) + margin + 1, mask.shape[1])
    return slice(row_start, row_stop), slice(col_start, col_stop)


def representative_lv_slice(segmentation: np.ndarray) -> int:
    counts = np.sum(segmentation == 3, axis=(0, 1))
    return int(np.argmax(counts))


def window_bounds(*volumes: np.ndarray, crop: tuple[slice, slice]) -> tuple[float, float]:
    samples = []
    for volume in volumes:
        cropped = volume[crop[0], crop[1], :]
        foreground = cropped[cropped > 0]
        samples.append(foreground)
    values = np.concatenate(samples)
    return tuple(float(value) for value in np.percentile(values, [1.0, 99.4]))


def draw_contours(ax: plt.Axes, segmentation: np.ndarray, *, plate: bool = False) -> None:
    if not plate:
        for label in DRAW_ORDER:
            _, color = LABELS[label]
            binary = (segmentation == label).astype(float)
            if np.any(binary):
                ax.contour(binary, levels=[0.5], colors=[color], linewidths=1.30)
        return

    # Colour keeps the clinical labels readable; the line style repeats the
    # same information for greyscale printing.
    strokes = [
        (np.isin(segmentation, [2, 3]), FLOW_ORANGE, "dashed"),
        (segmentation == 3, FLOW_BLUE, "solid"),
        (segmentation == 1, FLOW_GREEN, "dotted"),
    ]
    for binary, color, linestyle in strokes:
        if not np.any(binary):
            continue
        contour = ax.contour(
            binary.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=1.15,
            linestyles=[linestyle],
        )
        contour.set_path_effects([patheffects.withStroke(linewidth=2.1, foreground="white")])


def contour_segments(binary: np.ndarray) -> list[np.ndarray]:
    if not np.any(binary) or np.all(binary):
        return []
    fig, ax = plt.subplots(figsize=(1, 1))
    contours = ax.contour(binary.astype(float), levels=[0.5])
    segments = [segment.copy() for segment in contours.allsegs[0] if len(segment) > 2]
    plt.close(fig)
    return segments


def draw_scale_bar(ax: plt.Axes, spacing_mm: float, length_mm: float = 20.0) -> None:
    length_px = length_mm / spacing_mm
    x_left, x_right = ax.get_xlim()
    y_bottom, _ = ax.get_ylim()
    x0 = x_right - length_px - 8
    y0 = y_bottom + 8
    ax.plot([x0, x0 + length_px], [y0, y0], color="black", linewidth=3.2, solid_capstyle="butt")
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", linewidth=2.0, solid_capstyle="butt")
    label = ax.text(
        x0 + length_px / 2,
        y0 + 4,
        f"{int(length_mm)} mm",
        color="white",
        fontsize=6.0,
        ha="center",
        va="bottom",
    )
    label.set_path_effects([patheffects.withStroke(linewidth=1.4, foreground="black")])


def style_viewport(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.7)


def make_clinical_viewer(
    ed_volume: np.ndarray,
    ed_segmentation: np.ndarray,
    es_volume: np.ndarray,
    es_segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    crop: tuple[slice, slice],
) -> Path:
    ed_slice = representative_lv_slice(ed_segmentation)
    es_slice = representative_lv_slice(es_segmentation)
    vmin, vmax = window_bounds(ed_volume, es_volume, crop=crop)

    panels = [
        ("ED, frame 1", ed_volume, ed_segmentation, ed_slice),
        ("ES, frame 12", es_volume, es_segmentation, es_slice),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.20, wspace=0.08)

    legend_handles = [
        Line2D([0], [0], color=LABELS[label][1], lw=2.0, label=LABELS[label][0])
        for label in LEGEND_ORDER
    ]

    for ax, (title, volume, segmentation, slice_index) in zip(axes, panels):
        image = volume[crop[0], crop[1], slice_index]
        labels = segmentation[crop[0], crop[1], slice_index]
        ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        draw_contours(ax, labels)
        draw_scale_bar(ax, spacing[0])
        style_viewport(ax)
        ax.set_title(title, fontsize=9.5, color="#222222", pad=6)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        ncol=3,
        fontsize=7.5,
        labelcolor="#222222",
        handlelength=2.0,
        columnspacing=1.7,
    )

    output = OUT_DIR / "acdc_clinical_viewer_ed_es.png"
    fig.savefig(output, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def grayscale_slice_rgba(image: np.ndarray, vmin: float, vmax: float, alpha: float = 0.78) -> np.ndarray:
    normalized = np.clip((image - vmin) / (vmax - vmin), 0, 1)
    rgba = plt.cm.gray(normalized)
    rgba[..., 3] = alpha
    return rgba


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    dilated = mask.copy()
    for _ in range(iterations):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        neighbours = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        dilated = np.logical_or.reduce(neighbours)
    return dilated


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[0:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, 0:-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def add_contours_to_texture(facecolors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    textured = facecolors.copy()
    for label in DRAW_ORDER:
        _, color = LABELS[label]
        boundary = dilate_mask(boundary_mask(labels == label), iterations=0)
        if np.any(boundary):
            textured[boundary, :3] = plt.matplotlib.colors.to_rgb(color)
            textured[boundary, 3] = 1.0
    return textured


def make_3d_stack(
    volume: np.ndarray,
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    crop: tuple[slice, slice],
) -> Path:
    vmin, vmax = window_bounds(volume, crop=crop)
    row_count = crop[0].stop - crop[0].start
    col_count = crop[1].stop - crop[1].start
    step = 1

    x = (np.arange(0, col_count, step) - col_count / 2.0) * spacing[0]
    y = (np.arange(0, row_count, step) - row_count / 2.0) * spacing[1]
    x_grid, y_grid = np.meshgrid(x, y)

    fig = plt.figure(figsize=(7.0, 5.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    for slice_index in range(volume.shape[2]):
        image = volume[crop[0], crop[1], slice_index][::step, ::step]
        labels = segmentation[crop[0], crop[1], slice_index][::step, ::step]
        z_grid = np.full_like(x_grid, slice_index * spacing[2], dtype=float)
        facecolors = add_contours_to_texture(grayscale_slice_rgba(image, vmin, vmax, alpha=0.70), labels)
        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            edgecolor="none",
            antialiased=False,
            shade=False,
        )

        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        z = slice_index * spacing[2]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5, color="#c8c8c8", linewidth=0.45, alpha=0.80)

    ax.view_init(elev=24, azim=-58, roll=0)
    ax.set_box_aspect((1.05, 1.0, 0.82))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(-2, (volume.shape[2] - 1) * spacing[2] + 10)
    ax.set_axis_off()

    output = OUT_DIR / "acdc_3d_sax_stack.png"
    fig.savefig(output, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def make_problem_visualisation_combined(clinical_viewer: Path, stack_3d: Path) -> Path:
    left = Image.open(clinical_viewer).convert("RGB")
    right = Image.open(stack_3d).convert("RGB")

    target_height = max(left.height, right.height)
    if left.height != target_height:
        left_width = int(round(left.width * (target_height / left.height)))
        left = left.resize((left_width, target_height), Image.Resampling.LANCZOS)
    if right.height != target_height:
        right_width = int(round(right.width * (target_height / right.height)))
        right = right.resize((right_width, target_height), Image.Resampling.LANCZOS)

    margin = 24
    gap = 20
    label_band = 24
    canvas_width = left.width + right.width + gap + 2 * margin
    canvas_height = target_height + label_band + 2 * margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    left_x = margin
    right_x = margin + left.width + gap
    image_y = margin + label_band
    canvas.paste(left, (left_x, image_y))
    canvas.paste(right, (right_x, image_y))

    draw = ImageDraw.Draw(canvas)
    draw.text((left_x + 2, margin), "(a)", fill="#222222")
    draw.text((right_x + 2, margin), "(b)", fill="#222222")

    output = OUT_DIR / "acdc_problem_visualisation.png"
    canvas.save(output)
    return output


def set_mesh_axes(ax: plt.Axes, vertices: np.ndarray, *, zoom: float = 1.0) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-8)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(tuple(spans / spans.max()), zoom=zoom)
    ax.set_axis_off()
    ax.view_init(elev=MESH_ELEV, azim=MESH_AZIM)


def shade_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    facecolor: str = PLATE_SURFACE,
    values: np.ndarray | None = None,
    cmap_name: str = PLATE_CMAP,
    norm: colors.Normalize | None = None,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    polygons = vertices[faces]
    face_normals = np.cross(polygons[:, 1] - polygons[:, 0], polygons[:, 2] - polygons[:, 0])
    face_normals = face_normals / np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-8)
    light_direction = np.asarray([-0.25, -0.50, 0.83])
    light_direction = light_direction / np.linalg.norm(light_direction)
    light = 0.45 + 0.55 * np.clip(np.abs(face_normals @ light_direction), 0.0, 1.0)

    if values is None:
        base = np.asarray(colors.to_rgb(facecolor))
        facecolors = np.empty((len(faces), 4))
        facecolors[:, :3] = np.clip(base[None, :] * light[:, None] + 0.16 * (1.0 - light[:, None]), 0.0, 1.0)
        facecolors[:, 3] = alpha
    else:
        face_values = values[faces].mean(axis=1)
        colormap = plt.get_cmap(cmap_name)
        if norm is None:
            norm = colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
        facecolors = colormap(norm(face_values))
        facecolors[:, :3] = np.clip(facecolors[:, :3] * (0.72 + 0.28 * light[:, None]), 0.0, 1.0)
        facecolors[:, 3] = alpha

    return polygons, facecolors


def near_quadrant_mask(vertices: np.ndarray, faces: np.ndarray, *, elev: float, azim: float) -> np.ndarray:
    """Faces of the quadrant closest to the camera, i.e. the wedge to cut away."""
    elev_rad, azim_rad = np.deg2rad(elev), np.deg2rad(azim)
    view = np.asarray([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad),
    ])
    right = np.asarray([-np.sin(azim_rad), np.cos(azim_rad), 0.0])
    centroids = vertices[faces].mean(axis=1) - vertices.mean(axis=0)
    return (centroids @ view > 0.0) & (centroids @ right > 0.0)


def draw_ssm_mesh(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    facecolor: str = PLATE_SURFACE,
    values: np.ndarray | None = None,
    cmap_name: str = PLATE_CMAP,
    norm: colors.Normalize | None = None,
    stride: int = 1,
    alpha: float = 1.0,
) -> Poly3DCollection:
    polygons, facecolors = shade_faces(
        vertices,
        faces[::stride],
        facecolor=facecolor,
        values=values,
        cmap_name=cmap_name,
        norm=norm,
        alpha=alpha,
    )
    collection = Poly3DCollection(polygons, facecolors=facecolors, edgecolors="none", linewidths=0.0)
    ax.add_collection3d(collection)
    return collection


def draw_wall_cutaway(
    ax: plt.Axes,
    endocardium: np.ndarray,
    epicardium: np.ndarray,
    faces: np.ndarray,
    *,
    endo_color: str,
    epi_color: str,
    elev: float = MESH_ELEV,
    azim: float = MESH_AZIM,
) -> None:
    """Both surfaces in one collection so depth sorting stays consistent."""
    kept_epi = faces[~near_quadrant_mask(epicardium, faces, elev=elev, azim=azim)]
    endo_polygons, endo_colors = shade_faces(endocardium, faces, facecolor=endo_color)
    epi_polygons, epi_colors = shade_faces(epicardium, kept_epi, facecolor=epi_color)
    ax.add_collection3d(Poly3DCollection(
        np.concatenate([endo_polygons, epi_polygons], axis=0),
        facecolors=np.concatenate([endo_colors, epi_colors], axis=0),
        edgecolors="none",
        linewidths=0.0,
    ))


def selected_lv_slices(segmentation: np.ndarray, count: int = 10) -> np.ndarray:
    available = np.flatnonzero(np.any(segmentation == 3, axis=(0, 1)))
    if len(available) <= count:
        return available
    positions = np.linspace(0, len(available) - 1, count).round().astype(int)
    return available[positions]


def draw_contour_stack(
    ax: plt.Axes,
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    crop: tuple[slice, slice],
) -> None:
    slice_indices = selected_lv_slices(segmentation, count=10)
    row_count = crop[0].stop - crop[0].start
    col_count = crop[1].stop - crop[1].start
    x_min = -col_count * spacing[0] / 2.0
    x_max = col_count * spacing[0] / 2.0
    y_min = -row_count * spacing[1] / 2.0
    y_max = row_count * spacing[1] / 2.0
    z_values = (slice_indices.mean() - slice_indices) * spacing[2]

    for slice_index, z_value in zip(slice_indices, z_values):
        labels = segmentation[crop[0], crop[1], slice_index]
        ax.plot(
            [x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            [z_value] * 5,
            color=PLATE_RULE,
            linewidth=0.35,
            alpha=0.90,
        )
        for binary_mask, line_style, line_width in [
            (np.isin(labels, [2, 3]), PLATE_DASH, 0.75),
            (labels == 3, "solid", 0.90),
        ]:
            segments = contour_segments(binary_mask)
            for segment in segments:
                x_coords = (segment[:, 0] - col_count / 2.0) * spacing[0]
                y_coords = (segment[:, 1] - row_count / 2.0) * spacing[1]
                ax.plot(
                    x_coords,
                    y_coords,
                    np.full_like(x_coords, z_value),
                    color=PLATE_INK,
                    linestyle=line_style,
                    linewidth=line_width,
                )

    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.76))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_values.min() - 5.0, z_values.max() + 5.0)
    ax.set_axis_off()


def draw_slice_panel(
    ax: plt.Axes,
    volume: np.ndarray,
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    crop: tuple[slice, slice],
    *,
    plate: bool = False,
) -> None:
    slice_index = representative_lv_slice(segmentation)
    vmin, vmax = window_bounds(volume, crop=crop)
    ax.imshow(
        volume[crop[0], crop[1], slice_index],
        cmap="gray",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    draw_contours(ax, segmentation[crop[0], crop[1], slice_index], plate=plate)
    draw_scale_bar(ax, spacing[0], length_mm=20.0)
    style_viewport(ax)
    if plate:
        for spine in ax.spines.values():
            spine.set_color(PLATE_INK)
            spine.set_linewidth(0.6)


def aha_segment_values(vertices: np.ndarray, thickness: np.ndarray) -> dict[int, float]:
    z_norm = (vertices[:, 2] - vertices[:, 2].min()) / max(np.ptp(vertices[:, 2]), 1e-8)
    angles = (np.arctan2(vertices[:, 1], vertices[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    values: dict[int, float] = {}

    segment_id = 1
    for lower, upper, segment_count in [(0.67, 1.01, 6), (0.34, 0.67, 6), (0.10, 0.34, 4)]:
        ring_mask = (z_norm >= lower) & (z_norm < upper)
        for segment_index in range(segment_count):
            angle_lower = 2.0 * np.pi * segment_index / segment_count
            angle_upper = 2.0 * np.pi * (segment_index + 1) / segment_count
            segment_mask = ring_mask & (angles >= angle_lower) & (angles < angle_upper)
            if np.any(segment_mask):
                values[segment_id] = float(np.mean(thickness[segment_mask]))
            else:
                values[segment_id] = float(np.mean(thickness[ring_mask]))
            segment_id += 1

    apical_mask = z_norm < 0.10
    values[17] = float(np.mean(thickness[apical_mask])) if np.any(apical_mask) else float(np.mean(thickness))
    return values


def draw_aha17(ax: plt.Axes, segment_values: dict[int, float], norm: colors.Normalize) -> None:
    colormap = plt.get_cmap(PLATE_CMAP)
    number_effects = [patheffects.withStroke(linewidth=1.3, foreground="black")]
    rings = [
        (0.72, 1.00, 6, 1, 90.0),
        (0.45, 0.72, 6, 7, 90.0),
        (0.20, 0.45, 4, 13, 45.0),
    ]
    for inner_radius, outer_radius, segment_count, first_id, offset in rings:
        for segment_index in range(segment_count):
            segment_id = first_id + segment_index
            theta1 = offset - 360.0 * (segment_index + 1) / segment_count
            theta2 = offset - 360.0 * segment_index / segment_count
            wedge = patches.Wedge(
                (0.0, 0.0),
                outer_radius,
                theta1,
                theta2,
                width=outer_radius - inner_radius,
                facecolor=colormap(norm(segment_values[segment_id])),
                edgecolor=PLATE_INK,
                linewidth=0.45,
            )
            ax.add_patch(wedge)
            mid_angle = np.deg2rad((theta1 + theta2) / 2.0)
            radius = (inner_radius + outer_radius) / 2.0
            label = ax.text(
                radius * np.cos(mid_angle),
                radius * np.sin(mid_angle),
                str(segment_id),
                ha="center",
                va="center",
                fontsize=5.4,
                color="white",
            )
            label.set_path_effects(number_effects)

    centre = patches.Circle(
        (0.0, 0.0),
        0.20,
        facecolor=colormap(norm(segment_values[17])),
        edgecolor=PLATE_INK,
        linewidth=0.45,
    )
    ax.add_patch(centre)
    centre_label = ax.text(0.0, 0.0, "17", ha="center", va="center", fontsize=5.4, color="white")
    centre_label.set_path_effects(number_effects)
    ax.set_aspect("equal")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.axis("off")


def plate_legend_handles(*, include_rv: bool = False) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=PLATE_INK, lw=0.9, label="endocardium"),
        Line2D([0], [0], color=PLATE_INK, lw=0.9, linestyle=PLATE_DASH, label="epicardium"),
    ]
    if include_rv:
        handles.append(
            Line2D([0], [0], color=PLATE_INK, lw=0.9, linestyle=PLATE_DOT, label="right ventricle")
        )
    return handles


def label_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=FLOW_BLUE, lw=1.1, label="endocardium"),
        Line2D([0], [0], color=FLOW_ORANGE, lw=1.1, linestyle=(0, (3.0, 1.8)), label="epicardium"),
        Line2D([0], [0], color=FLOW_GREEN, lw=1.1, linestyle=(0, (1.0, 1.6)), label="right ventricle"),
    ]


def plate_legend(ax: plt.Axes, handles: list[Line2D], *, y: float) -> None:
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, y),
        frameon=False,
        ncol=len(handles),
        fontsize=5.9,
        labelcolor=PLATE_GREY,
        handlelength=1.60,
        columnspacing=0.90,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


def plate_header(fig: plt.Figure, ax: plt.Axes, tag: str, title: str, note: str | None = None) -> None:
    box = ax.get_position()
    fig.text(box.x0, box.y1 + 0.030, f"({tag})", ha="left", va="baseline",
             fontsize=7.2, fontweight="bold", color=PLATE_INK)
    fig.text(box.x0 + 0.033, box.y1 + 0.030, title, ha="left", va="baseline",
             fontsize=7.2, fontweight="bold", color=PLATE_INK)
    if note is not None:
        fig.text(box.x0, box.y1 + 0.008, note, ha="left", va="baseline",
                 fontsize=5.9, color=PLATE_GREY, style="italic")


def plate_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float]) -> None:
    fig.add_artist(patches.FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7.0,
        linewidth=0.7, color=PLATE_INK, shrinkA=0, shrinkB=0, clip_on=False, zorder=40))


def plate_route(fig: plt.Figure, points: list[tuple[float, float]]) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        fig.add_artist(Line2D(
            [start[0], end[0]], [start[1], end[1]], transform=fig.transFigure,
            color=PLATE_INK, lw=0.7, solid_capstyle="round", clip_on=False, zorder=40))
    plate_arrow(fig, points[-2], points[-1])


def draw_pipeline_arrows(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    slice_axis, contour_axis, ssm_axis, heatmap_axis, aha_axis = axes
    boxes = [axis.get_position() for axis in
             (slice_axis, contour_axis, ssm_axis, heatmap_axis, aha_axis)]
    slice_box, contour_box, ssm_box, heatmap_box, aha_box = boxes

    def mid_y(axis_box: object) -> float:
        return 0.5 * (axis_box.y0 + axis_box.y1)

    def mid_x(axis_box: object) -> float:
        return 0.5 * (axis_box.x0 + axis_box.x1)

    plate_arrow(fig, (slice_box.x1 + 0.010, mid_y(slice_box)),
                (contour_box.x0 - 0.010, mid_y(contour_box)))
    plate_arrow(fig, (contour_box.x1 + 0.010, mid_y(contour_box)),
                (ssm_box.x0 - 0.010, mid_y(ssm_box)))
    plate_arrow(fig, (heatmap_box.x1 + 0.010, mid_y(heatmap_box)),
                (aha_box.x0 - 0.010, mid_y(aha_box)))

    # Return sweep from the end of the top row to the start of the bottom row.
    plate_route(fig, [
        (mid_x(ssm_box), ssm_box.y0 - 0.050),
        (mid_x(ssm_box), WRAP_Y),
        (0.018, WRAP_Y),
        (0.018, mid_y(heatmap_box)),
        (heatmap_box.x0 - 0.010, mid_y(heatmap_box)),
    ])


def make_pipeline_visual_flow(
    volume: np.ndarray,
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    crop: tuple[slice, slice],
) -> Path:
    endocardium, epicardium, faces, thickness = ssm_wall_surfaces()
    thickness_norm = colors.Normalize(vmin=4.0, vmax=12.0)
    segment_values = aha_segment_values(endocardium, thickness)

    with plt.rc_context(PLATE_RC):
        fig = plt.figure(figsize=(7.2, 5.4), facecolor="white")

        # Two rows read left to right; the return sweep links (c) to (d).
        slice_axis = fig.add_axes([0.045, 0.585, 0.250, 0.340])
        contour_axis = fig.add_axes([0.355, 0.570, 0.270, 0.370], projection="3d")
        ssm_axis = fig.add_axes([0.690, 0.570, 0.270, 0.370], projection="3d")
        heatmap_axis = fig.add_axes([0.050, 0.070, 0.330, 0.370], projection="3d")
        aha_axis = fig.add_axes([0.470, 0.085, 0.300, 0.345])

        draw_slice_panel(slice_axis, volume, segmentation, spacing, crop, plate=True)
        draw_contour_stack(contour_axis, segmentation, spacing, crop)

        draw_wall_cutaway(ssm_axis, endocardium, epicardium, faces,
                          endo_color="#6F6F6F", epi_color="#CFCFCF")
        set_mesh_axes(ssm_axis, epicardium, zoom=1.22)

        draw_ssm_mesh(heatmap_axis, endocardium, faces, values=thickness,
                      norm=thickness_norm, alpha=1.0)
        set_mesh_axes(heatmap_axis, endocardium, zoom=1.30)

        draw_aha17(aha_axis, segment_values, thickness_norm)

        headers = [
            (slice_axis, "a", "Short-axis observation",
             "ACDC end-diastolic frame with expert labels"),
            (contour_axis, "b", "Contour extraction",
             "rings at their physical slice positions"),
            (ssm_axis, "c", "LV surface pair",
             "endocardial and epicardial geometry"),
            (heatmap_axis, "d", "Local wall thickness",
             "one value per endocardial vertex"),
            (aha_axis, "e", "AHA-17 summary",
             "segment means of the surface field"),
        ]
        for axis, tag, title, note in headers:
            plate_header(fig, axis, tag, title, note)

        plate_legend(slice_axis, label_legend_handles(), y=-0.030)
        plate_legend(contour_axis, plate_legend_handles(), y=0.145)
        ssm_axis.legend(
            handles=[
                patches.Patch(facecolor="#6F6F6F", edgecolor="none", label="endocardium"),
                patches.Patch(facecolor="#CFCFCF", edgecolor="none", label="epicardium (cut away)"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.50, -0.020),
            frameon=False,
            ncol=2,
            fontsize=5.9,
            labelcolor=PLATE_GREY,
            handlelength=1.10,
            handleheight=0.80,
            columnspacing=0.90,
            handletextpad=0.35,
            borderaxespad=0.0,
        )

        colorbar_axis = fig.add_axes([0.815, 0.145, 0.016, 0.250])
        scalar_mappable = plt.cm.ScalarMappable(norm=thickness_norm, cmap=PLATE_CMAP)
        colorbar = fig.colorbar(scalar_mappable, cax=colorbar_axis, orientation="vertical")
        colorbar.set_label("wall thickness (mm)", fontsize=6.2, color=PLATE_INK, labelpad=3)
        colorbar.outline.set_linewidth(0.5)
        colorbar.outline.set_edgecolor(PLATE_INK)
        colorbar.ax.tick_params(labelsize=5.7, length=2, width=0.5, colors=PLATE_INK)

        draw_pipeline_arrows(fig, [slice_axis, contour_axis, ssm_axis, heatmap_axis, aha_axis])

        output = OUT_DIR / "cardiosdf_pipeline_visual_flow.png"
        save_rgb_figure(fig, output, dpi=400, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return output


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ed_volume, spacing = load_nifti(ED_MRI)
    ed_segmentation, _ = load_nifti(ED_SEG)
    es_volume, _ = load_nifti(ES_MRI)
    es_segmentation, _ = load_nifti(ES_SEG)
    ed_segmentation = ed_segmentation.astype(np.uint8)
    es_segmentation = es_segmentation.astype(np.uint8)

    crop = union_crop(ed_segmentation, es_segmentation)
    pipeline_output = make_pipeline_visual_flow(ed_volume, ed_segmentation, spacing, crop)
    clinical_output = make_clinical_viewer(
        ed_volume, ed_segmentation, es_volume, es_segmentation, spacing, crop
    )
    stack_output = make_3d_stack(ed_volume, ed_segmentation, spacing, crop)
    combined_output = make_problem_visualisation_combined(clinical_output, stack_output)

    outputs = [pipeline_output, clinical_output, stack_output, combined_output]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()