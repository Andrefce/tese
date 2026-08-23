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


def draw_contours(ax: plt.Axes, segmentation: np.ndarray) -> None:
    for label in DRAW_ORDER:
        _, color = LABELS[label]
        binary = (segmentation == label).astype(float)
        if np.any(binary):
            ax.contour(binary, levels=[0.5], colors=[color], linewidths=1.30)


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


def set_mesh_axes(ax: plt.Axes, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centre = (mins + maxs) / 2.0
    half_range = float(np.max(maxs - mins) / 2.0)
    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(centre[2] - half_range, centre[2] + half_range)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-58)


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
    cutaway: bool = False,
) -> Poly3DCollection:
    selected_faces = faces[::stride]
    if cutaway:
        # Remove the octant facing the camera so the inner surface stays visible.
        centroids = vertices[selected_faces].mean(axis=1) - vertices.mean(axis=0)
        selected_faces = selected_faces[~((centroids[:, 0] > 0.0) & (centroids[:, 1] < 0.0))]

    polygons = vertices[selected_faces]
    face_normals = np.cross(polygons[:, 1] - polygons[:, 0], polygons[:, 2] - polygons[:, 0])
    face_normals = face_normals / np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-8)
    light_direction = np.asarray([-0.25, -0.50, 0.83])
    light_direction = light_direction / np.linalg.norm(light_direction)
    light = 0.45 + 0.55 * np.clip(np.abs(face_normals @ light_direction), 0.0, 1.0)

    if values is None:
        base = np.asarray(colors.to_rgb(facecolor))
        facecolors = np.empty((len(selected_faces), 4))
        facecolors[:, :3] = np.clip(base[None, :] * light[:, None] + 0.16 * (1.0 - light[:, None]), 0.0, 1.0)
        facecolors[:, 3] = alpha
    else:
        face_values = values[selected_faces].mean(axis=1)
        colormap = plt.get_cmap(cmap_name)
        if norm is None:
            norm = colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
        facecolors = colormap(norm(face_values))
        facecolors[:, :3] = np.clip(facecolors[:, :3] * (0.72 + 0.28 * light[:, None]), 0.0, 1.0)
        facecolors[:, 3] = alpha

    collection = Poly3DCollection(
        polygons,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
    )
    ax.add_collection3d(collection)
    return collection


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
            color="#D6DCE2",
            linewidth=0.42,
            alpha=0.86,
        )
        for binary_mask, line_color, line_width in [
            (np.isin(labels, [2, 3]), FLOW_ORANGE, 1.25),
            (labels == 3, FLOW_BLUE, 1.15),
        ]:
            segments = contour_segments(binary_mask)
            for segment in segments:
                x_coords = (segment[:, 0] - col_count / 2.0) * spacing[0]
                y_coords = (segment[:, 1] - row_count / 2.0) * spacing[1]
                ax.plot(x_coords, y_coords, np.full_like(x_coords, z_value), color=line_color, linewidth=line_width)

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
    draw_contours(ax, segmentation[crop[0], crop[1], slice_index])
    draw_scale_bar(ax, spacing[0], length_mm=20.0)
    style_viewport(ax)


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
    colormap = plt.get_cmap("turbo")
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
                edgecolor="white",
                linewidth=1.2,
            )
            ax.add_patch(wedge)
            mid_angle = np.deg2rad((theta1 + theta2) / 2.0)
            radius = (inner_radius + outer_radius) / 2.0
            ax.text(
                radius * np.cos(mid_angle),
                radius * np.sin(mid_angle),
                str(segment_id),
                ha="center",
                va="center",
                fontsize=5.6,
                color="white",
                weight="bold",
            )

    centre = patches.Circle(
        (0.0, 0.0),
        0.20,
        facecolor=colormap(norm(segment_values[17])),
        edgecolor="white",
        linewidth=1.2,
    )
    ax.add_patch(centre)
    ax.text(0.0, 0.0, "17", ha="center", va="center", fontsize=5.6, color="white", weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.axis("off")


def set_panel_caption(ax: plt.Axes, label: str, caption: str, *, y: float | None = None) -> None:
    title_kwargs = {
        "fontsize": 7.5,
        "color": "#333333",
        "pad": 4,
        "style": "italic",
    }
    if y is not None:
        title_kwargs["y"] = y
        title_kwargs["pad"] = 0
    ax.set_title(f"{label} {caption}", **title_kwargs)


def segmentation_legend_handles(*, include_rv: bool = False) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=FLOW_BLUE, lw=1.4, label="Endocardium"),
        Line2D([0], [0], color=FLOW_ORANGE, lw=1.4, label="Epicardium"),
    ]
    if include_rv:
        handles.append(Line2D([0], [0], color=FLOW_GREEN, lw=1.4, label="RV"))
    return handles


def draw_pipeline_arrows(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    slice_axis, contour_axis, ssm_axis, heatmap_axis, aha_axis = axes
    arrow_kw = dict(
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=11.0,
        linewidth=1.0,
        color="#444444",
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
        zorder=40,
    )

    def box(axis: plt.Axes) -> object:
        return axis.get_position()

    def mid_y(axis_box: object) -> float:
        return 0.5 * (axis_box.y0 + axis_box.y1)

    def mid_x(axis_box: object) -> float:
        return 0.5 * (axis_box.x0 + axis_box.x1)

    slice_box = box(slice_axis)
    contour_box = box(contour_axis)
    ssm_box = box(ssm_axis)
    heatmap_box = box(heatmap_axis)
    aha_box = box(aha_axis)

    arrows = [
        ((slice_box.x1 + 0.014, mid_y(slice_box)), (contour_box.x0 - 0.014, mid_y(contour_box))),
        ((contour_box.x1 + 0.014, mid_y(contour_box)), (ssm_box.x0 - 0.014, mid_y(ssm_box))),
        ((mid_x(ssm_box), ssm_box.y0 - 0.010), (mid_x(heatmap_box), heatmap_box.y1 + 0.010)),
        ((heatmap_box.x0 - 0.014, mid_y(heatmap_box)), (aha_box.x1 + 0.014, mid_y(aha_box))),
    ]

    for start, end in arrows:
        fig.add_artist(patches.FancyArrowPatch(start, end, **arrow_kw))


def make_pipeline_visual_flow(
    volume: np.ndarray,
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    crop: tuple[slice, slice],
) -> Path:
    endocardium, epicardium, faces, thickness = ssm_wall_surfaces()
    thickness_norm = colors.Normalize(vmin=4.0, vmax=12.0)
    segment_values = aha_segment_values(endocardium, thickness)

    # --- Figure: 2 rows, academic journal style ---
    fig = plt.figure(figsize=(7.2, 4.8), facecolor="white")

    # Top row: (a) SAX slice, (b) contours, (c) LV mesh
    top_y, top_h = 0.55, 0.38
    top_positions = [
        [0.04, top_y, 0.24, top_h],
        [0.36, top_y, 0.26, top_h],
        [0.70, top_y, 0.26, top_h],
    ]

    # Bottom row (reversed): (d) thickness directly under (c), (e) AHA-17
    bot_y, bot_h = 0.07, 0.38
    bot_positions = [
        [0.70, bot_y, 0.26, bot_h],   # (d) same x as (c)
        [0.36, bot_y, 0.22, bot_h],   # (e) closer to (d)
    ]

    slice_axis = fig.add_axes(top_positions[0])
    contour_axis = fig.add_axes(top_positions[1], projection="3d")
    ssm_axis = fig.add_axes(top_positions[2], projection="3d")
    heatmap_axis = fig.add_axes(bot_positions[0], projection="3d")
    aha_axis = fig.add_axes(bot_positions[1])

    # --- Draw content ---
    draw_slice_panel(slice_axis, volume, segmentation, spacing, crop)
    draw_contour_stack(contour_axis, segmentation, spacing, crop)
    draw_ssm_mesh(ssm_axis, epicardium, faces, facecolor=FLOW_ORANGE, stride=2, alpha=0.50)
    draw_ssm_mesh(ssm_axis, endocardium, faces, facecolor=FLOW_BLUE, stride=2, alpha=0.86)
    draw_ssm_mesh(heatmap_axis, endocardium, faces, values=thickness, norm=thickness_norm, stride=2, alpha=0.98)
    draw_aha17(aha_axis, segment_values, thickness_norm)

    all_axes = [slice_axis, contour_axis, ssm_axis, heatmap_axis, aha_axis]
    captions = ["SAX slice", "Contour extraction", "LV mesh (SSM)", "Wall thickness", "AHA-17"]
    for idx, (ax, caption) in enumerate(zip(all_axes, captions)):
        label = f"({chr(ord('a') + idx)})"
        if ax is heatmap_axis:
            set_panel_caption(ax, label, caption, y=0.96)
        else:
            set_panel_caption(ax, label, caption)

    slice_axis.legend(
        handles=segmentation_legend_handles(include_rv=True),
        loc="upper center",
        bbox_to_anchor=(0.50, -0.055),
        frameon=False,
        ncol=3,
        fontsize=5.9,
        labelcolor="#333333",
        handlelength=1.20,
        columnspacing=0.70,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    contour_axis.legend(
        handles=segmentation_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.02),
        frameon=False,
        ncol=2,
        fontsize=6.2,
        labelcolor="#333333",
        handlelength=1.35,
        columnspacing=0.80,
        handletextpad=0.40,
    )
    ssm_axis.legend(
        handles=segmentation_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.02),
        frameon=False,
        ncol=2,
        fontsize=6.2,
        labelcolor="#333333",
        handlelength=1.35,
        columnspacing=0.80,
        handletextpad=0.40,
    )

    # --- Colorbar ---
    heatmap_box = heatmap_axis.get_position()
    colorbar_axis = fig.add_axes([
        heatmap_box.x0 + 0.10 * heatmap_box.width,
        heatmap_box.y0 - 0.035,
        0.80 * heatmap_box.width,
        0.014,
    ])
    scalar_mappable = plt.cm.ScalarMappable(norm=thickness_norm, cmap="turbo")
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Thickness (mm)", fontsize=6.0, labelpad=1)
    colorbar.ax.tick_params(labelsize=5.5, length=2)

    # --- Arrows ---
    draw_pipeline_arrows(fig, all_axes)

    output = OUT_DIR / "cardiosdf_pipeline_visual_flow.png"
    save_rgb_figure(fig, output, dpi=300, bbox_inches="tight", pad_inches=0.04)
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