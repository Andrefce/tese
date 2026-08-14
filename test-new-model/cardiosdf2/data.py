"""ACDC -> CardioSDF training cache, dataset and group/phase-balanced sampler.

The original training stream (``notebooks/training.ipynb``) read pre-built
Kaggle caches that are not part of this repository. This module rebuilds an
equivalent cache locally, directly from the ACDC ground-truth segmentations,
using the same geometry primitives that ``scripts/eval_demo`` uses to produce
the *voxel* reference of the Results chapter. Training targets and evaluation
reference are therefore identical objects.

One cache entry per (patient, phase). Everything is stored in the normalised
model space defined by ``geometry.extract_contours`` (centroid-centred,
mean in-plane radius = 1, z flipped), so the network never sees millimetres.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates, zoom as ndi_zoom

from geometry import (  # noqa: E402  (path injected by the package __init__)
    _clean_inside, _crop_bounds, extract_contours, load_segmentation,
    make_watertight, marching_cubes_mesh, read_info_cfg,
    signed_distance_from_mask,
)

GROUPS = ("NOR", "HCM", "DCM", "MINF", "RV")
PHASES = ("ED", "ES")

ISO_PITCH = 1.0          # mm, isotropic resample pitch for the SDF targets
GRID_MARGIN_MM = 22.0    # zero padding beyond the epicardial bounding box
N_SURF_CACHE = 4096      # surface sample pool per surface
N_QUERY_CACHE = 8192     # cached query pool with SDF targets


# ──────────────────────────────────────────────────────────────────────────
# Cache construction
# ──────────────────────────────────────────────────────────────────────────
def find_gt_frame(patient_dir: Path, patient_id: str, frame_no: int) -> Path:
    stem = f"{patient_id}_frame{frame_no:02d}_gt"
    for candidate in (f"{stem}.nii.gz", f"{stem}.nii", stem):
        path = patient_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"{stem} not found in {patient_dir}")


def discover_patients(data_root: Path) -> list[tuple[Path, dict]]:
    out = []
    for cfg in sorted(Path(data_root).glob("*/Info.cfg")):
        out.append((cfg.parent, read_info_cfg(cfg)))
    return out


def _iso_fields(seg, pitch: float = ISO_PITCH, margin_mm: float = GRID_MARGIN_MM):
    """Isotropic signed-distance fields of the endocardium and epicardium.

    Returns the two fields plus the (row, col, slice) millimetre origin of the
    grid. The masks are zero-padded so the fields stay exact over the whole
    padded contour bounding box the decoder is queried on.
    """
    spacing = np.asarray(seg.spacing, np.float64)
    start, stop = _crop_bounds(seg.epi, spacing, 6.0)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    factors = spacing / pitch

    def resample(mask: np.ndarray) -> np.ndarray:
        return ndi_zoom(mask[sl].astype(np.float32), factors, order=1,
                        prefilter=False) >= 0.5

    lv = _clean_inside(resample(seg.lv))
    epi = _clean_inside(resample(seg.epi) | lv)
    pad = int(round(margin_mm / pitch))
    lv = np.pad(lv, pad, mode="constant", constant_values=False)
    epi = np.pad(epi, pad, mode="constant", constant_values=False)

    voxel = np.full(3, pitch)
    origin = start.astype(np.float64) * spacing - pad * pitch
    endo_field = signed_distance_from_mask(lv, voxel, smooth_sigma=0.6)
    epi_field = signed_distance_from_mask(epi, voxel, smooth_sigma=0.6)
    return endo_field, epi_field, origin, voxel, lv, epi


def _mesh_from_field(field, origin, voxel, name) -> trimesh.Trimesh:
    raw = marching_cubes_mesh(field, origin, voxel)
    if len(raw.vertices):
        v = np.asarray(raw.vertices)
        raw = trimesh.Trimesh(vertices=np.column_stack([-v[:, 1], -v[:, 0], v[:, 2]]),
                              faces=raw.faces, process=False)
    mesh, _ = make_watertight(raw, name, taubin_iters=12)
    return mesh


def _sample_field_norm(field, origin, pitch, pts_norm, centroid, scale, flip_z=True):
    """Trilinear lookup of a millimetre SDF grid at normalised-space points."""
    flip = np.array([1.0, 1.0, -1.0 if flip_z else 1.0])
    world = np.asarray(pts_norm, np.float64) * flip * scale + centroid
    grid = np.column_stack([-world[:, 1], -world[:, 0], world[:, 2]])
    idx = (grid - origin) / pitch
    val = map_coordinates(field, idx.T, order=1, mode="nearest")
    return (val / scale).astype(np.float32)


def _sample_surface(mesh: trimesh.Trimesh, n: int):
    """Area-weighted surface sample with per-point face normals."""
    mesh = mesh.copy()
    mesh.fix_normals()
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = np.asarray(mesh.face_normals)[face_idx]
    return np.asarray(pts, np.float32), np.asarray(normals, np.float32)


def build_sample(patient_dir: Path, info: dict, phase: str, out_dir: Path,
                 force: bool = False) -> Path:
    """Build (or reuse) the cache entry for one patient and one cardiac phase."""
    patient_id = patient_dir.name
    out_path = Path(out_dir) / f"{patient_id}_{phase}.npz"
    if out_path.exists() and not force:
        return out_path

    warnings.filterwarnings("ignore")
    seg = load_segmentation(find_gt_frame(patient_dir, patient_id, int(info[phase])))
    contours = extract_contours(seg)
    centroid, scale = contours["centroid"].astype(np.float64), float(contours["scale"])

    endo_field, epi_field, origin, voxel, lv_mask, epi_mask = _iso_fields(seg)
    endo_mesh = _mesh_from_field(endo_field, origin, voxel, "voxel-endo")
    epi_mesh = _mesh_from_field(epi_field, origin, voxel, "voxel-epi")
    if len(endo_mesh.faces) == 0 or len(epi_mesh.faces) == 0:
        raise ValueError(f"empty reference mesh for {patient_id} {phase}")

    flip = np.array([1.0, 1.0, -1.0])

    def to_norm(pts):
        return ((np.asarray(pts, np.float64) - centroid) / scale * flip).astype(np.float32)

    se_pts_mm, se_n_mm = _sample_surface(endo_mesh, N_SURF_CACHE)
    sp_pts_mm, sp_n_mm = _sample_surface(epi_mesh, N_SURF_CACHE)
    se_pts, sp_pts = to_norm(se_pts_mm), to_norm(sp_pts_mm)
    se_n = (se_n_mm * flip).astype(np.float32)
    sp_n = (sp_n_mm * flip).astype(np.float32)
    se_n /= np.clip(np.linalg.norm(se_n, axis=1, keepdims=True), 1e-9, None)
    sp_n /= np.clip(np.linalg.norm(sp_n, axis=1, keepdims=True), 1e-9, None)

    # U2 target: the wall the decoder must output. At an endocardial surface
    # point f_endo = 0 and f_epi = -delta, so delta_gt is the unsigned distance
    # from that point to the epicardial surface.
    wall_gt = -_sample_field_norm(epi_field, origin, ISO_PITCH, se_pts,
                                  centroid, scale)

    xyz = contours["xyz"]
    lo, hi = xyz.min(0) - 0.30, xyz.max(0) + 0.30
    rng = np.random.default_rng(abs(hash((patient_id, phase))) % (2 ** 32))
    n_near = N_QUERY_CACHE // 2
    base = np.concatenate([se_pts, sp_pts])[rng.integers(0, 2 * N_SURF_CACHE, n_near)]
    sigma = rng.choice([0.015, 0.05], size=(n_near, 1)).astype(np.float32)
    q_near = base + rng.normal(0.0, 1.0, base.shape).astype(np.float32) * sigma
    q_free = rng.uniform(lo, hi, (N_QUERY_CACHE - n_near, 3)).astype(np.float32)
    q_pts = np.concatenate([q_near, q_free]).astype(np.float32)

    q_e = _sample_field_norm(endo_field, origin, ISO_PITCH, q_pts, centroid, scale)
    q_p = _sample_field_norm(epi_field, origin, ISO_PITCH, q_pts, centroid, scale)

    np.savez_compressed(
        out_path,
        patient=patient_id, group=info.get("Group", "?"), phase=phase,
        contour_xyz=xyz.astype(np.float32),
        contour_tissue=contours["tissue"].astype(np.float32),
        centroid=centroid.astype(np.float32), scale=np.float32(scale),
        bbox_lo=lo.astype(np.float32), bbox_hi=hi.astype(np.float32),
        surf_endo_pts=se_pts, surf_endo_n=se_n,
        surf_epi_pts=sp_pts, surf_epi_n=sp_n,
        surf_endo_wall=wall_gt.astype(np.float32),
        query_pts=q_pts, query_e_sdf=q_e, query_p_sdf=q_p,
        endo_volume_ml=np.float32(abs(endo_mesh.volume) / 1000.0),
        epi_volume_ml=np.float32(abs(epi_mesh.volume) / 1000.0),
        n_slices=np.int32(len(contours["slices"])),
    )
    return out_path


# ──────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SampleSpec:
    path: Path
    patient: str
    group: str
    phase: str


def index_cache(cache_dir: Path) -> list[SampleSpec]:
    specs = []
    for path in sorted(Path(cache_dir).glob("patient*_E[DS].npz")):
        with np.load(path, allow_pickle=False) as data:
            specs.append(SampleSpec(path, str(data["patient"]), str(data["group"]),
                                    str(data["phase"])))
    return specs


def split_by_patient(specs: list[SampleSpec], val_fraction: float = 0.15,
                     seed: int = 42) -> tuple[list[SampleSpec], list[SampleSpec]]:
    """Patient-disjoint split, stratified by pathology group (no leakage)."""
    rng = np.random.default_rng(seed)
    val_patients: set[str] = set()
    by_group: dict[str, list[str]] = {}
    for spec in specs:
        by_group.setdefault(spec.group, []).append(spec.patient)
    for group, patients in by_group.items():
        uniq = np.array(sorted(set(patients)))
        n_val = max(1, int(round(len(uniq) * val_fraction)))
        val_patients.update(rng.permutation(uniq)[:n_val].tolist())
    train = [s for s in specs if s.patient not in val_patients]
    val = [s for s in specs if s.patient in val_patients]
    return train, val


class LVSDFDataset:
    """Per-item point sets drawn from a cache entry.

    ``torch.utils.data.Dataset`` is duck-typed here to keep this module free of
    a torch import at cache-build time.
    """

    def __init__(self, specs: list[SampleSpec], cfg: dict, augment: bool = False,
                 seed: int = 0):
        self.specs = list(specs)
        self.cfg = cfg
        self.augment = augment
        self.seed = seed
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.specs)

    def _arrays(self, i: int) -> dict:
        if i not in self._cache:
            with np.load(self.specs[i].path, allow_pickle=False) as data:
                self._cache[i] = {k: data[k] for k in data.files
                                  if data[k].dtype.kind in "fiu"}
        return self._cache[i]

    def __getitem__(self, i: int) -> dict:
        cfg = self.cfg
        d = self._arrays(i)
        spec = self.specs[i]
        rng = np.random.default_rng((self.seed, i, np.random.randint(1 << 30)))
        phase_val = 0.0 if spec.phase == "ED" else 1.0

        def take(key, n, *others):
            idx = rng.integers(0, len(d[key]), n)
            return (d[key][idx],) + tuple(d[o][idx] for o in others)

        se_pts, se_n, se_wall = take("surf_endo_pts", cfg["n_surf_endo"],
                                     "surf_endo_n", "surf_endo_wall")
        sp_pts, sp_n = take("surf_epi_pts", cfg["n_surf_epi"], "surf_epi_n")
        q_pts, q_e, q_p = take("query_pts", cfg["n_query_sdf"],
                               "query_e_sdf", "query_p_sdf")

        anchor = np.concatenate([se_pts, sp_pts])
        near_idx = rng.integers(0, len(anchor), cfg["n_near"])
        near = (anchor[near_idx]
                + rng.normal(0.0, cfg["near_sigma"], (cfg["n_near"], 3))).astype(np.float32)
        lo, hi = d["bbox_lo"], d["bbox_hi"]
        free = rng.uniform(lo, hi, (cfg["n_free"], 3)).astype(np.float32)

        contour = np.column_stack([
            d["contour_xyz"], d["contour_tissue"],
            np.full(len(d["contour_xyz"]), phase_val, np.float32),
        ]).astype(np.float32)
        if self.augment:
            contour = _augment_contour(contour, cfg, rng)

        return {
            "contour": contour,
            "surf_endo_pts": se_pts, "surf_endo_n": se_n, "surf_endo_wall": se_wall,
            "surf_epi_pts": sp_pts, "surf_epi_n": sp_n,
            "near_pts": near, "free_pts": free,
            "query_pts": q_pts, "query_e_sdf": q_e, "query_p_sdf": q_p,
            "scale": np.float32(d["scale"]),
        }


def _augment_contour(contour: np.ndarray, cfg: dict, rng) -> np.ndarray:
    """Acquisition-style perturbation of the encoder input only."""
    out = contour.copy()
    xyz = out[:, :3]
    if cfg.get("aug_rotate_max_deg", 0.0) > 0 and rng.random() < cfg.get("aug_rotate_prob", 0.0):
        a = np.deg2rad(rng.uniform(-cfg["aug_rotate_max_deg"], cfg["aug_rotate_max_deg"]))
        c, s = np.cos(a), np.sin(a)
        xyz[:, :2] = xyz[:, :2] @ np.array([[c, -s], [s, c]], np.float32).T
    if cfg.get("aug_translate_xy_std", 0.0) > 0:
        xyz[:, :2] += rng.normal(0.0, cfg["aug_translate_xy_std"], 2).astype(np.float32)
    if cfg.get("aug_scale_std", 0.0) > 0:
        xyz *= np.float32(1.0 + rng.normal(0.0, cfg["aug_scale_std"]))
    if cfg.get("aug_jitter_std", 0.0) > 0:
        xyz += rng.normal(0.0, cfg["aug_jitter_std"], xyz.shape).astype(np.float32)
    out[:, :3] = xyz
    drop = cfg.get("aug_slice_drop_prob", 0.0)
    if drop > 0:
        z_levels = np.unique(np.round(out[:, 2], 5))
        keep = z_levels[rng.random(len(z_levels)) >= drop]
        if len(keep) >= 2:
            out = out[np.isin(np.round(out[:, 2], 5), keep)]
    return out


def collate(batch: list[dict]) -> dict:
    """Pad the variable-length contour, stack everything else."""
    import torch

    n_max = max(len(b["contour"]) for b in batch)
    dim = batch[0]["contour"].shape[1]
    contour = np.zeros((len(batch), n_max, dim), np.float32)
    mask = np.zeros((len(batch), n_max), bool)
    for i, b in enumerate(batch):
        n = len(b["contour"])
        contour[i, :n] = b["contour"]
        mask[i, :n] = True

    out = {"contour": torch.from_numpy(contour),
           "contour_mask": torch.from_numpy(mask)}
    for key in ("surf_endo_pts", "surf_endo_n", "surf_endo_wall", "surf_epi_pts",
                "surf_epi_n", "near_pts", "free_pts", "query_pts", "query_e_sdf",
                "query_p_sdf", "scale"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]).astype(np.float32))
    return out


def balanced_weights(specs: list[SampleSpec]) -> np.ndarray:
    """Sampling weights that equalise every (group, phase) stratum."""
    keys = [(s.group, s.phase) for s in specs]
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return np.array([1.0 / counts[k] for k in keys], np.float64)


def cache_manifest(cache_dir: Path) -> dict:
    specs = index_cache(cache_dir)
    manifest: dict = {"n_samples": len(specs), "strata": {}}
    for spec in specs:
        key = f"{spec.group}/{spec.phase}"
        manifest["strata"][key] = manifest["strata"].get(key, 0) + 1
    (Path(cache_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
