"""
data/helpers.py
SSM loading + contour extraction + graph construction helpers.
"""

import gzip
import os

import numpy as np
import vtk
from sklearn.neighbors import NearestNeighbors


# ── SSM I/O ───────────────────────────────────────────────────────────────

def read_vtk_mesh(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    pts  = mesh.GetPoints().GetData()
    verts = np.array(
        [pts.GetTuple3(i) for i in range(pts.GetNumberOfTuples())],
        dtype=np.float32,
    )
    polys = mesh.GetPolys().GetData()
    faces, i = [], 0
    while i < polys.GetNumberOfValues():
        n = polys.GetValue(i)
        if n == 3:
            faces.append([polys.GetValue(i + 1), polys.GetValue(i + 2), polys.GetValue(i + 3)])
        i += n + 1
    return verts, np.array(faces, dtype=np.int64)


def load_csv_gz(path):
    with gzip.open(str(path), 'rt') as f:
        return np.loadtxt(f, delimiter=',')


def load_ssm(ssm_dir):
    """Return (MEAN_VERTS, FACES, PCS, VARIANCES, SCALED_PCS)."""
    print('Loading SSM...')
    mean_verts, faces = read_vtk_mesh(f'{ssm_dir}/LV_ED_mean.vtk')
    pcs               = load_csv_gz(f'{ssm_dir}/LV_ED_pc_100_modes.csv.gz')
    variances         = load_csv_gz(f'{ssm_dir}/LV_ED_var_100_modes.csv.gz').ravel()
    scaled_pcs        = pcs * np.sqrt(variances)   # (3V, 100)
    print(f'  Mean mesh : {len(mean_verts):,} vertices | {len(faces):,} faces')
    print(f'  PCA modes : {pcs.shape[1]}')
    return mean_verts, faces, pcs, variances, scaled_pcs


def sample_shape(rng, pcs, scaled_pcs, mean_verts, num_modes=10, sigma_clip=3.0):
    w = np.zeros(pcs.shape[1])
    w[:num_modes] = np.clip(rng.standard_normal(num_modes), -sigma_clip, sigma_clip)
    disp  = scaled_pcs @ w
    verts = (mean_verts.flatten() + disp).reshape(-1, 3).astype(np.float32)
    return verts, w[:num_modes].astype(np.float32)


def weights_to_mesh(weights, pcs, scaled_pcs, mean_verts, faces):
    """Reconstruct full SSM mesh from PCA weight vector."""
    w_full = np.zeros(pcs.shape[1])
    w_full[:len(weights)] = weights
    disp  = scaled_pcs @ w_full
    verts = (mean_verts.flatten() + disp).reshape(-1, 3).astype(np.float32)
    return verts, faces


# ── Contour helpers ───────────────────────────────────────────────────────

def separate_endo_epi(pts2d, num_bins=80, min_wall_mm=4.0):
    if len(pts2d) < 6:
        return np.empty((0, 2)), np.empty((0, 2))
    c = pts2d.mean(0)
    a = np.arctan2(pts2d[:, 1] - c[1], pts2d[:, 0] - c[0])
    d = np.linalg.norm(pts2d - c, axis=1)
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    endo, epi, walls = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (a >= lo) & (a < hi)
        if not m.any():
            continue
        bp, bd = pts2d[m], d[m]
        if len(bd) > 1:
            walls.append(bd.max() - bd.min())
            endo.append(bp[bd.argmin()])
            epi.append(bp[bd.argmax()])
        else:
            epi.append(bp[0])
    if not walls or np.median(walls) < min_wall_mm:
        return np.empty((0, 2)), np.empty((0, 2))

    def order(pts):
        if not pts:
            return np.empty((0, 2))
        arr = np.array(pts)
        ang = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
        return arr[np.argsort(ang)]

    return order(endo), order(epi)


def resample_contour(cont, n):
    if len(cont) < 2:
        return np.zeros((n, 2), dtype=np.float32)
    d   = np.diff(cont, axis=0, prepend=cont[-1:])
    arc = np.cumsum(np.linalg.norm(d, axis=1))
    arc /= arc[-1]
    t = np.linspace(0, 1, n)
    return np.column_stack(
        [np.interp(t, arc, cont[:, 0]), np.interp(t, arc, cont[:, 1])]
    ).astype(np.float32)


def extract_slices(verts, num_slices=20, n_pts=50, eps=2.0, z_trim=0.10):
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    z_range = z_max - z_min
    z_vals  = np.linspace(z_min + z_trim * z_range, z_max - z_trim * z_range, num_slices)
    out = []
    for z in z_vals:
        mask = np.abs(verts[:, 2] - z) < eps
        pts  = verts[mask, :2]
        if len(pts) < 10:
            continue
        endo, epi = separate_endo_epi(pts)
        if len(endo) < 4 or len(epi) < 4:
            continue
        out.append(dict(
            z        = float(z),
            endo     = resample_contour(endo, n_pts),
            epi      = resample_contour(epi,  n_pts),
            centroid = pts.mean(0).astype(np.float32),
        ))
    return out


def augment_contour(pts, noise_std=1.5, drop_prob=0.1):
    pts  = pts + np.random.randn(*pts.shape) * noise_std
    mask = np.random.rand(len(pts)) > drop_prob
    return pts[mask] if mask.sum() > 5 else pts


# ── Graph construction ────────────────────────────────────────────────────

def build_graph(slices, knn_intra=8, knn_inter=3):
    all_nodes, ntypes, sids = [], [], []
    for sid, sl in enumerate(slices):
        c = sl['centroid']
        for cont, tt in [(sl['endo'], 0), (sl['epi'], 1)]:
            n = len(cont)
            z = np.full((n, 1), sl['z'], dtype=np.float32)
            r = np.linalg.norm(cont - c, axis=1, keepdims=True).astype(np.float32)
            t = np.full((n, 1), tt, dtype=np.float32)
            all_nodes.append(np.hstack([cont, z, r, t]))
            ntypes.append(np.full(n, tt, dtype=np.int8))
            sids.append(np.full(n, sid, dtype=np.int32))

    if not all_nodes:
        return None

    nodes   = np.vstack(all_nodes).astype(np.float32)
    nt_arr  = np.concatenate(ntypes)
    sid_arr = np.concatenate(sids)
    edges, edge_feats = [], []

    for sid in range(len(slices)):
        idx = np.where(sid_arr == sid)[0]
        if len(idx) < 2:
            continue
        k = min(knn_intra + 1, len(idx))
        _, nb_ = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(
            nodes[idx, :3]
        ).kneighbors(nodes[idx, :3])
        for i, row in enumerate(nb_):
            for j in row[1:]:
                for s, t in [(idx[i], idx[j]), (idx[j], idx[i])]:
                    edges.append([s, t])
                    diff = nodes[t, :3] - nodes[s, :3]
                    dist = np.linalg.norm(diff)
                    edge_feats.append(np.append(diff / (dist + 1e-8), [dist, nodes[s, 4], nodes[t, 4]]))

    for sid in range(len(slices) - 1):
        ia = np.where(sid_arr == sid)[0]
        ib = np.where(sid_arr == sid + 1)[0]
        if not len(ia) or not len(ib):
            continue
        k = min(knn_inter, len(ib))
        _, nb_ = NearestNeighbors(n_neighbors=k).fit(nodes[ib, :3]).kneighbors(nodes[ia, :3])
        for i, row in enumerate(nb_):
            for j in row:
                for s, t in [(ia[i], ib[j]), (ib[j], ia[i])]:
                    edges.append([s, t])
                    diff = nodes[t, :3] - nodes[s, :3]
                    dist = np.linalg.norm(diff)
                    edge_feats.append(np.append(diff / (dist + 1e-8), [dist, nodes[s, 4], nodes[t, 4]]))

    edges      = np.array(edges,      dtype=np.int32)   if edges      else np.empty((0, 2), dtype=np.int32)
    edge_feats = np.array(edge_feats, dtype=np.float32) if edge_feats else np.empty((0, 6), dtype=np.float32)
    return dict(
        nodes      = nodes,
        edges      = edges,
        edge_feats = edge_feats,
        node_types = nt_arr,
        slice_ids  = sid_arr,
        num_nodes  = len(nodes),
        num_edges  = len(edges),
        num_slices = len(slices),
    )


def nodes_to_surface(nodes, node_types, slice_ids):
    unique_sids = np.unique(slice_ids)
    all_verts, all_faces = [], []
    for ttype in [0, 1]:
        type_mask = node_types == ttype
        rings = []
        for sid in unique_sids:
            m   = type_mask & (slice_ids == sid)
            pts = nodes[m]
            if len(pts) < 3:
                continue
            c   = pts[:, :2].mean(0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            rings.append(pts[np.argsort(ang)])
        if len(rings) < 2:
            continue
        for ri in range(len(rings) - 1):
            r0, r1  = rings[ri], rings[ri + 1]
            n0, n1  = len(r0), len(r1)
            base    = len(all_verts)
            all_verts.extend(r0.tolist())
            all_verts.extend(r1.tolist())
            for j in range(n0):
                a  = base + j
                b  = base + (j + 1) % n0
                c0 = base + n0 + (j * n1 // n0) % n1
                c1 = base + n0 + ((j + 1) * n1 // n0) % n1
                all_faces.append([a, b, c0])
                all_faces.append([b, c1, c0])

    if not all_verts or not all_faces:
        return None, None
    return np.array(all_verts, dtype=np.float32), np.array(all_faces, dtype=np.int64)
