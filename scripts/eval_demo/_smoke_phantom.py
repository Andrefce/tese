"""Quick smoke test: one phantom, all methods."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import phantoms as ph
import thickness as tk
from geometry import outward_normals

pitch = 0.75
phantom = ph.concentric_sphere(r_endo=22.0, t=8.0, subdivisions=3)
endo, epi = phantom.endo, phantom.epi
print("endo", len(endo.vertices), "watertight", endo.is_watertight)
print("epi ", len(epi.vertices), "watertight", epi.is_watertight)

normals = outward_normals(endo, np.asarray(epi.vertices))
t0 = time.perf_counter()
ctx = tk.build_volume_context(endo, epi, pitch)
print(f"volume ctx in {time.perf_counter()-t0:.1f}s  shape={ctx.myo_mask.shape} "
      f"endo={ctx.endo_mask.sum()} epi={ctx.epi_mask.sum()} myo={ctx.myo_mask.sum()}")

t0 = time.perf_counter()
phi, diag = tk.solve_laplace(ctx)
print(f"laplace in {time.perf_counter()-t0:.1f}s", diag,
      "phi range", np.nanmin(phi), np.nanmax(phi))

V = np.asarray(endo.vertices)
for res in [
    tk.method_laplace_streamline(ctx, V, normals, phi),
    tk.method_laplace_gradient(ctx, V, normals, phi),
    tk.method_edt_boundary_sum(ctx, V, normals),
    tk.method_sphere_propagation(ctx, V, normals),
    tk.method_surface_correspondence(endo, epi, normals),
    tk.method_cone_rays(endo, epi, normals),
]:
    m = ph.error_metrics(res.values, phantom.true_thickness, res.runtime_s, phantom.valid)
    print(f"{res.name:36s} {m}")
    if res.diagnostics:
        print("     diag:", res.diagnostics)
