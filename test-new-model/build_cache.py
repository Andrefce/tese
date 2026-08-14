"""Build the local ACDC training cache (one entry per patient and phase).

    python build_cache.py --data-root training --out cache --workers 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardiosdf2 import data as D  # noqa: E402


def _job(args):
    patient_dir, info, phase, out_dir, force = args
    try:
        path = D.build_sample(Path(patient_dir), info, phase, Path(out_dir), force)
        return (True, f"{Path(patient_dir).name}_{phase}", str(path))
    except Exception as exc:                                # noqa: BLE001
        return (False, f"{Path(patient_dir).name}_{phase}",
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=here / "training")
    ap.add_argument("--out", type=Path, default=here / "cache")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    patients = D.discover_patients(args.data_root)
    if args.limit:
        patients = patients[:args.limit]
    jobs = [(str(p), info, phase, str(args.out), args.force)
            for p, info in patients for phase in D.PHASES]
    print(f"{len(patients)} patients -> {len(jobs)} samples", flush=True)

    t0 = time.time()
    failures = []
    if args.workers <= 1:
        results = map(_job, jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(_job, jobs, chunksize=1)

    for i, (ok, tag, msg) in enumerate(results, start=1):
        if not ok:
            failures.append((tag, msg))
            print(f"  [{i}/{len(jobs)}] FAIL {tag}: {msg.splitlines()[0]}", flush=True)
        elif i % 10 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] {tag}  ({time.time() - t0:.0f}s)", flush=True)

    manifest = D.cache_manifest(args.out)
    print(f"\ndone in {time.time() - t0:.0f}s — {manifest['n_samples']} cached, "
          f"{len(failures)} failed")
    for key in sorted(manifest["strata"]):
        print(f"  {key:10s} {manifest['strata'][key]}")
    if failures:
        print("\nfailures:")
        for tag, msg in failures:
            print(f"  {tag}: {msg.splitlines()[0]}")


if __name__ == "__main__":
    main()
