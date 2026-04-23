"""Export top-N tracks to CSV (per LC, per method, per polarity).

Reads:  outputs/<lc>/tracks/<method>/tracks_{C,AC}_top6.txt
        outputs/<lc>/<method input_nc>
Writes: outputs/<lc>/tracks/<method>/track_centers_{C,AC}.csv
"""
from __future__ import annotations
import csv
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from track_utils import parse_stitchnodes  # noqa
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")


def local_area_deg2(field, op, thr, lat0, lon0, lat1d, lon1d,
                    radius_deg=10.0):
    dlat = abs(lat1d[1] - lat1d[0])
    dlon = abs(lon1d[1] - lon1d[0])
    dlon_grid = ((lon1d[None, :] - lon0 + 180.0) % 360.0) - 180.0
    dlat_grid = lat1d[:, None] - lat0
    cos_lat = np.cos(np.deg2rad(lat0))
    near = (dlat_grid ** 2 + (dlon_grid * cos_lat) ** 2) <= radius_deg ** 2
    if op == ">":
        sig = (field >= thr) & near
    else:  # "<"
        sig = (field <= thr) & near
    if not np.any(sig):
        return float("nan")
    cell = dlat * dlon * np.cos(np.deg2rad(lat1d))[:, None]
    return float(np.broadcast_to(cell, sig.shape)[sig].sum())


def run(lc, method, polarity="C"):
    spec = CFG.METHOD[method]
    out_dir = ROOT / lc
    field = xr.open_dataset(out_dir / spec["input_nc"])[spec["var"]]
    if float(field["lat"][0]) > float(field["lat"][-1]):
        field = field.isel(lat=slice(None, None, -1))
    lat1d = field["lat"].values
    lon1d = field["lon"].values
    times = field["time"].values

    src = out_dir / "tracks" / method / f"tracks_{polarity}_top6.txt"
    if not src.exists():
        print(f"[{lc}:{method}:{polarity}] missing {src}; skipping")
        return
    tracks = parse_stitchnodes(src)
    op, thr = spec[polarity]["thresh"]

    csv_path = out_dir / "tracks" / method / f"track_centers_{polarity}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lc", "method", "track_id", "time_iso", "day_hour",
                    "lon", "lat", "val", "area_deg2"])
        for tid, tr in enumerate(tracks):
            for p in tr:
                t_np = np.datetime64(p["time"])
                dh = (p["time"] - datetime(2000, 1, 1)).total_seconds() / 3600.0
                ti = int(np.argmin(np.abs(times - t_np)))
                vf = field.isel(time=ti).values
                a = local_area_deg2(vf, op, thr, p["lat"], p["lon"],
                                    lat1d, lon1d, radius_deg=10.0)
                w.writerow([lc, method, tid, p["time"].isoformat(),
                            f"{dh:.1f}", f"{p['lon']:.4f}",
                            f"{p['lat']:.4f}", f"{p['val']:.6e}",
                            f"{a:.3f}"])
    print(f"[{lc}:{method}:{polarity}] wrote {csv_path}  ({len(tracks)} tracks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--method", default=None)
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    methods = [args.method] if args.method else CFG.METHODS
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for m in methods:
            for pol in pols:
                run(lc, m, polarity=pol)
