"""Export top-6 PV-anomaly tracks to CSV for Lagrangian compositing.

Columns: lc, track_id, time_iso, day_hour, lon, lat, pv_anom_330,
         area_deg2
    area_deg2 = local-radius (10°) area where pv_anom_330 >= 0.2 PVU,
    diagnostic only.

Output: outputs/<lc>/tracks/track_centers.csv
"""
from __future__ import annotations
import csv
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from track_utils import parse_stitchnodes  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")


def local_area_deg2(vor_frame, thr, lat0, lon0, lat1d, lon1d,
                     radius_deg=10.0):
    dlat = abs(lat1d[1] - lat1d[0])
    dlon = abs(lon1d[1] - lon1d[0])
    dlon_grid = ((lon1d[None, :] - lon0 + 180.0) % 360.0) - 180.0
    dlat_grid = lat1d[:, None] - lat0
    cos_lat = np.cos(np.deg2rad(lat0))
    r2 = dlat_grid**2 + (dlon_grid * cos_lat) ** 2
    near = r2 <= radius_deg**2
    mask = (vor_frame >= thr) & near
    if not np.any(mask):
        return float("nan")
    cos_grid = np.cos(np.deg2rad(lat1d))[:, None]
    cell = dlat * dlon * cos_grid
    return float(np.broadcast_to(cell, mask.shape)[mask].sum())


def run(lc):
    out_dir = ROOT / lc
    pv = xr.open_dataset(out_dir / "pv330_anom.nc")["pv_anom_330"]
    if float(pv["lat"][0]) > float(pv["lat"][-1]):
        pv = pv.isel(lat=slice(None, None, -1))
    lat1d = pv["lat"].values
    lon1d = pv["lon"].values
    times = pv["time"].values

    src = out_dir / "tracks" / "tracks_max_top6_pv330.txt"
    tracks = parse_stitchnodes(src)

    THR = 0.2
    csv_path = out_dir / "tracks" / "track_centers.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lc", "track_id", "time_iso", "day_hour",
                    "lon", "lat", "pv_anom_330", "area_deg2"])
        for tid, tr in enumerate(tracks):
            for p in tr:
                t_np = np.datetime64(p["time"])
                dh = (p["time"] - datetime(2000, 1, 1)).total_seconds() / 3600.0
                ti = int(np.argmin(np.abs(times - t_np)))
                vf = pv.isel(time=ti).values
                a = local_area_deg2(vf, THR, p["lat"], p["lon"],
                                     lat1d, lon1d, radius_deg=10.0)
                w.writerow([lc, tid, p["time"].isoformat(), f"{dh:.1f}",
                            f"{p['lon']:.4f}", f"{p['lat']:.4f}",
                            f"{p['val']:.6e}", f"{a:.3f}"])
    print(f"[{lc}] wrote {csv_path}")


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        run(lc)
