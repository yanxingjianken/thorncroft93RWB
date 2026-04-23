"""Fixed-latitude Lagrangian composite on 2 PVU, theta=330 K PV, and PV-anom-330 patches.

For each of 6 tracks (per LC, per polarity), at every hour the track
exists, patches follow the **tracked longitude** but sit on a
**fixed absolute latitude band** centred on ``CENTER_LAT = 55 N``:

    lon_q  = (track_lon(t) + x_rel) mod 360    x_rel in [-20..+20]°
    lat_q  =  CENTER_LAT    + y_rel              y_rel in [-20..+20]°

At 1° resolution this gives a 41 × 41 patch spanning absolute
latitudes 35..75 °N.  Tracked features near the edges of that band
stay inside the patch and contribute their structure to the
composite (no cos φ reweighting).

Polarity-specific outputs (per LC):
    composites/C_composite.nc   (cyclonic tracks)
    composites/AC_composite.nc  (anticyclonic tracks)
"""
from __future__ import annotations
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")

WIN_START = datetime(2000, 1, 7, 0)   # hour 0 of the 145-hour composite


def _sort_lat(da):
    if float(da["lat"][0]) > float(da["lat"][-1]):
        da = da.isel(lat=slice(None, None, -1))
    return da


def build(lc: str, polarity: str = "C"):
    out_dir = ROOT / lc
    proc = xr.open_dataset(out_dir / "processed.nc")
    pv_theta = _sort_lat(proc["pv_on_theta"].sel(theta=CFG.THETA_LEVEL))
    th_pv2 = _sort_lat(proc["theta_on_pv2"])
    pv_anom = _sort_lat(xr.open_dataset(out_dir / "pv330_anom.nc")
                        ["pv_anom_330"])
    times_pv = pv_theta["time"].values
    times_a = pv_anom["time"].values

    csv_path = out_dir / "tracks" / f"track_centers_{polarity}.csv"
    csv = pd.read_csv(csv_path, parse_dates=["time_iso"])

    x_rel = np.arange(-CFG.PATCH_HALF, CFG.PATCH_HALF + CFG.DX / 2, CFG.DX)
    y_rel = np.arange(-CFG.PATCH_HALF, CFG.PATCH_HALF + CFG.DX / 2, CFG.DX)
    nx, ny = len(x_rel), len(y_rel)
    assert nx == 41 and ny == 41, (nx, ny)

    # Fixed absolute-latitude band:  CENTER_LAT + y_rel  -> 35..75 N
    lat_abs = CFG.CENTER_LAT + y_rel
    lat_q = xr.DataArray(lat_abs, dims="y")

    n_members = 6
    t_hour = np.arange(CFG.N_COMPOSITE_HOURS, dtype="int16")
    shape_m = (n_members, CFG.N_COMPOSITE_HOURS, ny, nx)
    members_pv = np.full(shape_m, np.nan, dtype="float32")
    members_th = np.full(shape_m, np.nan, dtype="float32")
    members_anom = np.full(shape_m, np.nan, dtype="float32")
    track_lon = np.full((n_members, CFG.N_COMPOSITE_HOURS), np.nan,
                        dtype="float32")
    track_lat = np.full((n_members, CFG.N_COMPOSITE_HOURS), np.nan,
                        dtype="float32")

    for tid, grp in csv.groupby("track_id"):
        grp = grp.sort_values("time_iso").reset_index(drop=True)
        tid_i = int(tid)
        if tid_i >= n_members:
            continue
        for _, row in grp.iterrows():
            t_target = pd.to_datetime(row["time_iso"])
            ihour = int((t_target - WIN_START).total_seconds() / 3600)
            if ihour < 0 or ihour >= CFG.N_COMPOSITE_HOURS:
                continue
            lon_q = xr.DataArray((row["lon"] + x_rel) % 360.0, dims="x")
            ti_pv = int(np.argmin(np.abs(times_pv -
                                          np.datetime64(t_target))))
            ti_a = int(np.argmin(np.abs(times_a -
                                         np.datetime64(t_target))))
            members_pv[tid_i, ihour] = pv_theta.isel(time=ti_pv).interp(
                lat=lat_q, lon=lon_q, method="linear"
            ).values.astype("float32")
            members_th[tid_i, ihour] = th_pv2.isel(time=ti_pv).interp(
                lat=lat_q, lon=lon_q, method="linear"
            ).values.astype("float32")
            members_anom[tid_i, ihour] = pv_anom.isel(time=ti_a).interp(
                lat=lat_q, lon=lon_q, method="linear"
            ).values.astype("float32")
            track_lon[tid_i, ihour] = float(row["lon"])
            track_lat[tid_i, ihour] = float(row["lat"])

    comp_pv = np.nanmean(members_pv, axis=0).astype("float32")
    comp_th = np.nanmean(members_th, axis=0).astype("float32")
    comp_anom = np.nanmean(members_anom, axis=0).astype("float32")
    comp_count = np.sum(~np.isnan(members_pv[..., 0, 0]),
                         axis=0).astype("int8")

    ds = xr.Dataset(
        {
            "pv_composite": (("t", "y", "x"), comp_pv),
            "theta_pv2_composite": (("t", "y", "x"), comp_th),
            "pv_anom_composite": (("t", "y", "x"), comp_anom),
            "pv_members": (("member", "t", "y", "x"), members_pv),
            "theta_pv2_members": (("member", "t", "y", "x"), members_th),
            "pv_anom_members": (("member", "t", "y", "x"), members_anom),
            "track_lon": (("member", "t"), track_lon),
            "track_lat": (("member", "t"), track_lat),
            "n_members": (("t",), comp_count),
        },
        coords={
            "t": ("t", t_hour),
            "y": ("y", y_rel.astype("float32")),
            "x": ("x", x_rel.astype("float32")),
            "lat_abs": ("y", lat_abs.astype("float32")),
            "member": ("member", np.arange(n_members, dtype="int8")),
        },
        attrs={
            "lc": lc,
            "polarity": polarity,
            "theta_K": float(CFG.THETA_LEVEL),
            "patch_half_deg": float(CFG.PATCH_HALF),
            "dx_deg": float(CFG.DX),
            "center_lat": float(CFG.CENTER_LAT),
            "lat_fixed_range": f"{CFG.LAT_MIN:.0f}-{CFG.LAT_MAX:.0f} N",
            "frame": ("fixed-latitude (35..75 N), tracked-longitude "
                      "Lagrangian patch"),
            "time_origin": "hours since 2000-01-07T00:00:00",
        },
    )
    comp_dir = out_dir / "composites"
    comp_dir.mkdir(exist_ok=True)
    out = comp_dir / f"{polarity}_composite.nc"
    ds.to_netcdf(out)
    print(f"[{lc}:{polarity}] wrote {out}  shape={comp_pv.shape}  "
          f"n_per_t min/max={comp_count.min()}/{comp_count.max()} "
          f"frames with >=1 member: {(comp_count > 0).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for pol in pols:
            build(lc, polarity=pol)
