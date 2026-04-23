"""Lagrangian composite on 2 PVU, theta=330 K PV, and PV-anom-330 patches.

For each of 6 PV-anom tracks (per LC), at every hour the track
exists (variable span, >=90 h), patches are centered on the
track's own (lon(t), lat(t)) -- fully Lagrangian.  Patch is
40 deg x 40 deg at 1.0 deg resolution -> 41 x 41 grid.

Output variables (t, y, x):
    pv_composite        PV on theta=330 K (member-mean)      [PVU]
    theta_pv2_composite theta on the 2 PVU surface (member-mean) [K]
    pv_anom_composite   PV anomaly on 330 K (member-mean)    [PVU]
Plus per-member arrays and track coords.  t = 0..144 hours since
2000-01-07 00 UTC; tracks covering a sub-range leave NaNs for the
unsampled hours.
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")

PATCH_HALF = 20.0   # degrees -> 41 x 41 grid
DX = 1.0
THETA_LEVEL = 330.0
WIN_START = datetime(2000, 1, 7, 0)
N_HOURS = 145


def _sort_lat(da):
    if float(da["lat"][0]) > float(da["lat"][-1]):
        da = da.isel(lat=slice(None, None, -1))
    return da


def build(lc: str):
    out_dir = ROOT / lc
    proc = xr.open_dataset(out_dir / "processed.nc")
    pv_theta = _sort_lat(proc["pv_on_theta"].sel(theta=THETA_LEVEL))
    th_pv2 = _sort_lat(proc["theta_on_pv2"])
    pv_anom = _sort_lat(xr.open_dataset(out_dir / "pv330_anom.nc")
                        ["pv_anom_330"])
    times_pv = pv_theta["time"].values
    times_a = pv_anom["time"].values

    csv = pd.read_csv(out_dir / "tracks" / "track_centers.csv",
                      parse_dates=["time_iso"])

    x_rel = np.arange(-PATCH_HALF, PATCH_HALF + DX / 2, DX)  # 41 pts
    y_rel = np.arange(-PATCH_HALF, PATCH_HALF + DX / 2, DX)
    nx, ny = len(x_rel), len(y_rel)

    t_hour = np.arange(N_HOURS, dtype="int16")

    shape_m = (6, N_HOURS, ny, nx)
    members_pv = np.full(shape_m, np.nan, dtype="float32")
    members_th = np.full(shape_m, np.nan, dtype="float32")
    members_anom = np.full(shape_m, np.nan, dtype="float32")
    track_lon = np.full((6, N_HOURS), np.nan, dtype="float32")
    track_lat = np.full((6, N_HOURS), np.nan, dtype="float32")

    for tid, grp in csv.groupby("track_id"):
        grp = grp.sort_values("time_iso").reset_index(drop=True)
        for _, row in grp.iterrows():
            t_target = pd.to_datetime(row["time_iso"])
            ihour = int((t_target - WIN_START).total_seconds() / 3600)
            if ihour < 0 or ihour >= N_HOURS:
                continue
            lat_q = xr.DataArray(row["lat"] + y_rel, dims="y")
            lon_q = xr.DataArray((row["lon"] + x_rel) % 360.0, dims="x")
            ti_pv = int(np.argmin(np.abs(times_pv -
                                          np.datetime64(t_target))))
            ti_a = int(np.argmin(np.abs(times_a -
                                         np.datetime64(t_target))))
            members_pv[int(tid), ihour] = pv_theta.isel(time=ti_pv).interp(
                lat=lat_q, lon=lon_q, method="linear"
            ).values.astype("float32")
            members_th[int(tid), ihour] = th_pv2.isel(time=ti_pv).interp(
                lat=lat_q, lon=lon_q, method="linear"
            ).values.astype("float32")
            members_anom[int(tid), ihour] = pv_anom.isel(time=ti_a).interp(
                lat=lat_q, lon=lon_q, method="linear"
            ).values.astype("float32")
            track_lon[int(tid), ihour] = float(row["lon"])
            track_lat[int(tid), ihour] = float(row["lat"])

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
            "member": ("member", np.arange(6, dtype="int8")),
        },
        attrs={
            "lc": lc, "polarity": "cyclone",
            "theta_K": float(THETA_LEVEL),
            "patch_half_deg": float(PATCH_HALF),
            "dx_deg": float(DX),
            "frame": "Lagrangian (patch center moves with tracked cyclone)",
            "time_origin": "hours since 2000-01-07T00:00:00",
        },
    )
    comp_dir = out_dir / "composites"
    comp_dir.mkdir(exist_ok=True)
    out = comp_dir / "C_composite.nc"
    ds.to_netcdf(out)
    print(f"[{lc}] wrote {out}  shape={comp_pv.shape}  "
          f"n_per_t min/max={comp_count.min()}/{comp_count.max()} "
          f"frames with >=1 member: {(comp_count > 0).sum()}")


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        build(lc)
