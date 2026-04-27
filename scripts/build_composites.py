"""Track-centred Lagrangian composites per LC, per method, per polarity.

For the chosen ``method``, the composite carries:
  total       : the method's total field   (e.g. pv330, zeta_250, theta_pv2)
  anom        : the method's anomaly       (e.g. pv_anom_330, zeta_anom_250,
                                              theta_anom_pv2)

A patch of (2*PATCH_HALF_LAT+1) rows × (2*PATCH_HALF_LON+1) cols at DX° spacing follows the
tracked centre (lon, lat) at every hour the track exists.

Reads:  outputs/<lc>/tracks/<method>/track_centers_{C,AC}.csv
        outputs/<lc>/<method input_nc>
Writes: outputs/<lc>/composites/<method>/{C,AC}_composite.nc
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

ROOT = Path("/net/flood/data2/users/x_yan/literature_review/rwb/thorncroft93_baroclinic/"
            "thorncroft_rwb/outputs")


def _win_start(lc: str):
    h0, _ = CFG.window_for(lc)
    return datetime(2000, 1, 1) + pd.Timedelta(hours=h0)


def _sort_lat(da):
    if float(da["lat"][0]) > float(da["lat"][-1]):
        da = da.isel(lat=slice(None, None, -1))
    return da


def _pad_periodic_lon(da: xr.DataArray) -> xr.DataArray:
    """Extend the DataArray along lon so it is a valid periodic grid for
    xarray linear interpolation across the 0/360 seam.

    Prepends one point at lon=lon[-1]-360 and appends one at lon[0]+360.
    """
    lon = da["lon"].values
    # Assume monotonic increasing after _sort_lat; longitudes may be 0..359.
    left = da.isel(lon=[-1]).assign_coords(lon=[float(lon[-1]) - 360.0])
    right = da.isel(lon=[0]).assign_coords(lon=[float(lon[0]) + 360.0])
    return xr.concat([left, da, right], dim="lon")


def _pole_reflect_coords(lat_q: np.ndarray, lon_q: np.ndarray):
    """Reflect latitudes that overshoot the poles.

    Going past the North Pole by delta along longitude L is the same as
    being at (90-delta) along longitude (L+180). Symmetric for south
    pole. Returns (lat_q_ref, lon_q_ref) as 1-D arrays.
    """
    lat_r = lat_q.copy()
    lon_r = lon_q.copy()
    over_n = lat_r > 90.0
    lat_r[over_n] = 180.0 - lat_r[over_n]
    lon_r[over_n] = (lon_r[over_n] + 180.0) % 360.0
    over_s = lat_r < -90.0
    lat_r[over_s] = -180.0 - lat_r[over_s]
    lon_r[over_s] = (lon_r[over_s] + 180.0) % 360.0
    return lat_r, lon_r


def build(lc: str, method: str, polarity: str = "C"):
    spec = CFG.METHOD[method]
    out_dir = ROOT / lc
    in_ds = xr.open_dataset(out_dir / spec["input_nc"])
    total = _sort_lat(in_ds[spec["total_var"]])
    anom = _sort_lat(in_ds[spec["var"]])
    times = total["time"].values

    csv_path = out_dir / "tracks" / method / f"track_centers_{polarity}.csv"
    if not csv_path.exists():
        print(f"[{lc}:{method}:{polarity}] missing {csv_path}; skipping")
        return
    csv = pd.read_csv(csv_path, parse_dates=["time_iso"])
    win_start = _win_start(lc)

    x_rel = np.arange(-CFG.PATCH_HALF_LON, CFG.PATCH_HALF_LON + CFG.DX / 2, CFG.DX)
    y_rel = np.arange(-CFG.PATCH_HALF_LAT, CFG.PATCH_HALF_LAT + CFG.DX / 2, CFG.DX)
    nx, ny = len(x_rel), len(y_rel)

    n_members = CFG.TOP_N
    n_hours = CFG.N_COMPOSITE_HOURS
    t_hour = np.arange(n_hours, dtype="int16")
    shape_m = (n_members, n_hours, ny, nx)
    members_total = np.full(shape_m, np.nan, dtype="float32")
    members_anom = np.full(shape_m, np.nan, dtype="float32")
    track_lon = np.full((n_members, n_hours), np.nan, dtype="float32")
    track_lat = np.full((n_members, n_hours), np.nan, dtype="float32")

    # Pre-extend source grids periodically in lon so we can interpolate
    # across the 0/360 seam without NaNs.
    total_pad = _pad_periodic_lon(total)
    anom_pad = _pad_periodic_lon(anom)

    # 2-D sample grid relative to the track centre.
    # (n_y, n_x): row = lat offset, col = lon offset.
    Xrel2d, Yrel2d = np.meshgrid(x_rel, y_rel)

    for tid, grp in csv.groupby("track_id"):
        tid_i = int(tid)
        if tid_i >= n_members:
            continue
        grp = grp.sort_values("time_iso").reset_index(drop=True)
        for _, row in grp.iterrows():
            t_target = pd.to_datetime(row["time_iso"])
            ihour = int((t_target - win_start).total_seconds() / 3600)
            if ihour < 0 or ihour >= n_hours:
                continue
            lat_samp = float(row["lat"]) + Yrel2d             # (ny, nx)
            lon_samp = (float(row["lon"]) + Xrel2d) % 360.0
            lat_samp_ref, lon_samp_ref = _pole_reflect_coords(
                lat_samp.ravel(), lon_samp.ravel())
            lat_da = xr.DataArray(lat_samp_ref, dims="p")
            lon_da = xr.DataArray(lon_samp_ref, dims="p")
            ti = int(np.argmin(np.abs(times - np.datetime64(t_target))))
            try:
                patch_total = total_pad.isel(time=ti).interp(
                    lat=lat_da, lon=lon_da, method="linear"
                ).values.reshape(ny, nx).astype("float32")
                patch_anom = anom_pad.isel(time=ti).interp(
                    lat=lat_da, lon=lon_da, method="linear"
                ).values.reshape(ny, nx).astype("float32")
            except Exception as exc:
                print(f"  [{lc}:{method}:{polarity} tid={tid_i} "
                      f"ihour={ihour}] interp failed: {exc}")
                continue
            members_total[tid_i, ihour] = patch_total
            members_anom[tid_i, ihour] = patch_anom
            track_lon[tid_i, ihour] = float(row["lon"])
            track_lat[tid_i, ihour] = float(row["lat"])

    comp_total = np.nanmean(members_total, axis=0).astype("float32")
    comp_anom = np.nanmean(members_anom, axis=0).astype("float32")
    n_per_t = np.sum(~np.isnan(members_total[..., 0, 0]),
                     axis=0).astype("int8")

    ds = xr.Dataset(
        {
            "total_composite": (("t", "y", "x"), comp_total),
            "anom_composite": (("t", "y", "x"), comp_anom),
            "total_members": (("member", "t", "y", "x"), members_total),
            "anom_members": (("member", "t", "y", "x"), members_anom),
            "track_lon": (("member", "t"), track_lon),
            "track_lat": (("member", "t"), track_lat),
            "n_members": (("t",), n_per_t),
        },
        coords={
            "t": ("t", t_hour),
            "y": ("y", y_rel.astype("float32")),
            "x": ("x", x_rel.astype("float32")),
            "member": ("member", np.arange(n_members, dtype="int8")),
        },
        attrs={
            "lc": lc,
            "method": method,
            "polarity": polarity,
            "total_var": spec["total_var"],
            "anom_var": spec["var"],
            "units": spec["units"],
            "patch_half_lon_deg": float(CFG.PATCH_HALF_LON),
            "patch_half_lat_deg": float(CFG.PATCH_HALF_LAT),
            "dx_deg": float(CFG.DX),
            "frame": ("track-centred Lagrangian patch: x = lon - "
                      "lon_track, y = lat - lat_track"),
            "time_origin": (f"hours since {win_start.isoformat()} "
                            f"(LC window start)"),
        },
    )
    comp_dir = out_dir / "composites" / method
    comp_dir.mkdir(parents=True, exist_ok=True)
    out = comp_dir / f"{polarity}_composite.nc"
    ds.to_netcdf(out)
    print(f"[{lc}:{method}:{polarity}] wrote {out}  shape={comp_total.shape} "
          f"n_per_t {n_per_t.min()}..{n_per_t.max()}")


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
                build(lc, m, polarity=pol)
