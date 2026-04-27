#!/usr/bin/env python
"""Build per-LC, per-method TempestExtremes input NetCDFs.

For each ``lc`` reads ``outputs/<lc>/processed.nc`` and writes three
day-6..10 anomaly files (97 hourly frames each) used by
``run_tempest.sh``:

  outputs/<lc>/pv330_anom.nc       — pv330,    pv_anom_330       [PVU]
  outputs/<lc>/zeta250_anom.nc     — zeta_250, zeta_anom_250     [s^-1]
  outputs/<lc>/theta_pv2_anom.nc   — theta_pv2, theta_anom_pv2   [K]

All arrays are flipped S->N, NaN-filled with the zonal mean (so
TempestExtremes sees finite values everywhere), and clipped to the
day-6..10 window (hours 144..240 inclusive).

Anomaly := raw minus instantaneous zonal mean (no smoothing).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config as CFG
from _config import THETA_LEVEL, PRESSURE_LEVEL_ZETA

ROOT = Path("/net/flood/data2/users/x_yan/literature_review/rwb/thorncroft93_baroclinic/"
            "thorncroft_rwb")


def _flip_lat(da: xr.DataArray) -> xr.DataArray:
    if float(da["lat"][0]) > float(da["lat"][-1]):
        return da.isel(lat=slice(None, None, -1))
    return da


def _time_subset(da: xr.DataArray, lc: str) -> xr.DataArray:
    t0 = da["time"].values[0]
    hours = (da["time"].values - t0) / np.timedelta64(1, "h")
    h0, h1 = CFG.window_for(lc)
    mask = (hours >= h0) & (hours <= h1)
    return da.isel(time=np.where(mask)[0])


def _make_anom(raw: xr.DataArray, name_total: str, name_anom: str,
               long_name: str, units: str, lc: str) -> xr.Dataset:
    raw = _flip_lat(raw)
    raw = _time_subset(raw, lc)
    zm = raw.mean("lon", skipna=True)
    anom = raw - zm
    raw_filled = raw.fillna(zm)
    anom_filled = anom.fillna(0.0)
    out = xr.Dataset({
        name_total: raw_filled.astype(np.float32).rename(name_total),
        name_anom:  anom_filled.astype(np.float32).rename(name_anom),
    })
    out[name_total].attrs.update(long_name=long_name, units=units)
    out[name_anom].attrs.update(
        long_name=f"{long_name} anomaly (zonal mean removed)", units=units)
    return out


def _write(out: xr.Dataset, dst: Path) -> None:
    enc = {
        "time": {"units": "hours since 2000-01-01 00:00:00",
                 "calendar": "standard", "dtype": "float64"},
    }
    for v in out.data_vars:
        enc[v] = {"zlib": True, "complevel": 4}
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(dst, encoding=enc)


def prep(lc: str) -> None:
    ds = xr.open_dataset(ROOT / "outputs" / lc / "processed.nc")
    out_dir = ROOT / "outputs" / lc

    # -------- pv330 ----------------------------------------------------
    pv330 = ds["pv_on_theta"].sel(theta=THETA_LEVEL).drop_vars(
        "theta", errors="ignore")
    out_pv = _make_anom(pv330, "pv330", "pv_anom_330",
                        f"Ertel PV on theta={THETA_LEVEL:.0f} K", "PVU",
                        lc)
    dst = out_dir / "pv330_anom.nc"
    _write(out_pv, dst)
    print(f"[{lc}] pv330 -> {dst}  n_t={out_pv.sizes['time']}")

    # -------- zeta250 (interp to 250 hPa from plev grid) ---------------
    vor = ds["vor_p"]
    plev = vor["plev"].values
    if PRESSURE_LEVEL_ZETA in plev:
        zeta = vor.sel(plev=PRESSURE_LEVEL_ZETA).drop_vars("plev",
                                                            errors="ignore")
    else:
        # Linear interp in pressure; processed.nc plev is monotonic.
        zeta = vor.interp(plev=PRESSURE_LEVEL_ZETA).drop_vars(
            "plev", errors="ignore")
    out_z = _make_anom(zeta, "zeta_250", "zeta_anom_250",
                       "relative vorticity at 250 hPa", "1/s", lc)
    dst = out_dir / "zeta250_anom.nc"
    _write(out_z, dst)
    print(f"[{lc}] zeta250 -> {dst}  n_t={out_z.sizes['time']}")

    # -------- theta on 2 PVU ------------------------------------------
    th = ds["theta_on_pv2"]
    out_t = _make_anom(th, "theta_pv2", "theta_anom_pv2",
                       "potential temperature on +/-2 PVU", "K", lc)
    dst = out_dir / "theta_pv2_anom.nc"
    _write(out_t, dst)
    print(f"[{lc}] theta_pv2 -> {dst}  n_t={out_t.sizes['time']}")


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        prep(lc)
