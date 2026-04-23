#!/usr/bin/env python
"""Prepare a TempestExtremes-friendly NetCDF of PV anomaly on the 330 K
isentrope for the Thorncroft LC runs.

Reads ``outputs/<lc>/processed.nc``, selects ``pv_on_theta`` at
θ=330 K, removes the instantaneous zonal mean to form the anomaly
``pv_anom_330`` (units PVU), flips latitude S→N, subsets time to
days 6–13 (hours 144..312 inclusive, UTC hourly), and writes
``outputs/<lc>/pv330_anom.nc``.

Output variables:
    pv330           raw PV on 330 K  [PVU]
    pv_anom_330     pv330 − ⟨pv330⟩_lon  [PVU]

NaNs are filled so TempestExtremes can operate on every grid point.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb")

START_HOUR = 6 * 24     # day 6 -> hour 144
END_HOUR = 13 * 24      # day 13 -> hour 312 (inclusive)
THETA_LEVEL = 330.0


def prep(lc: str) -> Path:
    ds = xr.open_dataset(ROOT / "outputs" / lc / "processed.nc")
    pv = ds["pv_on_theta"].sel(theta=THETA_LEVEL).drop_vars("theta",
                                                            errors="ignore")

    if float(pv["lat"][0]) > float(pv["lat"][-1]):
        pv = pv.isel(lat=slice(None, None, -1))

    zm = pv.mean("lon", skipna=True)
    anom = pv - zm
    pv_filled = pv.fillna(zm)
    anom = anom.fillna(0.0)

    t0 = pv["time"].values[0]
    hours = (pv["time"].values - t0) / np.timedelta64(1, "h")
    mask = (hours >= START_HOUR) & (hours <= END_HOUR)
    idx = np.where(mask)[0]
    pv_filled = pv_filled.isel(time=idx)
    anom = anom.isel(time=idx)
    print(f"[{lc}] pv330 time subset: {pv_filled.sizes['time']} frames, "
          f"{str(pv_filled['time'].values[0])} .. "
          f"{str(pv_filled['time'].values[-1])}")

    out = xr.Dataset({
        "pv330": pv_filled.astype(np.float32).rename("pv330"),
        "pv_anom_330": anom.astype(np.float32).rename("pv_anom_330"),
    })
    out["pv330"].attrs.update(
        long_name=f"Ertel PV on theta={THETA_LEVEL:.0f} K", units="PVU")
    out["pv_anom_330"].attrs.update(
        long_name=f"PV anomaly (zonal-mean removed) on theta="
                  f"{THETA_LEVEL:.0f} K",
        units="PVU")

    dst = ROOT / "outputs" / lc / "pv330_anom.nc"
    enc = {
        "time": {
            "units": "hours since 2000-01-01 00:00:00",
            "calendar": "standard",
            "dtype": "float64",
        },
        "pv330": {"zlib": True, "complevel": 4},
        "pv_anom_330": {"zlib": True, "complevel": 4},
    }
    out.to_netcdf(dst, encoding=enc)
    print(f"  -> {dst}")
    return dst


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        prep(lc)
