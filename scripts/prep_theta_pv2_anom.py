#!/usr/bin/env python
"""Prepare a TempestExtremes-friendly NetCDF with θ on the ±2-PVU
dynamical tropopause and its zonal-mean-removed anomaly.

Output:
    outputs/<lc>/theta_pv2_anom.nc
        theta_pv2(time, lat, lon)       [K]   — raw θ on ±2 PVU
        theta_pv2_anom(time, lat, lon)  [K]   — θ' = θ − <θ>_lon

Lat is flipped to ascending (south → north) for TE compatibility, and
NaNs are filled with the instantaneous zonal-mean so TE can operate on
every grid point without choking on missing values.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb")


def prep(lc: str) -> Path:
    ds = xr.open_dataset(ROOT / "outputs" / lc / "processed.nc")
    th = ds["theta_on_pv2"]                           # (time, lat, lon)

    # Flip lat to ascending if needed (TempestExtremes prefers S→N).
    if float(th["lat"][0]) > float(th["lat"][-1]):
        th = th.isel(lat=slice(None, None, -1))

    # Clip spurious stratospheric hits (θ > 420 K on 2 PVU is likely a
    # mis-interpolation into the deep stratosphere / tropics).
    th = th.where(th < 420.0)

    # Zonal mean (skip NaN).
    zm = th.mean("lon", skipna=True)                 # (time, lat)
    anom = th - zm

    # Fill NaN anomalies with 0 so TE has valid numbers everywhere.
    anom = anom.fillna(0.0)
    # Fill raw θ NaN with zonal-mean.
    th_filled = th.fillna(zm)

    out = xr.Dataset({
        "theta_pv2": th_filled.astype(np.float32).rename("theta_pv2"),
        "theta_pv2_anom": anom.astype(np.float32).rename("theta_pv2_anom"),
    })
    out["theta_pv2"].attrs.update(
        long_name="theta on 2-PVU dynamical tropopause", units="K")
    out["theta_pv2_anom"].attrs.update(
        long_name="theta' = theta - zonal-mean(theta) on 2 PVU", units="K")

    # CF time encoding that TE accepts.
    dst = ROOT / "outputs" / lc / "theta_pv2_anom.nc"
    enc = {
        "time": {
            "units": "hours since 2000-01-01 00:00:00",
            "calendar": "standard",
            "dtype": "float64",
        },
        "theta_pv2": {"zlib": True, "complevel": 4},
        "theta_pv2_anom": {"zlib": True, "complevel": 4},
    }
    out.to_netcdf(dst, encoding=enc)
    print(f"  -> {dst}")
    return dst


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        prep(lc)
