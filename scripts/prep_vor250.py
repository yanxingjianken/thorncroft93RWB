#!/usr/bin/env python
"""Prepare a TempestExtremes-friendly NetCDF of 250 hPa relative
vorticity for the Thorncroft LC runs.

Reads ``outputs/<lc>/processed.nc``, log-pressure interpolates
``vor_p`` between 200 and 300 hPa to 250 hPa, flips lat S→N, subsets
time to days 6–12 (hours 144–288), and writes
``outputs/<lc>/vor250.nc`` with a single variable ``vor250`` and a
CF-compliant time axis that TempestExtremes can parse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb")

# Day 6 -> hour 144, Day 12 -> hour 288 (inclusive).
START_HOUR = 6 * 24
END_HOUR = 12 * 24


def _log_interp_250(vor_p: xr.DataArray) -> xr.DataArray:
    """Linear-in-log-p interpolation between 200 and 300 hPa to 250 hPa."""
    p200 = 20000.0
    p300 = 30000.0
    p250 = 25000.0
    v200 = vor_p.sel(plev=p200)
    v300 = vor_p.sel(plev=p300)
    w = (np.log(p250) - np.log(p200)) / (np.log(p300) - np.log(p200))
    v250 = (1.0 - w) * v200 + w * v300
    v250 = v250.drop_vars("plev", errors="ignore")
    v250.attrs.update(long_name="relative vorticity at 250 hPa",
                      units="s-1")
    return v250.rename("vor250")


def prep(lc: str) -> Path:
    ds = xr.open_dataset(ROOT / "outputs" / lc / "processed.nc")
    vor_p = ds["vor_p"]

    # Interpolate to 250 hPa (log-p).
    v250 = _log_interp_250(vor_p)

    # Flip lat to ascending (S -> N) for TempestExtremes.
    if float(v250["lat"][0]) > float(v250["lat"][-1]):
        v250 = v250.isel(lat=slice(None, None, -1))

    # Subset time to days 6 .. 12 (inclusive).
    t0 = v250["time"].values[0]
    times = v250["time"].values
    hours = (times - t0) / np.timedelta64(1, "h")
    mask = (hours >= START_HOUR) & (hours <= END_HOUR)
    v250 = v250.isel(time=np.where(mask)[0])
    print(f"[{lc}] vor250 time subset: {v250.sizes['time']} frames, "
          f"{str(v250['time'].values[0])} .. {str(v250['time'].values[-1])}")

    # Fill any NaN to be safe for TE.
    v250 = v250.fillna(0.0).astype(np.float32)

    out = v250.to_dataset(name="vor250")
    dst = ROOT / "outputs" / lc / "vor250.nc"
    enc = {
        "time": {
            "units": "hours since 2000-01-01 00:00:00",
            "calendar": "standard",
            "dtype": "float64",
        },
        "vor250": {"zlib": True, "complevel": 4},
    }
    out.to_netcdf(dst, encoding=enc)
    print(f"  -> {dst}")
    return dst


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        prep(lc)
