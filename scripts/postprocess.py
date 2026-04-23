#!/usr/bin/env python
"""Postprocess a Thorncroft LC run.

SpeedyWeather writes one `run_<id>_NNNN/output.nc` per experiment under
`outputs/<lc>/raw/`. This script symlinks it to `outputs/<lc>/raw.nc`
and produces `outputs/<lc>/processed.nc` with derived fields used by
`plotting/thorncroft_figs.py`:

    - theta (potential temperature) on sigma levels
    - pv on sigma levels (full 3-D Ertel PV computed on σ, approximate)
    - pv_315K, pv_330K (linearly interpolated to specified θ surfaces)
    - u, v, theta at 850 hPa and 250 hPa (σ ≈ p/ps with ps=1000 hPa)
    - mslp (≈ ps, ≈ exp(pres) Pa)
    - zm_u = zonal-mean u(t, layer, lat)
    - eke(t) = global eddy kinetic energy
    - upvp_zm = zonal-mean u'v'(t, layer, lat)

Usage:
    micromamba run -n speedy_weather python scripts/postprocess.py lc1
    micromamba run -n speedy_weather python scripts/postprocess.py lc2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

OMEGA = 7.2921159e-5
R_DRY = 287.05
G = 9.80665
CP = 1004.64
KAPPA = R_DRY / CP
P0 = 1e5    # reference pressure [Pa]


def find_output(raw_dir: Path) -> Path:
    runs = sorted(raw_dir.glob("run_*/output.nc"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no run_*/output.nc in {raw_dir}")
    return runs[-1]


def surface_pressure_pa(ds: xr.Dataset) -> xr.DataArray:
    """Return surface pressure in Pa as (time, lat, lon).

    SpeedyWeather NetCDF output writes ``mslp`` in hPa (with flat
    topography ps ≡ mslp). Older configurations wrote ``pres`` as
    ln(ps [Pa]); we support both.
    """
    if "mslp" in ds.data_vars:
        units = ds["mslp"].attrs.get("units", "hPa").lower()
        ps = ds["mslp"] * (100.0 if units in ("hpa", "mb", "millibar") else 1.0)
    elif "pres" in ds.data_vars:
        ps = np.exp(ds["pres"])
    else:
        raise KeyError("neither 'mslp' nor 'pres' found in dataset")
    ps.attrs.update(long_name="surface pressure", units="Pa")
    return ps


def sigma_to_pressure(ds: xr.Dataset) -> xr.DataArray:
    """Pressure on full σ levels, shape (time, layer, lat, lon) [Pa]."""
    ps = surface_pressure_pa(ds)                  # Pa, (time, lat, lon)
    sigma = ds["layer"].astype(float)             # full-level σ
    p = sigma * ps
    p.attrs.update(long_name="pressure", units="Pa")
    return p


def potential_temperature(T: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    theta = T * (P0 / p) ** KAPPA
    theta.attrs.update(long_name="potential temperature", units="K")
    return theta


def interp_to_theta(var: xr.DataArray, theta: xr.DataArray,
                    theta_levels: np.ndarray) -> xr.DataArray:
    """Linear interpolation of `var` from σ-levels to θ=const surfaces.

    var, theta shapes: (time, layer, lat, lon). theta is (usually) not
    monotonic in σ, so we use a simple pressure-weighted search.
    """
    var = var.transpose("time", "layer", "lat", "lon")
    theta = theta.transpose("time", "layer", "lat", "lon")
    out = np.full(
        (var.sizes["time"], theta_levels.size, var.sizes["lat"], var.sizes["lon"]),
        np.nan, dtype=np.float32,
    )
    V = var.values
    TH = theta.values
    nlayer = V.shape[1]
    for i, th0 in enumerate(theta_levels):
        # find uppermost σ where θ crosses th0 (θ typically increases with
        # decreasing σ). Scan from top (σ small) to bottom.
        for k in range(1, nlayer):
            # θ(k-1) above (smaller σ), θ(k) below (larger σ)
            th_up = TH[:, k - 1, :, :]
            th_dn = TH[:, k, :, :]
            mask = ((th_up >= th0) & (th_dn < th0)) | ((th_up < th0) & (th_dn >= th0))
            w = (th0 - th_dn) / (th_up - th_dn + 1e-12)
            v = V[:, k, :, :] + w * (V[:, k - 1, :, :] - V[:, k, :, :])
            # only fill where not already filled
            sel = np.isnan(out[:, i, :, :]) & mask
            out[:, i, :, :] = np.where(sel, v, out[:, i, :, :])
    return xr.DataArray(
        out,
        dims=("time", "theta", "lat", "lon"),
        coords={"time": var.time, "theta": theta_levels,
                "lat": var.lat, "lon": var.lon},
        name=f"{var.name}_on_theta",
        attrs={"long_name": f"{var.name} on theta surfaces"},
    )


def interp_to_pressure(var: xr.DataArray, p: xr.DataArray,
                       p_levels: np.ndarray) -> xr.DataArray:
    """Linear-in-log(p) interpolation to pressure levels [Pa]."""
    var = var.transpose("time", "layer", "lat", "lon")
    p = p.transpose("time", "layer", "lat", "lon")
    out = np.full(
        (var.sizes["time"], p_levels.size, var.sizes["lat"], var.sizes["lon"]),
        np.nan, dtype=np.float32,
    )
    V = var.values
    P = p.values
    nlayer = V.shape[1]
    for i, p0 in enumerate(p_levels):
        lp0 = np.log(p0)
        for k in range(1, nlayer):
            lp_up = np.log(P[:, k - 1, :, :])
            lp_dn = np.log(P[:, k, :, :])
            # assume p increases with k (σ increases from top to surface)
            mask = (lp_dn >= lp0) & (lp_up < lp0)
            w = (lp0 - lp_dn) / (lp_up - lp_dn + 1e-12)
            v = V[:, k, :, :] + w * (V[:, k - 1, :, :] - V[:, k, :, :])
            sel = np.isnan(out[:, i, :, :]) & mask
            out[:, i, :, :] = np.where(sel, v, out[:, i, :, :])
    return xr.DataArray(
        out,
        dims=("time", "plev", "lat", "lon"),
        coords={"time": var.time, "plev": p_levels,
                "lat": var.lat, "lon": var.lon},
        name=f"{var.name}",
        attrs={"long_name": f"{var.name} on pressure levels"},
    )


def interp_theta_to_pv2(pv: xr.DataArray, theta: xr.DataArray,
                        pv_target_abs: float = 2.0) -> xr.DataArray:
    """θ on the ±pv_target_abs-PVU dynamical-tropopause surface.

    Scans each column from model top downward for the interface where
    PV crosses the target value (+target in the NH, −target in the SH)
    and linearly interpolates θ to that surface.  Columns that never
    reach the target (deep tropics) are left as NaN.
    """
    pv = pv.transpose("time", "layer", "lat", "lon")
    theta = theta.transpose("time", "layer", "lat", "lon")
    PV = pv.values
    TH = theta.values
    nt, nk, nlat, nlon = PV.shape
    lat_arr = pv["lat"].values
    pv_target = np.where(lat_arr >= 0, pv_target_abs,
                         -pv_target_abs).astype(np.float32)
    pv_target_b = np.broadcast_to(pv_target[None, :, None],
                                  (nt, nlat, nlon))
    out = np.full((nt, nlat, nlon), np.nan, dtype=np.float32)
    for k in range(1, nk):
        pv_up = PV[:, k - 1, :, :]   # above (smaller σ, stratosphere side)
        pv_dn = PV[:, k, :, :]       # below (larger σ, troposphere side)
        cross_nh = (pv_up >= pv_target_b) & (pv_dn < pv_target_b)
        cross_sh = (pv_up <= pv_target_b) & (pv_dn > pv_target_b)
        mask = cross_nh | cross_sh
        w = (pv_target_b - pv_dn) / (pv_up - pv_dn + 1e-12)
        th_val = TH[:, k, :, :] + w * (TH[:, k - 1, :, :] - TH[:, k, :, :])
        fill = np.isnan(out) & mask
        out = np.where(fill, th_val, out)
    da = xr.DataArray(
        out, dims=("time", "lat", "lon"),
        coords={"time": pv["time"], "lat": pv["lat"], "lon": pv["lon"]},
        name="theta_on_pv2",
        attrs={
            "long_name": "potential temperature on ±2 PVU "
                         "(dynamical tropopause)",
            "units": "K",
            "pv_target_abs_PVU": pv_target_abs,
        },
    )
    return da


def ertel_pv_sigma(ds: xr.Dataset, theta: xr.DataArray,
                   p: xr.DataArray) -> xr.DataArray:
    """Approximate Ertel PV on σ-levels:

        PV ≈ -g (ζ + f) ∂θ/∂p
    """
    lat = ds["lat"]
    f = 2.0 * OMEGA * np.sin(np.deg2rad(lat))
    zeta = ds["vor"].transpose("time", "layer", "lat", "lon")
    theta = theta.transpose("time", "layer", "lat", "lon")
    p = p.transpose("time", "layer", "lat", "lon")
    # ∂θ/∂p via centred differences along layer (using p on that layer)
    theta_v = theta.values
    p_v = p.values
    dtheta_dp = np.zeros_like(theta_v)
    # interior
    dtheta_dp[:, 1:-1, :, :] = (
        (theta_v[:, 2:, :, :] - theta_v[:, :-2, :, :])
        / (p_v[:, 2:, :, :] - p_v[:, :-2, :, :] + 1e-12)
    )
    dtheta_dp[:, 0, :, :] = (theta_v[:, 1, :, :] - theta_v[:, 0, :, :]) / \
                             (p_v[:, 1, :, :] - p_v[:, 0, :, :] + 1e-12)
    dtheta_dp[:, -1, :, :] = (theta_v[:, -1, :, :] - theta_v[:, -2, :, :]) / \
                              (p_v[:, -1, :, :] - p_v[:, -2, :, :] + 1e-12)
    pv = -G * (zeta + f) * xr.DataArray(
        dtheta_dp, dims=zeta.dims, coords=zeta.coords)
    # In PVU (1 PVU = 1e-6 K m^2 /kg /s)
    pv = pv * 1e6
    pv.attrs.update(long_name="Ertel PV (approx, σ-level)", units="PVU")
    pv.name = "pv"
    return pv


def main(lc: str):
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "outputs" / lc / "raw"
    src = find_output(raw_dir)
    print(f"[postprocess] reading {src}")
    ds = xr.open_dataset(src)

    # Standardise time axis (index from day 0)
    if "time" in ds.coords:
        t = ds["time"].values
        hours = (t - t[0]) / np.timedelta64(1, "h")
        ds = ds.assign_coords(time_hours=("time", hours))
        ds = ds.assign_coords(time_days=("time", hours / 24.0))

    # p, theta, pv on σ
    p = sigma_to_pressure(ds)
    # SpeedyWeather NetCDF output writes `temp` in °C; convert to K.
    temp_K = ds["temp"]
    if "units" in temp_K.attrs and temp_K.attrs["units"].lower() in ("°c", "c", "degc", "degree_celsius"):
        temp_K = temp_K + 273.15
    else:
        # Heuristic: SpeedyWeather default output is °C; if values <200 K
        # we assume °C and shift.
        if float(temp_K.min()) < 150.0:
            temp_K = temp_K + 273.15
    theta = potential_temperature(temp_K, p)
    pv = ertel_pv_sigma(ds, theta, p)

    # θ-surfaces (315 K for LC1 anticyclonic signature, 330 K for LC2
    # cyclonic roll-up, following Thorncroft 1993 Figs 5 & 6).
    theta_levels = np.array([315.0, 330.0, 350.0], dtype=np.float32)
    pv_on_theta = interp_to_theta(pv, theta, theta_levels)
    u_on_theta = interp_to_theta(ds["u"], theta, theta_levels)
    v_on_theta = interp_to_theta(ds["v"], theta, theta_levels)

    # Pressure-level fields (Pa) — expanded vertical grid for pvtend
    # (Helmholtz decomposition, QG-ω) and Rossby-wave-breaking
    # diagnostics. 10 standard levels from 100 to 925 hPa.
    p_levels = np.array([10000., 20000., 30000., 40000., 50000.,
                         60000., 70000., 80000., 85000., 92500.],
                        dtype=np.float32)
    u_p = interp_to_pressure(ds["u"], p, p_levels)
    v_p = interp_to_pressure(ds["v"], p, p_levels)
    theta_p = interp_to_pressure(theta, p, p_levels)
    vor_p = interp_to_pressure(ds["vor"], p, p_levels)
    T_p = interp_to_pressure(temp_K, p, p_levels)
    pv_p = interp_to_pressure(pv, p, p_levels)
    u_p.name, v_p.name = "u_p", "v_p"
    theta_p.name, vor_p.name = "theta_p", "vor_p"
    T_p.name, pv_p.name = "T_p", "pv_p"
    T_p.attrs.update(long_name="temperature on pressure levels", units="K")
    pv_p.attrs.update(long_name="Ertel PV on pressure levels", units="PVU")

    # θ on the ±2 PVU dynamical-tropopause surface.
    theta_on_pv2 = interp_theta_to_pv2(pv, theta, pv_target_abs=2.0)

    # MSLP (approximate — flat topography so ps ≈ MSLP)
    ps = surface_pressure_pa(ds)
    ps = ps.copy()
    ps.attrs.update(long_name="surface pressure (≈ MSLP, flat topo)",
                    units="Pa")
    ps.name = "mslp"

    # Zonal-mean u (on σ-levels)
    zm_u = ds["u"].mean("lon")
    zm_u.name = "zm_u"

    # Zonal-mean θ (on σ-levels) — for Thorncroft Fig 3 contours
    zm_theta = theta.mean("lon")
    zm_theta.name = "zm_theta"
    zm_theta.attrs.update(long_name="zonal-mean potential temperature",
                          units="K")

    # Eddy-momentum flux, zonal-mean (on σ-levels)
    up = ds["u"] - ds["u"].mean("lon")
    vp = ds["v"] - ds["v"].mean("lon")
    upvp = (up * vp).mean("lon")
    upvp.name = "upvp_zm"

    # Column-integrated EKE per unit horizontal area [J/m^2]:
    #     EKE = (ps/g) * Σ_k 0.5*(u'_k^2 + v'_k^2) * Δσ_k
    # Paper reports global-mean EKE of ~0 → 1.5e6 J/m^2 over 15 days.
    g_accel = 9.80665
    sig = np.asarray(ds["layer"].values, dtype=float)
    # σ_half with boundaries 0 (top) and 1 (surface), midpoints in between.
    sh = np.empty(sig.size + 1)
    sh[0] = 0.0
    sh[-1] = 1.0
    sh[1:-1] = 0.5 * (sig[:-1] + sig[1:])
    dsig = np.abs(np.diff(sh))
    dsigma = xr.DataArray(dsig, dims=("layer",),
                          coords={"layer": ds["layer"]})
    coslat = np.cos(np.deg2rad(ds["lat"]))
    w_area = coslat / coslat.mean()
    eke3d = 0.5 * (up**2 + vp**2)
    eke_col = (eke3d * dsigma).sum("layer") * ps / g_accel   # (time, lat, lon)
    eke = ((eke_col * w_area).mean(("lat", "lon")))
    eke.name = "eke"
    eke.attrs.update(long_name="column-integrated eddy kinetic energy (global mean)",
                     units="J/m^2")

    # Surface-layer temperature at σ ≈ 0.967 for paper Figs 5/8 (bottom
    # layer after top→surface ordering).
    T_surface = temp_K.isel(layer=-1)
    T_surface.name = "T_surface"
    T_surface.attrs.update(long_name="temperature at sigma~0.967",
                           units="K")

    out = xr.Dataset({
        "pv_sigma": pv,
        "pv_on_theta": pv_on_theta,
        "u_on_theta": u_on_theta,
        "v_on_theta": v_on_theta,
        "u_p": u_p,
        "v_p": v_p,
        "theta_p": theta_p,
        "vor_p": vor_p,
        "T_p": T_p,
        "pv_p": pv_p,
        "theta_on_pv2": theta_on_pv2,
        "mslp": ps,
        "T_surface": T_surface,
        "zm_u": zm_u,
        "zm_theta": zm_theta,
        "upvp_zm": upvp,
        "eke": eke,
        "theta_sigma": theta.rename("theta_sigma"),
    })
    # Carry helper time coords
    if "time_days" in ds.coords:
        out = out.assign_coords(time_days=ds["time_days"],
                                time_hours=ds["time_hours"])

    dst = project_root / "outputs" / lc / "processed.nc"
    print(f"[postprocess] writing {dst}")
    comp = {"zlib": True, "complevel": 4}
    enc = {v: comp for v in out.data_vars}
    out.to_netcdf(dst, encoding=enc)
    print("[postprocess] done")


if __name__ == "__main__":
    lc = sys.argv[1] if len(sys.argv) > 1 else "lc1"
    main(lc)
