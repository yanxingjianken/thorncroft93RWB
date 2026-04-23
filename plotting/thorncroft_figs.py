#!/usr/bin/env python
"""Sanity-check figures replicating Thorncroft, Hoskins & McIntyre
(1993, QJRMS, doi 10.1002/qj.49711950903) LC1/LC2 life-cycle runs.

Paper-figure mapping:

    paper Fig 3   LC1+LC2  lat-σ zonal-mean [u] (shading) and [θ]
                  (contours) at day 0 and day 10.
    paper Fig 4   EKE(t) for LC1 and LC2, column-integrated J/m².
    paper Fig 5   LC1  T at σ=0.967 between days 4 and 9. NH polar
                  sector, 4 K contours, 0°C dotted, <0°C dashed.
    paper Fig 6   LC1  surface pressure between days 4 and 9. NH polar
                  sector, 4 mb contours, 1000 mb dotted, <1000 mb dashed.
    paper Fig 7   LC1  PV on 315 K θ, NH polar stereo, days 4–9.
    paper Fig 8   LC2  T at σ=0.967 days 4–9 (same style as Fig 5).
    paper Fig 9   LC2  surface pressure days 4–9 (same style as Fig 6).
    paper Fig 10  LC2  PV on 330 K θ, NH polar stereo, days 4–9.
    paper Fig 15  LC1+LC2  eddy momentum flux [u'v'] lat-σ cross
                  sections at days 0, 9, 11.

Usage:
    micromamba run -n speedy_weather python plotting/thorncroft_figs.py
    micromamba run -n speedy_weather python plotting/thorncroft_figs.py lc1
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


def _pick_time(da, day):
    if "time_days" in da.coords:
        idx = int(np.argmin(np.abs(da["time_days"].values - day)))
    else:
        idx = min(int(round(day * 24)), da.sizes["time"] - 1)
    return da.isel(time=idx)


def _polar_axes(fig, subplot_spec):
    """NH polar stereographic axes, longitude meridians every 30°,
    latitude circles every 20°, NO text labels (per Thorncroft figure
    style). Returns (ax, cartopy_transform_or_None)."""
    if HAS_CARTOPY:
        ax = fig.add_subplot(*subplot_spec,
                             projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.3, color="gray")
        ax.gridlines(draw_labels=False, linewidth=0.3,
                     color="gray", alpha=0.5,
                     xlocs=np.arange(-180, 181, 30),
                     ylocs=np.arange(20, 91, 20))
        return ax, ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(*subplot_spec)
        ax.set_ylim(20, 90)
        ax.set_xlabel("longitude [°E]")
        return ax, None


# -----------------------------------------------------------------
# Fig 7 (LC1) / Fig 10 (LC2) — PV on θ, days 4–9
# -----------------------------------------------------------------
def paper_fig7_10(ds, theta_K, out_path, label, paper_num,
                  day_list=(4, 5, 6, 7, 8, 9)):
    """Days 4–9 of θ on the ±2-PVU dynamical tropopause (RWB view).

    `theta_K` is retained in the signature for call-site compatibility
    but is no longer used; the panel now plots θ on ±2 PVU.
    """
    del theta_K  # kept for backwards compat
    if "theta_on_pv2" not in ds.data_vars:
        print(f"  [skip] theta_on_pv2 not in {label}")
        return
    theta2 = ds["theta_on_pv2"]
    lon = ds["lon"].values
    lat = ds["lat"].values
    lon_w = np.concatenate([lon, [lon[0] + 360.0]])

    fig = plt.figure(figsize=(14, 9))
    levels = np.arange(270, 365, 5)
    cf = None
    for i, d in enumerate(day_list):
        ax, transform = _polar_axes(fig, (2, 3, i + 1))
        snap = _pick_time(theta2, d)
        field = snap.values
        # fill NaN with zonal-mean so panels aren't blank at poles
        zm = np.nanmean(field, axis=1, keepdims=True)
        field = np.where(np.isfinite(field), field, zm)
        field = np.nan_to_num(field, nan=295.0)
        field = np.concatenate([field, field[:, :1]], axis=1)
        kw = dict(levels=levels, cmap="RdYlBu_r", extend="both")
        if transform is not None:
            kw["transform"] = transform
        cf = ax.contourf(lon_w, lat, field, **kw)
        ax.set_title(f"day {d}")
    fig.suptitle(f"{label} — θ [K] on ±2 PVU dynamical tropopause  "
                 f"(analogue of Thorncroft 1993 Fig {paper_num})")
    if cf is not None:
        fig.colorbar(cf, ax=fig.axes, orientation="horizontal",
                     shrink=0.6, pad=0.05, label="θ on 2 PVU [K]")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -----------------------------------------------------------------
# Fig 5 / 8 — T at σ=0.967, days 4–9. 4 K contours, 0°C dotted,
# <0°C dashed.
# -----------------------------------------------------------------
def paper_fig5_8(ds, out_path, label, paper_num,
                 day_list=(4, 5, 6, 7, 8, 9)):
    if "T_surface" not in ds.data_vars:
        print(f"  [skip] T_surface not in {label}")
        return
    T = ds["T_surface"] - 273.15                     # K → °C
    lon = ds["lon"].values
    lat = ds["lat"].values
    lon_w = np.concatenate([lon, [lon[0] + 360.0]])

    fig = plt.figure(figsize=(14, 9))
    levels = np.arange(-40, 41, 4)
    cf = None
    for i, d in enumerate(day_list):
        ax, transform = _polar_axes(fig, (2, 3, i + 1))
        snap = _pick_time(T, d)
        field = np.nan_to_num(snap.values, nan=0.0, posinf=0.0, neginf=0.0)
        field = np.concatenate([field, field[:, :1]], axis=1)
        kw = dict(levels=levels, cmap="RdBu_r", extend="both")
        if transform is not None:
            kw["transform"] = transform
        cf = ax.contourf(lon_w, lat, field, **kw)
        pos_lvls = [lv for lv in levels if lv > 0]
        neg_lvls = [lv for lv in levels if lv < 0]
        kw_line = dict(colors="k", linewidths=0.4)
        if transform is not None:
            kw_line["transform"] = transform
        if pos_lvls:
            ax.contour(lon_w, lat, field, levels=pos_lvls,
                       linestyles="solid", **kw_line)
        if neg_lvls:
            ax.contour(lon_w, lat, field, levels=neg_lvls,
                       linestyles="dashed", **kw_line)
        kw_zero = dict(colors="k", linewidths=0.8, linestyles=":")
        if transform is not None:
            kw_zero["transform"] = transform
        ax.contour(lon_w, lat, field, levels=[0.0], **kw_zero)
        ax.set_title(f"day {d}")
    fig.suptitle(
        f"{label} — T at σ=0.967 [°C]  "
        f"(analogue of Thorncroft 1993 Fig {paper_num}; "
        "4 K contours, 0°C dotted, <0°C dashed)")
    if cf is not None:
        fig.colorbar(cf, ax=fig.axes, orientation="horizontal",
                     shrink=0.6, pad=0.05, label="T [°C]")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -----------------------------------------------------------------
# Fig 6 / 9 — surface pressure, days 4–9. 4 mb contours,
# 1000 mb dotted, <1000 mb dashed.
# -----------------------------------------------------------------
def paper_fig6_9(ds, out_path, label, paper_num,
                 day_list=(4, 5, 6, 7, 8, 9)):
    if "mslp" not in ds.data_vars:
        print(f"  [skip] mslp not in {label}")
        return
    ps = ds["mslp"] / 100.0                          # Pa → hPa ≡ mb
    lon = ds["lon"].values
    lat = ds["lat"].values
    lon_w = np.concatenate([lon, [lon[0] + 360.0]])

    vals = ps.values
    lo = int(np.floor(np.nanpercentile(vals, 1) / 4.0) * 4)
    hi = int(np.ceil(np.nanpercentile(vals, 99) / 4.0) * 4)
    lo = min(lo, 980)
    hi = max(hi, 1040)
    levels = np.arange(lo, hi + 1, 4)

    fig = plt.figure(figsize=(14, 9))
    cf = None
    for i, d in enumerate(day_list):
        ax, transform = _polar_axes(fig, (2, 3, i + 1))
        snap = _pick_time(ps, d)
        field = np.nan_to_num(snap.values, nan=1013.25)
        field = np.concatenate([field, field[:, :1]], axis=1)
        kw = dict(levels=levels, cmap="RdBu_r", extend="both")
        if transform is not None:
            kw["transform"] = transform
        cf = ax.contourf(lon_w, lat, field, **kw)
        high_lvls = [lv for lv in levels if lv > 1000]
        low_lvls = [lv for lv in levels if lv < 1000]
        kw_line = dict(colors="k", linewidths=0.4)
        if transform is not None:
            kw_line["transform"] = transform
        if high_lvls:
            ax.contour(lon_w, lat, field, levels=high_lvls,
                       linestyles="solid", **kw_line)
        if low_lvls:
            ax.contour(lon_w, lat, field, levels=low_lvls,
                       linestyles="dashed", **kw_line)
        kw_zero = dict(colors="k", linewidths=0.8, linestyles=":")
        if transform is not None:
            kw_zero["transform"] = transform
        ax.contour(lon_w, lat, field, levels=[1000.0], **kw_zero)
        ax.set_title(f"day {d}")
    fig.suptitle(
        f"{label} — surface pressure [mb]  "
        f"(analogue of Thorncroft 1993 Fig {paper_num}; "
        "4 mb contours, 1000 mb dotted)")
    if cf is not None:
        fig.colorbar(cf, ax=fig.axes, orientation="horizontal",
                     shrink=0.6, pad=0.05, label="ps [mb]")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -----------------------------------------------------------------
# Fig 3 — lat-σ zonal-mean [u] (shading) and [θ] (contours)
# -----------------------------------------------------------------
def paper_fig3(ds_lc1, ds_lc2, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             sharex=True, sharey=True)
    u_levels = np.arange(-20, 70, 5)
    θ_levels = np.arange(250, 500, 10)
    cm = None
    for col, (ds, lab) in enumerate([(ds_lc1, "LC1"), (ds_lc2, "LC2")]):
        if ds is None:
            continue
        zm_u = ds["zm_u"]
        zm_θ = ds["zm_theta"] if "zm_theta" in ds else None
        for row, d in enumerate([0, 10]):
            ax = axes[row, col]
            u_snap = _pick_time(zm_u, d)
            sig = u_snap["layer"].values
            lat = u_snap["lat"].values
            cm = ax.contourf(lat, sig, u_snap.values,
                             levels=u_levels, cmap="RdBu_r",
                             extend="both")
            if zm_θ is not None:
                θs = _pick_time(zm_θ, d).values
                ax.contour(lat, sig, θs, levels=θ_levels,
                           colors="k", linewidths=0.5)
            ax.set_ylim(1.0, 0.0)
            ax.set_xlim(0, 90)
            ax.set_title(f"{lab}  day {d}")
            ax.tick_params(axis="both", labelsize=9)
            if row == 1:
                ax.set_xlabel("latitude [°N]")
            if col == 0:
                ax.set_ylabel("σ = p / p_s")
    fig.suptitle("Zonal-mean [u] (shading) and [θ] (contours)  "
                 "(analogue of Thorncroft 1993 Fig 3)")
    if cm is not None:
        fig.colorbar(cm, ax=axes.ravel().tolist(), label="u [m/s]",
                     shrink=0.85, orientation="vertical")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -----------------------------------------------------------------
# Fig 4 — EKE vs time, column-integrated [J m⁻²]
# -----------------------------------------------------------------
def paper_fig4(ds_lc1, ds_lc2, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, lab, c in [(ds_lc1, "LC1", "tab:blue"),
                       (ds_lc2, "LC2", "tab:red")]:
        if ds is None:
            continue
        t = ds["time_days"].values if "time_days" in ds.coords \
            else np.arange(ds.sizes["time"]) / 24.0
        eke = ds["eke"].values.astype(float)
        # Mask non-physical late-time spikes.
        eke = np.where(np.isfinite(eke) & (eke < 1e8), eke, np.nan)
        ax.plot(t, eke, lw=1.8, label=lab, color=c)
    ax.set_xlabel("day")
    ax.set_ylabel("EKE  [J m⁻²]")
    ax.set_title("Column-integrated eddy kinetic energy  "
                 "(analogue of Thorncroft 1993 Fig 4)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -----------------------------------------------------------------
# Fig 15 — eddy momentum flux [u'v'] lat-σ cross sections
# -----------------------------------------------------------------
def paper_fig15(ds, out_path, label, days_sel=(0, 9, 11)):
    upvp = ds["upvp_zm"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    vals = upvp.values
    vals = vals[np.isfinite(vals) & (np.abs(vals) < 1e5)]
    vmax = float(np.nanpercentile(np.abs(vals), 98)) if vals.size else 1.0
    vmax = max(vmax, 1.0)
    levels = np.linspace(-vmax, vmax, 21)
    cm = None
    for ax, d in zip(axes, days_sel):
        snap = _pick_time(upvp, d)
        sig = snap["layer"].values
        lat = snap["lat"].values
        field = np.nan_to_num(snap.values, nan=0.0, posinf=0.0, neginf=0.0)
        cm = ax.contourf(lat, sig, field,
                         levels=levels, cmap="RdBu_r", extend="both")
        ax.set_ylim(1.0, 0.0)
        ax.set_xlim(0, 90)
        ax.set_xlabel("latitude [°N]")
        ax.set_title(f"day {d}")
        ax.tick_params(axis="both", labelsize=9)
    axes[0].set_ylabel("σ = p / p_s")
    fig.suptitle(f"{label} — eddy momentum flux [u'v'] (EP-flux analogue) "
                 "(analogue of Thorncroft 1993 Fig 15)")
    if cm is not None:
        fig.colorbar(cm, ax=axes.ravel().tolist(),
                     label="[u'v']  [m² s⁻²]", shrink=0.85)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main(lc_list):
    project_root = Path(__file__).resolve().parents[1]
    datasets = {}
    for lc in ("lc1", "lc2"):
        p = project_root / "outputs" / lc / "processed.nc"
        if p.exists():
            datasets[lc] = xr.open_dataset(p)
        else:
            print(f"[thorncroft_figs] {p} missing — skipping {lc}")

    for lc, label, theta_K, pv_num, T_num, ps_num in [
        ("lc1", "LC1", 315.0, "7", "5", "6"),
        ("lc2", "LC2", 330.0, "10", "8", "9"),
    ]:
        if lc not in datasets or lc not in lc_list:
            continue
        ds = datasets[lc]
        out_dir = project_root / "outputs" / lc / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        paper_fig5_8(ds, out_dir / f"paper_fig{T_num}_T_sigma0967.png",
                     label, T_num)
        paper_fig6_9(ds, out_dir / f"paper_fig{ps_num}_ps.png",
                     label, ps_num)
        paper_fig7_10(ds, theta_K,
                      out_dir / f"paper_fig{pv_num}_pv_on_theta.png",
                      label, pv_num)
        paper_fig15(ds,
                    out_dir / "paper_fig15_upvp_cross.png", label)

    if "lc1" in datasets or "lc2" in datasets:
        out_dir = project_root / "outputs" / "lc1" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        paper_fig3(datasets.get("lc1"), datasets.get("lc2"),
                   out_dir / "paper_fig3_zm_u_theta.png")
        paper_fig4(datasets.get("lc1"), datasets.get("lc2"),
                   out_dir / "paper_fig4_eke.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a in ("lc1", "lc2")]
    if not args:
        args = ["lc1", "lc2"]
    main(args)
