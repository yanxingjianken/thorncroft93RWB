#!/usr/bin/env python
"""Hourly MP4 animations of (a) PV on 315 K / 330 K θ, (b) 250 hPa
relative vorticity, and (c) near-surface temperature for each LC
experiment.

Frames are sub-sampled to every 3 h by default (~128 frames for a
16-day run) so ffmpeg encoding stays fast while still smooth. Polar
gridlines are drawn without latitude text labels (Thorncroft figure
style).

Usage:
    micromamba run -n speedy_weather python plotting/make_figures.py lc1
    micromamba run -n speedy_weather python plotting/make_figures.py lc2
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import animation

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


def _setup_polar(ax):
    if not HAS_CARTOPY:
        return
    import matplotlib.path as mpath
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.3, color="gray")
    ax.gridlines(draw_labels=False, linewidth=0.3,
                 color="gray", alpha=0.6,
                 xlocs=np.arange(-180, 181, 30),
                 ylocs=np.arange(20, 91, 15))
    # Restore the circular clip boundary that ax.clear() removes
    # (cartopy does not re-apply it automatically after cla/clear).
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.column_stack([0.5 + 0.5 * np.sin(theta),
                              0.5 + 0.5 * np.cos(theta)])
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)


def animate_field(field, title_prefix, out_path,
                  levels, cmap, units,
                  polar=True, fps=10, frame_stride=3):
    """Render an animated GIF.

    Parameters
    ----------
    field : DataArray with dims (time, lat, lon)
    frame_stride : int
        Use every Nth frame (default 3 → every 3 h for hourly data).
    """
    lon = field["lon"].values
    lat = field["lat"].values
    lon_w = np.concatenate([lon, [lon[0] + 360.0]])
    t_days = field["time_days"].values if "time_days" in field.coords \
        else np.arange(field.sizes["time"]) / 24.0

    # Clip/blank blow-up frames.
    lvl_max = float(np.max(np.abs(levels)))
    data = field.values
    data = np.where(np.isfinite(data), data, np.nan)
    data = np.where(np.abs(data) > 10.0 * lvl_max, np.nan, data)
    field = field.copy(data=data)

    fig = plt.figure(figsize=(7.0, 7.0))
    if HAS_CARTOPY and polar:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        _setup_polar(ax)
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim(20, 90)
        ax.set_xlabel("longitude [°E]")
        ax.set_ylabel("latitude [°N]")
        transform = None

    # Seed colorbar from day 0.
    f0 = np.nan_to_num(field.isel(time=0).values, nan=0.0)
    f0_w = np.concatenate([f0, f0[:, :1]], axis=1)
    kw0 = dict(levels=levels, cmap=cmap, extend="both")
    if transform is not None:
        kw0["transform"] = transform
    cf_container = [ax.contourf(lon_w, lat, f0_w, **kw0)]
    cbar = fig.colorbar(cf_container[0], ax=ax, orientation="horizontal",
                        shrink=0.75, pad=0.08, label=units)
    cbar.ax.tick_params(labelsize=9)
    title_artist = ax.set_title(f"{title_prefix}  day {t_days[0]:.2f}")

    frame_idx = list(range(0, field.sizes["time"], max(1, int(frame_stride))))
    if frame_idx[-1] != field.sizes["time"] - 1:
        frame_idx.append(field.sizes["time"] - 1)

    def frame(i):
        # Remove only the contourf layer — coastlines/gridlines stay put.
        # (ax.clear() would erase the cartopy polar boundary; cf.remove()
        # is the correct approach in matplotlib >= 3.8.)
        cf_container[0].remove()
        f = np.nan_to_num(field.isel(time=i).values, nan=0.0)
        f_w = np.concatenate([f, f[:, :1]], axis=1)
        kw = dict(levels=levels, cmap=cmap, extend="both")
        if transform is not None:
            kw["transform"] = transform
        cf_container[0] = ax.contourf(lon_w, lat, f_w, **kw)
        title_artist.set_text(f"{title_prefix}  day {t_days[i]:.2f}")
        return [cf_container[0]]

    ani = animation.FuncAnimation(
        fig, frame, frames=frame_idx,
        interval=1000 / fps, blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400,
                                    codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p"])
    ani.save(out_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"  -> {out_path}")


def main(lc):
    project_root = Path(__file__).resolve().parents[1]
    proc = project_root / "outputs" / lc / "processed.nc"
    if not proc.exists():
        raise FileNotFoundError(proc)
    ds = xr.open_dataset(proc)
    out_dir = project_root / "outputs" / lc / "mp4"

    # Both LC1 and LC2 plotted on 330 K for comparability.
    theta_K = 330.0
    pv_theta = ds["pv_on_theta"].sel(theta=theta_K, method="nearest")
    animate_field(pv_theta, f"{lc.upper()} PV on {theta_K:.0f} K",
                  out_dir / f"pv_{int(theta_K)}K.mp4",
                  levels=np.linspace(-8, 8, 21), cmap="RdBu_r",
                  units="PV [PVU]", polar=True)

    # θ on ±2 PVU — dynamical-tropopause view (RWB diagnostic).
    if "theta_on_pv2" in ds.data_vars:
        animate_field(ds["theta_on_pv2"],
                      f"{lc.upper()} θ on ±2 PVU",
                      out_dir / "theta_on_pv2.mp4",
                      levels=np.arange(270, 365, 5), cmap="RdYlBu_r",
                      units="θ [K]", polar=True)

    # 250 hPa relative vorticity, auto-scale from pre-blowup 99th pct.
    zeta_250 = ds["vor_p"].sel(plev=25000.0, method="nearest")
    arr = zeta_250.values
    arr = arr[np.isfinite(arr) & (np.abs(arr) < 1e-2)]
    vmax = float(np.percentile(np.abs(arr), 99)) if arr.size else 3e-4
    vmax = vmax or 3e-4
    animate_field(zeta_250, f"{lc.upper()} ζ at 250 hPa",
                  out_dir / "zeta_250hPa.mp4",
                  levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r",
                  units="ζ [s⁻¹]", polar=True)

    # Surface T (σ≈0.967) animated — useful for following the cyclone.
    if "T_surface" in ds.data_vars:
        T_c = ds["T_surface"] - 273.15
        animate_field(T_c, f"{lc.upper()} T at σ=0.967",
                      out_dir / "T_surface.mp4",
                      levels=np.arange(-40, 41, 4), cmap="RdBu_r",
                      units="T [°C]", polar=True)


if __name__ == "__main__":
    lc = sys.argv[1] if len(sys.argv) > 1 else "lc1"
    main(lc)
