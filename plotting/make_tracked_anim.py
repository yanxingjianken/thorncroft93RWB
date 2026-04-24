"""Animate each method's anomaly field with its C/AC tracks overlaid.

For every (LC, method) renders one mp4 in outputs/<lc>/plots/<method>_tracked.mp4.
All 3 methods are rendered by default so the user can visually compare them;
the winner from compare_methods.py is *not* singled out.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_utils import parse_stitchnodes  # noqa
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb")


def to_dt(np_dt64):
    ts = (np_dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(float(ts))


def _cbar_label(method: str) -> str:
    return {
        "pv330":     r"$q'_{330\,K}$  [PVU]",
        "zeta250":   r"$\zeta'_{250\,hPa}$  [s$^{-1}$]",
        "theta_pv2": r"$\theta'_{2\,PVU}$  [K]",
    }.get(method, "anomaly")


def animate(lc: str, method: str, stride: int = 1,
            trail_hours: int = 24):
    out_dir = ROOT / "outputs" / lc
    spec = CFG.METHOD[method]

    ds = xr.open_dataset(out_dir / spec["input_nc"])
    pv = ds[spec["var"]]
    if float(pv["lat"][0]) > float(pv["lat"][-1]):
        pv = pv.isel(lat=slice(None, None, -1))

    def _load_top6(pol):
        p = out_dir / "tracks" / method / f"tracks_{pol}_top6.txt"
        return parse_stitchnodes(p) if p.exists() else []

    tr_C = _load_top6("C")
    tr_AC = _load_top6("AC")
    print(f"[{lc}:{method}] C={len(tr_C)} AC={len(tr_AC)} top-6 tracks")

    lat = pv["lat"].values
    lon = pv["lon"].values
    # Extend lon by one column for periodic coverage so Cartopy contourf
    # doesn't leave a gap / produce blank frames at the 0°/360° seam.
    lon_wrap = np.append(lon, lon[0] + 360.0)
    times = pv["time"].values
    frames = list(range(0, len(times), stride))

    vmax = float(np.nanpercentile(np.abs(pv.values), 99.0))
    levels = np.linspace(-vmax, vmax, 21)
    cmap = plt.get_cmap("RdBu_r")

    proj = ccrs.NorthPolarStereo(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8, 8), dpi=110)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-180, 180, 20, 90], crs=data_crs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.5)
    theta_c = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta_c), np.cos(theta_c)]).T
    ax.set_boundary(mpath.Path(verts * radius + center),
                    transform=ax.transAxes)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)

    def _get_frame_data(frame_idx):
        d = pv.isel(time=frame_idx).values
        return np.concatenate([d, d[:, :1]], axis=1)

    cf = [ax.contourf(lon_wrap, lat, _get_frame_data(0), levels=levels,
                      cmap=cmap, extend="both", transform=data_crs)]
    cb = fig.colorbar(cf[0], ax=ax, shrink=0.75, pad=0.05,
                      orientation="vertical")
    cb.set_label(_cbar_label(method))
    title = ax.set_title("", fontsize=11)
    artists = []

    def draw(frame_idx):
        # Remove previous contourf using the modern API (matplotlib >= 3.8).
        # ContourSet.remove() atomically removes all child collections;
        # the old collections-iteration loop was fragile in mpl 3.10 because
        # .collections is deprecated/removed and the fallback path could fail
        # silently, leaving stale artists that produced blank frames.
        try:
            cf[0].remove()
        except Exception:
            pass
        for a in artists:
            try: a.remove()
            except Exception: pass
        artists.clear()

        t = times[frame_idx]; t_dt = to_dt(t)
        day = (t_dt - datetime(2000, 1, 1)).total_seconds() / 86400.0
        cf[0] = ax.contourf(lon_wrap, lat, _get_frame_data(frame_idx),
                            levels=levels, cmap=cmap, extend="both",
                            transform=data_crs)

        def draw_tracks(all_tracks, colour):
            for tr in all_tracks:
                pts = [p for p in tr if p["time"] <= t_dt
                       and (t_dt - p["time"]).total_seconds() / 3600.0
                       <= trail_hours + 0.1]
                if not pts:
                    continue
                lons = np.array([p["lon"] for p in pts], dtype=float)
                lats = np.array([p["lat"] for p in pts], dtype=float)
                lons_u = np.rad2deg(np.unwrap(np.deg2rad(lons)))
                ln, = ax.plot(lons_u, lats, "-", color=colour, lw=1.0,
                              alpha=0.9, transform=data_crs)
                artists.append(ln)
                if pts[-1]["time"] == t_dt:
                    sc = ax.scatter([pts[-1]["lon"]], [pts[-1]["lat"]],
                                    c=colour, s=28, edgecolor="k", lw=0.4,
                                    zorder=5, transform=data_crs)
                    artists.append(sc)

        draw_tracks(tr_C, "blue")
        draw_tracks(tr_AC, "orange")

        title.set_text(
            f"{lc.upper()} [{method}]  day {day:5.2f}   "
            "C (blue) & AC (orange) tracks")
        return ()

    anim = FuncAnimation(fig, draw, frames=frames, interval=120, blit=False)
    out_plots = out_dir / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)
    mp4_path = out_plots / f"{method}_tracked.mp4"
    print(f"[anim] writing {mp4_path}  ({len(frames)} frames)")
    anim.save(mp4_path, writer=FFMpegWriter(fps=CFG.ANIM_FPS, bitrate=2400))
    print(f"[anim] done -> {mp4_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--methods", nargs="*", default=list(CFG.METHODS),
                    help="Which methods to animate (default: all 3).")
    args = ap.parse_args()
    for lc in args.lcs:
        for m in args.methods:
            animate(lc, method=m)
