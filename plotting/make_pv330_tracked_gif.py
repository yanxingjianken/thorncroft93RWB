"""Animate 330 K PV anomaly with TempestExtremes tracks + blob outlines.

Outputs: outputs/<lc>/plots/pv330_tracked.gif
  - Up to 6 top-ranked PV-anom tracks (blue lines) on NH polar stereo
  - Blob outlines (cyan) where pv_anom_330 > adaptive threshold
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_utils import parse_stitchnodes, keep_top_n  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb")


def load_blob(nc_path):
    if not nc_path.exists():
        return None
    bds = xr.open_dataset(nc_path)
    for v in bds.data_vars:
        arr = bds[v]
        if arr.ndim == 3:
            if float(arr["lat"][0]) > float(arr["lat"][-1]):
                arr = arr.isel(lat=slice(None, None, -1))
            return arr.astype("float32")
    return None


def to_dt(np_dt64):
    ts = (np_dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(float(ts))


def animate(lc, stride=1, trail_hours=24):
    out_dir = ROOT / "outputs" / lc
    pv = xr.open_dataset(out_dir / "pv330_anom.nc")["pv_anom_330"]
    if float(pv["lat"][0]) > float(pv["lat"][-1]):
        pv = pv.isel(lat=slice(None, None, -1))

    blob_pos = load_blob(out_dir / "tracks" / "blobs_pv330_pos.nc")

    top6_path = out_dir / "tracks" / "tracks_max_top6_pv330.txt"
    if top6_path.exists():
        tr_max = parse_stitchnodes(top6_path)
    else:
        tr_max = keep_top_n(parse_stitchnodes(
            out_dir / "tracks" / "tracks_max_pv330.txt"), 6)
    print(f"[{lc}] {len(tr_max)} PV-anom tracks (top-6)")

    lat = pv["lat"].values
    lon = pv["lon"].values
    times = pv["time"].values
    frames = list(range(0, len(times), stride))

    proj = ccrs.NorthPolarStereo(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8, 8), dpi=110)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-180, 180, 20, 90], crs=data_crs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.5)
    theta_c = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta_c), np.cos(theta_c)]).T
    ax.set_boundary(mpath.Path(verts * radius + center), transform=ax.transAxes)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)

    vmax = 0.8
    levels = np.linspace(-vmax, vmax, 21)
    cmap = plt.get_cmap("RdBu_r")
    cf = [ax.contourf(lon, lat, pv.isel(time=0).values,
                      levels=levels, cmap=cmap, extend="both",
                      transform=data_crs)]
    cb = fig.colorbar(cf[0], ax=ax, shrink=0.75, pad=0.05, orientation="vertical")
    cb.set_label(r"$q'_{330K}$  [PVU]")

    title = ax.set_title("", fontsize=11)
    artists = []

    def draw(frame_idx):
        for c in cf[0].collections if hasattr(cf[0], "collections") else [cf[0]]:
            try: c.remove()
            except Exception: pass
        for a in artists:
            try: a.remove()
            except Exception: pass
        artists.clear()

        t = times[frame_idx]; t_dt = to_dt(t)
        day = (t_dt - datetime(2000, 1, 1)).total_seconds() / 86400.0
        cf[0] = ax.contourf(lon, lat, pv.isel(time=frame_idx).values,
                            levels=levels, cmap=cmap, extend="both",
                            transform=data_crs)

        # Need to find frame index in blob dataset (matching time)
        for blob, colour in ((blob_pos, "cyan"),):
            if blob is None:
                continue
            try:
                bi = int(np.argmin(np.abs(blob["time"].values - t)))
            except Exception:
                continue
            b = blob.isel(time=bi).values
            if np.any(b > 0):
                cs = ax.contour(lon, lat, (b > 0).astype(float),
                                levels=[0.5], colors=colour,
                                linewidths=1.2, transform=data_crs)
                artists.extend(cs.collections if hasattr(cs, "collections") else [cs])

        def draw_tracks(all_tracks, colour):
            for tr in all_tracks:
                pts = [p for p in tr if p["time"] <= t_dt
                       and (t_dt - p["time"]).total_seconds() / 3600.0
                       <= trail_hours + 0.1]
                if not pts:
                    continue
                # Unwrap longitudes so a track crossing the 0/360
                # dateline does not draw a line around the globe.
                lons = np.array([p["lon"] for p in pts], dtype=float)
                lats = np.array([p["lat"] for p in pts], dtype=float)
                # shift all longitudes to [-180, 180) relative to the
                # first point, then np.unwrap in degrees
                lons_u = np.unwrap(np.deg2rad(lons))
                lons_u = np.rad2deg(lons_u)
                ln, = ax.plot(lons_u, lats, "-", color=colour, lw=1.0,
                              alpha=0.9, transform=data_crs)
                artists.append(ln)
                if pts[-1]["time"] == t_dt:
                    sc = ax.scatter([pts[-1]["lon"]], [pts[-1]["lat"]],
                                    c=colour, s=28, edgecolor="k", lw=0.4,
                                    zorder=5, transform=data_crs)
                    artists.append(sc)

        draw_tracks(tr_max, "blue")    # PV-anom maxima tracks

        title.set_text(f"{lc.upper()}  day {day:5.2f}   "
                       r"$q'_{330K}$ + tracked PV-anom maxima "
                       "(blue line, cyan blob outline)")
        return ()

    anim = FuncAnimation(fig, draw, frames=frames, interval=120, blit=False)
    gif_path = out_dir / "plots" / "pv330_tracked.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[anim] writing {gif_path}  ({len(frames)} frames)")
    anim.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"[anim] done -> {gif_path}")


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        animate(lc)
