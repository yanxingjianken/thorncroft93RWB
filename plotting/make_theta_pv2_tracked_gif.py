#!/usr/bin/env python
"""Animate θ-on-2 PVU (GIF) with TempestExtremes blob outlines and
StitchNodes tracks overlaid (cyclonic RWB in blue, anticyclonic in red).

Outputs:  outputs/<lc>/figures/theta_on_pv2_tracked.gif
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb")


# ------------------------------------------------------------------ tracks
def parse_stitchnodes(path: Path) -> list[list[dict]]:
    """Return a list of tracks; each track = list of {time, lon, lat, val}."""
    tracks: list[list[dict]] = []
    cur: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            toks = line.split()
            if toks[0] == "start":
                if cur:
                    tracks.append(cur)
                cur = []
                continue
            # i j lon lat val year month day hour
            try:
                lon = float(toks[2]); lat = float(toks[3]); val = float(toks[4])
                yr, mo, dy, hr = map(int, toks[5:9])
            except (ValueError, IndexError):
                continue
            cur.append(
                {"time": datetime(yr, mo, dy, hr),
                 "lon": lon, "lat": lat, "val": val})
        if cur:
            tracks.append(cur)
    return tracks


def tracks_by_time(tracks: list[list[dict]]
                   ) -> dict[datetime, list[tuple[int, dict]]]:
    """index -> list of (track_id, point) for every timestep any track touches."""
    idx: dict[datetime, list[tuple[int, dict]]] = defaultdict(list)
    for tid, tr in enumerate(tracks):
        for pt in tr:
            idx[pt["time"]].append((tid, pt))
    return idx


def keep_top_n(tracks: list[list[dict]], n: int = 6) -> list[list[dict]]:
    """Return the `n` longest tracks (by number of points, then duration).

    The Thorncroft LC setup is initialised with a single zonal wave-6
    perturbation so we expect 6 ridges (θ' max) and 6 troughs (θ' min).
    """
    def key(tr):
        if not tr:
            return (0, 0.0)
        dur = (tr[-1]["time"] - tr[0]["time"]).total_seconds()
        return (len(tr), dur)
    return sorted(tracks, key=key, reverse=True)[:n]


# ------------------------------------------------------------------ animation
def animate(lc: str, stride: int = 1, trail_hours: int = 24):
    out_dir = ROOT / "outputs" / lc
    proc = xr.open_dataset(out_dir / "processed.nc")
    theta = proc["theta_on_pv2"]                         # (time, lat, lon)
    if float(theta["lat"][0]) > float(theta["lat"][-1]):
        theta = theta.isel(lat=slice(None, None, -1))

    # Blob files – find the variable automatically.
    def load_blob(nc_path: Path) -> xr.DataArray | None:
        if not nc_path.exists():
            return None
        bds = xr.open_dataset(nc_path)
        # pick the first 3-D int-ish var
        for v in bds.data_vars:
            arr = bds[v]
            if arr.ndim == 3:
                if float(arr["lat"][0]) > float(arr["lat"][-1]):
                    arr = arr.isel(lat=slice(None, None, -1))
                return arr.astype("float32")
        return None

    blob_neg = load_blob(out_dir / "tracks" / "blobs_neg.nc")
    blob_pos = load_blob(out_dir / "tracks" / "blobs_pos.nc")

    tr_min = keep_top_n(parse_stitchnodes(out_dir / "tracks" / "tracks_min.txt"), 6)
    tr_max = keep_top_n(parse_stitchnodes(out_dir / "tracks" / "tracks_max.txt"), 6)
    print(f"[{lc}] keeping {len(tr_min)} cyclonic + {len(tr_max)} anticyclonic tracks "
          f"(wave-6 expected)")
    idx_min = tracks_by_time(tr_min)
    idx_max = tracks_by_time(tr_max)

    lat = theta["lat"].values
    lon = theta["lon"].values
    times = theta["time"].values               # np.datetime64[ns]
    frames = list(range(0, len(times), stride))

    # NH polar stereographic
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8, 8), dpi=110)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-180, 180, 20, 90], crs=data_crs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.5)
    # polar circular clip
    import matplotlib.path as mpath
    theta_c = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta_c), np.cos(theta_c)]).T
    ax.set_boundary(mpath.Path(verts * radius + center), transform=ax.transAxes)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)

    levels = np.arange(270, 365, 5)
    cmap = plt.get_cmap("RdYlBu_r")
    cf = [ax.contourf(lon, lat, theta.isel(time=0).values,
                      levels=levels, cmap=cmap, extend="both",
                      transform=data_crs)]
    cb = fig.colorbar(cf[0], ax=ax, shrink=0.75, pad=0.05, orientation="vertical")
    cb.set_label("θ on ±2 PVU  [K]")

    title = ax.set_title("", fontsize=11)
    artists: list = []   # tempest overlays to clear each frame

    def to_datetime(np_dt64) -> datetime:
        ts = (np_dt64 - np.datetime64("1970-01-01T00:00:00")) \
            / np.timedelta64(1, "s")
        return datetime.utcfromtimestamp(float(ts))

    def draw(frame_idx: int):
        # remove previous contourf + overlays
        for c in cf[0].collections if hasattr(cf[0], "collections") else [cf[0]]:
            try: c.remove()
            except Exception: pass
        for a in artists:
            try: a.remove()
            except Exception: pass
        artists.clear()

        t = times[frame_idx]
        t_dt = to_datetime(t)
        day = (t_dt - datetime(2000, 1, 1)).total_seconds() / 86400.0

        cf[0] = ax.contourf(lon, lat, theta.isel(time=frame_idx).values,
                            levels=levels, cmap=cmap, extend="both",
                            transform=data_crs)

        # Blob outlines (negative = cyclonic in cyan, positive = anticyclonic magenta)
        if blob_neg is not None:
            b = blob_neg.isel(time=frame_idx).values
            if np.any(b > 0):
                cs = ax.contour(lon, lat, (b > 0).astype(float),
                                levels=[0.5], colors="cyan",
                                linewidths=1.2, transform=data_crs)
                artists.extend(cs.collections if hasattr(cs, "collections") else [cs])
        if blob_pos is not None:
            b = blob_pos.isel(time=frame_idx).values
            if np.any(b > 0):
                cs = ax.contour(lon, lat, (b > 0).astype(float),
                                levels=[0.5], colors="magenta",
                                linewidths=1.2, transform=data_crs)
                artists.extend(cs.collections if hasattr(cs, "collections") else [cs])

        # Tracks: draw trail (last trail_hours) + head marker for every
        # track active at/before this time.
        def draw_tracks(all_tracks, colour):
            for tr in all_tracks:
                pts = [p for p in tr if p["time"] <= t_dt
                       and (t_dt - p["time"]).total_seconds() / 3600.0
                       <= trail_hours + 0.1]
                if not pts:
                    continue
                lons = [p["lon"] for p in pts]
                lats = [p["lat"] for p in pts]
                ln, = ax.plot(lons, lats, "-", color=colour, lw=1.0,
                              alpha=0.9, transform=data_crs)
                artists.append(ln)
                # head marker only if current time coincides with latest pt
                if pts[-1]["time"] == t_dt:
                    sc = ax.scatter([pts[-1]["lon"]], [pts[-1]["lat"]],
                                    c=colour, s=28, edgecolor="k", lw=0.4,
                                    zorder=5, transform=data_crs)
                    artists.append(sc)

        draw_tracks(tr_min, "blue")
        draw_tracks(tr_max, "red")

        title.set_text(
            f"{lc.upper()}  day {day:5.2f}   θ on ±2 PVU + TempestExtremes "
            "RWB tracks (blue=cyclonic θ' min, red=anticyclonic θ' max)")
        return ()

    anim = FuncAnimation(fig, draw, frames=frames, interval=120, blit=False)
    gif_path = out_dir / "figures" / "theta_on_pv2_tracked.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[anim] writing {gif_path}  ({len(frames)} frames)")
    anim.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"[anim] done -> {gif_path}")


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        animate(lc)
