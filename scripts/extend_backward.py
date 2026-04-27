"""Backward-extend the canonical zeta250 tracks.

For each LC the standard pipeline tracks from day 6 (LC1) or day 8 (LC2)
through the full break. This script:

  1. Reads the existing top-6 tracks from
     ``outputs/<lc>/tracks/zeta250/tracks_{C,AC}_top6.txt``.
  2. **Linearly interpolates** any missing hourly entries inside each
     track's time range (some StitchNodes outputs skip hours).  The
     existing tracked feature itself is *not* changed.
  3. **Walks backward** in time from each track's first point, hour by
     hour. At every backward step we search the zeta250 anomaly field
     within ``±R_SEARCH_DEG`` of the last known centre for the local
     extremum (max for C, min for AC). The new point is accepted only
     if its magnitude exceeds the standard mask threshold and its great-
     circle distance to the previous point is < ``JUMP_MAX_DEG_PER_H``.
     The walk stops as soon as either gate fails (this naturally cuts
     off when the feature emerges out of the broad meridionally-aligned
     background, before the LC1/LC2 tilt signatures diverge sharply).
  4. Writes the extended top-6 file and CSV under ``zeta250_back``.
  5. Renders a polar-cap zeta250 animation (same style as
     ``zeta250_tracked.mp4``) showing the extended tracks.
  6. Builds Lagrangian composites and runs the tilt / γ·Φ_def
     analysis on the backward-extended window.

Outputs (per LC, each polarity):

  outputs/<lc>/tracks/zeta250_back/tracks_{pol}_top6.txt
  outputs/<lc>/tracks/zeta250_back/track_centers_{pol}.csv
  outputs/<lc>/composites/zeta250_back/{pol}_composite.nc
  outputs/<lc>/projections/zeta250_back/data/tilt_{pol}.npz
  outputs/<lc>/projections/zeta250_back/plots/theta_tilt_{pol}.png
  outputs/<lc>/projections/zeta250_back/plots/theta_tilt_accum_{pol}.png
  outputs/<lc>/projections/zeta250_back/plots/tilt_animation_{pol}.mp4
  outputs/<lc>/projections/zeta250_back/idealized_plot/<lc>_idealized_3row_{pol}.png
  outputs/<lc>/plots/zeta250_tracked_extended_backward.mp4   ← C + AC
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation

import pvtend

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa: E402
from _grad_safe import safe_gradient  # noqa: E402
from select_top6 import great_circle_deg, write_tracks  # noqa: E402
from tilt_evolution import (  # noqa: E402
    _ellipse_axis_endpoints, _fill, _fit_ellipse, _make_animation,
    _strict_mask, _unwrap_mod180, _wrap_diff,
)
from track_utils import parse_stitchnodes  # noqa: E402
import idealized_plot as IDEAL  # noqa: E402

# ---------------------------------------------------------------------------
ROOT = Path("/net/flood/data2/users/x_yan/literature_review/rwb/thorncroft93_baroclinic/"
            "thorncroft_rwb")
M_PER_DEG = 111_000.0
METHOD = "zeta250"
METHOD_TAG = "zeta250_back"

# Backward walk: search radius around last known centre and per-hour jump cap.
# v2.9.0: tightened to prevent the walk from jumping to a spurious nearby
# feature once the original feature dissolves into the meridional background.
R_SEARCH_DEG = 3.5           # ≤ STITCH_RANGE_DEG of standard tracker
JUMP_MAX_DEG_PER_H = 3.5     # was 7.0 (too loose); typical RWB drift ≤ 3°/h
MIN_LAT_BACK = float(CFG.LAT_MIN)
MAX_LAT_BACK = float(CFG.LAT_MAX)

# In the pre-development phase the feature is weaker than ``mask_thresh``.
# We loosen the magnitude gate to ``BACK_THRESH_FACTOR × mask_thresh`` and
# instead lean on (a) the jump-per-hour cap, (b) cumulative-drift cap from
# the original first-tracked position, and (c) a magnitude-continuity gate
# that stops the walk if the candidate point is markedly weaker than the
# previous one (catching jumps to spurious neighbouring extrema).
BACK_THRESH_FACTOR = 0.4
MAX_TOTAL_DRIFT_DEG = 25.0   # great-circle from the original first point
BACK_MIN_VAL_FRACTION = 0.5  # |val_new| ≥ 0.5·|val_prev| OR |val_new| ≥ 0.5·thr

# Earliest backward start per LC (don't walk before these days).
WINDOW_BACK_BY_LC: dict[str, tuple[int, int]] = {
    "lc1": (2 * 24, 10 * 24),   # day 2 → day 10
    "lc2": (4 * 24, 12 * 24),   # day 4 → day 12
}


# ---------------------------------------------------------------------------
# Step 0 — input field
# ---------------------------------------------------------------------------
def _load_extended_field(lc: str):
    """Return (zeta_anom, zeta_total, lat, lon, times, hours_since_t0).

    Uses the already-built ``zeta250_anom_back.nc`` covering the full
    extended backward window. Both arrays are sorted S→N in latitude.
    """
    nc = ROOT / "outputs" / lc / "zeta250_anom_back.nc"
    if not nc.exists():
        raise FileNotFoundError(
            f"{nc} missing — re-run prep step (zeta250_anom_back.nc)")
    ds = xr.open_dataset(nc)
    spec = CFG.METHOD[METHOD]
    anom = ds[spec["var"]]
    total = ds[spec["total_var"]]
    if float(anom["lat"][0]) > float(anom["lat"][-1]):
        anom = anom.isel(lat=slice(None, None, -1))
        total = total.isel(lat=slice(None, None, -1))
    return ds, anom, total


# ---------------------------------------------------------------------------
# Step 1 — Linear-interp missing hours inside the existing track
# ---------------------------------------------------------------------------
def _interp_missing(track: list[dict]) -> list[dict]:
    """Hourly resample ``track`` between t_first and t_last with linear
    interpolation in (lon, lat, val). Lon is unwrapped before interpolating.
    """
    if len(track) < 2:
        return list(track)
    track = sorted(track, key=lambda p: p["time"])
    t_arr = np.array([(p["time"] - track[0]["time"]).total_seconds() / 3600.0
                      for p in track])
    lon_arr = np.array([p["lon"] for p in track], dtype=float)
    lat_arr = np.array([p["lat"] for p in track], dtype=float)
    val_arr = np.array([p["val"] for p in track], dtype=float)
    lon_unwrap = np.rad2deg(np.unwrap(np.deg2rad(lon_arr)))

    n_hours = int(round(t_arr[-1] - t_arr[0])) + 1
    t_target = np.arange(n_hours, dtype=float)
    lon_i = np.interp(t_target, t_arr, lon_unwrap) % 360.0
    lat_i = np.interp(t_target, t_arr, lat_arr)
    val_i = np.interp(t_target, t_arr, val_arr)

    out = []
    t0 = track[0]["time"]
    for k in range(n_hours):
        out.append({
            "time": t0 + timedelta(hours=int(t_target[k])),
            "lon":  float(lon_i[k]),
            "lat":  float(lat_i[k]),
            "val":  float(val_i[k]),
        })
    return out


# ---------------------------------------------------------------------------
# Step 2 — Walk backward from the first point
# ---------------------------------------------------------------------------
def _local_extremum(anom_da: xr.DataArray, t_idx: int,
                    lon0: float, lat0: float, polarity: str,
                    radius_deg: float = R_SEARCH_DEG,
                    thresh_factor: float = BACK_THRESH_FACTOR):
    """Return (lon, lat, val) of the local max (C) / min (AC) within a
    ``±radius_deg`` lon/lat box around (lon0, lat0). Returns None if box
    is empty or the extremum doesn't satisfy the polarity sign threshold.
    """
    spec = CFG.METHOD[METHOD]
    op, thresh = spec[polarity]["thresh"]
    thresh = thresh * thresh_factor
    lat = anom_da["lat"].values
    lon = anom_da["lon"].values
    field = anom_da.isel(time=t_idx).values

    # Latitude box (no wrap)
    lat_lo = max(MIN_LAT_BACK, lat0 - radius_deg)
    lat_hi = min(MAX_LAT_BACK, lat0 + radius_deg)
    j_mask = (lat >= lat_lo) & (lat <= lat_hi)
    if not j_mask.any():
        return None
    j_idx = np.where(j_mask)[0]

    # Longitude box (handle 0/360 wrap by computing signed angular distance)
    dlon = ((lon - lon0 + 180.0) % 360.0) - 180.0
    i_mask = np.abs(dlon) <= radius_deg
    if not i_mask.any():
        return None
    i_idx = np.where(i_mask)[0]

    sub = field[np.ix_(j_idx, i_idx)]
    if not np.isfinite(sub).any():
        return None

    if polarity == "C":
        flat_arg = np.nanargmax(sub)
    else:
        flat_arg = np.nanargmin(sub)
    jj, ii = np.unravel_index(flat_arg, sub.shape)
    val = float(sub[jj, ii])
    if op == ">" and not (val > thresh):
        return None
    if op == "<" and not (val < thresh):
        return None
    return float(lon[i_idx[ii]]), float(lat[j_idx[jj]]), val


def _walk_backward(track: list[dict], anom_da: xr.DataArray,
                   times: np.ndarray, polarity: str,
                   t_min: datetime, max_back_h: int):
    """Prepend hourly points walked backward from track[0] until a gate
    fails or t_min is reached.

    Gates (any failing → stop walk):
        1. magnitude > BACK_THRESH_FACTOR × mask_thresh
        2. per-hour jump < JUMP_MAX_DEG_PER_H
        3. cumulative drift from anchor (original first point) < MAX_TOTAL_DRIFT_DEG
        4. magnitude continuity:
              |val_new| ≥ BACK_MIN_VAL_FRACTION · |val_prev|
              OR |val_new| ≥ 0.5 · mask_thresh
           (catches jumps to a much weaker spurious neighbouring extremum)
    """
    spec = CFG.METHOD[METHOD]
    mask_thr = float(spec["mask_thresh"])
    extended = list(track)
    if not extended:
        return extended
    prev = extended[0]
    anchor_lon, anchor_lat = prev["lon"], prev["lat"]

    for step in range(1, max_back_h + 1):
        t_target = prev["time"] - timedelta(hours=1)
        if t_target < t_min:
            break

        # Map to time index in anom_da
        t_np = np.datetime64(t_target)
        diffs = np.abs(times - t_np)
        ti = int(np.argmin(diffs))
        # If the closest available time is further than 30 min away, abort.
        if diffs[ti] > np.timedelta64(30, "m"):
            break

        result = _local_extremum(anom_da, ti, prev["lon"], prev["lat"],
                                 polarity)
        if result is None:
            break
        lon_n, lat_n, val_n = result

        # Gate 2 — per-hour jump
        d_jump = great_circle_deg(prev["lon"], prev["lat"], lon_n, lat_n)
        if d_jump > JUMP_MAX_DEG_PER_H:
            break
        # Gate 3 — cumulative drift from anchor
        d_drift = great_circle_deg(anchor_lon, anchor_lat, lon_n, lat_n)
        if d_drift > MAX_TOTAL_DRIFT_DEG:
            break
        # Gate 4 — magnitude continuity vs previous tracked point
        prev_mag = abs(prev["val"])
        new_mag = abs(val_n)
        if (new_mag < BACK_MIN_VAL_FRACTION * prev_mag
                and new_mag < 0.5 * mask_thr):
            break

        new_pt = {"time": t_target, "lon": lon_n, "lat": lat_n,
                  "val": val_n}
        extended.insert(0, new_pt)
        prev = new_pt

    return extended


# ---------------------------------------------------------------------------
# Step 3 — Build extended tracks for one polarity
# ---------------------------------------------------------------------------
def build_extended_tracks(lc: str, polarity: str = "C"):
    """Read existing top-6, interp gaps, walk backward; write new file."""
    src = ROOT / "outputs" / lc / "tracks" / METHOD / f"tracks_{polarity}_top6.txt"
    if not src.exists():
        raise FileNotFoundError(src)
    tracks = parse_stitchnodes(src)
    print(f"[{lc}:{polarity}] read {len(tracks)} top-6 tracks; "
          f"first points day "
          f"{[(t[0]['time'] - datetime(2000,1,1)).total_seconds() / 86400.0 for t in tracks]}")

    ds, anom, _ = _load_extended_field(lc)
    times = anom["time"].values
    h0_sim = WINDOW_BACK_BY_LC[lc][0]
    t_min = datetime(2000, 1, 1) + timedelta(hours=h0_sim)
    max_back_h = WINDOW_BACK_BY_LC[lc][1] - h0_sim   # generous cap

    extended_tracks = []
    for tid, tr in enumerate(tracks):
        n0 = len(tr)
        tr_filled = _interp_missing(tr)
        n1 = len(tr_filled)
        tr_extended = _walk_backward(tr_filled, anom, times,
                                     polarity, t_min, max_back_h)
        n2 = len(tr_extended)
        first_old = tr[0]["time"]
        first_new = tr_extended[0]["time"]
        back_h = (first_old - first_new).total_seconds() / 3600.0
        gap_h  = n1 - n0
        print(f"  [tid={tid}] {n0} pts → +{gap_h} interp → +{back_h:.0f}h back "
              f"({n2} pts; from {first_new} to {tr_extended[-1]['time']})")
        extended_tracks.append(tr_extended)
    ds.close()

    # Save extended top-6
    out_dir = ROOT / "outputs" / lc / "tracks" / METHOD_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_top6 = out_dir / f"tracks_{polarity}_top6.txt"
    write_tracks(extended_tracks, dst_top6)
    print(f"[{lc}:{polarity}] wrote {dst_top6}")
    return extended_tracks


# ---------------------------------------------------------------------------
# Step 4 — Export per-hour CSV (consumed by the composite builder)
# ---------------------------------------------------------------------------
def _local_area_deg2(field2d, op, thr, lat_c, lon_c, lat1d, lon1d,
                     radius_deg=10.0):
    j = (lat1d >= lat_c - radius_deg) & (lat1d <= lat_c + radius_deg)
    dlon = ((lon1d - lon_c + 180.0) % 360.0) - 180.0
    i = np.abs(dlon) <= radius_deg
    if not (j.any() and i.any()):
        return 0.0
    sub = field2d[np.ix_(np.where(j)[0], np.where(i)[0])]
    if op == ">":
        m = sub > thr
    else:
        m = sub < thr
    return float(m.sum()) * (1.0 ** 2)   # DX=1°


def export_csv(lc: str, tracks: list[list[dict]], polarity: str = "C"):
    spec = CFG.METHOD[METHOD]
    op, thr = spec[polarity]["thresh"]
    ds, anom, _ = _load_extended_field(lc)
    times = anom["time"].values
    lat1d = anom["lat"].values
    lon1d = anom["lon"].values
    out_dir = ROOT / "outputs" / lc / "tracks" / METHOD_TAG
    csv_path = out_dir / f"track_centers_{polarity}.csv"
    n_rows = 0
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lc", "method", "track_id", "time_iso", "day_hour",
                    "lon", "lat", "val", "area_deg2"])
        for tid, tr in enumerate(tracks):
            for p in tr:
                t_np = np.datetime64(p["time"])
                ti = int(np.argmin(np.abs(times - t_np)))
                vf = anom.isel(time=ti).values
                a = _local_area_deg2(vf, op, thr, p["lat"], p["lon"],
                                     lat1d, lon1d, radius_deg=10.0)
                dh = (p["time"] - datetime(2000, 1, 1)).total_seconds() / 3600.0
                w.writerow([lc, METHOD_TAG, tid, p["time"].isoformat(),
                            f"{dh:.1f}", f"{p['lon']:.4f}",
                            f"{p['lat']:.4f}", f"{p['val']:.6e}",
                            f"{a:.3f}"])
                n_rows += 1
    ds.close()
    print(f"[{lc}:{polarity}] CSV → {csv_path}  ({n_rows} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# Step 5 — Polar-cap MP4 (same style as zeta250_tracked.mp4)
# ---------------------------------------------------------------------------
def _to_dt(np_dt64) -> datetime:
    s = (np_dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(float(s))


def render_polar_cap_anim(lc: str, tracks_C: list[list[dict]],
                          tracks_AC: list[list[dict]] | None = None,
                          stride: int = 1, trail_hours: int = 24):
    """Polar-cap zeta250 anomaly with the extended tracks overlaid.

    Saves a single MP4 per LC at::

        outputs/<lc>/plots/zeta250_tracked_extended_backward.mp4
    """
    ds, anom, _ = _load_extended_field(lc)
    lat = anom["lat"].values
    lon = anom["lon"].values
    lon_wrap = np.append(lon, lon[0] + 360.0)
    times = anom["time"].values
    frames = list(range(0, len(times), stride))

    vmax = float(np.nanpercentile(np.abs(anom.values), 99.0))
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

    def _frame(idx):
        d = anom.isel(time=idx).values
        return np.concatenate([d, d[:, :1]], axis=1)

    cf = [ax.contourf(lon_wrap, lat, _frame(0), levels=levels,
                      cmap=cmap, extend="both", transform=data_crs)]
    cb = fig.colorbar(cf[0], ax=ax, shrink=0.75, pad=0.05,
                      orientation="vertical")
    cb.set_label(r"$\zeta'_{250\,hPa}$  [s$^{-1}$]")
    title = ax.set_title("", fontsize=11)
    artists = []

    standard_start_dt = (datetime(2000, 1, 1)
                         + timedelta(hours=CFG.WINDOW_BY_LC[lc][0]))

    def _draw(frame_idx):
        try:
            cf[0].remove()
        except Exception:
            pass
        for a in artists:
            try: a.remove()
            except Exception: pass
        artists.clear()

        t = times[frame_idx]; t_dt = _to_dt(t)
        day = (t_dt - datetime(2000, 1, 1)).total_seconds() / 86400.0
        cf[0] = ax.contourf(lon_wrap, lat, _frame(frame_idx),
                            levels=levels, cmap=cmap, extend="both",
                            transform=data_crs)

        def _draw_set(all_tracks, full_colour, back_colour):
            if not all_tracks:
                return
            for tr in all_tracks:
                pts = [p for p in tr if p["time"] <= t_dt
                       and (t_dt - p["time"]).total_seconds() / 3600.0
                       <= trail_hours + 0.1]
                if not pts:
                    continue
                # Split trail into back-extended (before standard onset) vs
                # standard. Keep the visual cue but don't break continuity.
                back_pts = [p for p in pts if p["time"] < standard_start_dt]
                fwd_pts  = [p for p in pts if p["time"] >= standard_start_dt]
                if back_pts:
                    bl = back_pts + (fwd_pts[:1] if fwd_pts else [])
                    lons = np.array([p["lon"] for p in bl], dtype=float)
                    lats = np.array([p["lat"] for p in bl], dtype=float)
                    lons = np.rad2deg(np.unwrap(np.deg2rad(lons)))
                    ln, = ax.plot(lons, lats, "-", color=back_colour,
                                  lw=1.6, alpha=0.95, transform=data_crs)
                    artists.append(ln)
                if fwd_pts:
                    lons = np.array([p["lon"] for p in fwd_pts], dtype=float)
                    lats = np.array([p["lat"] for p in fwd_pts], dtype=float)
                    lons = np.rad2deg(np.unwrap(np.deg2rad(lons)))
                    ln, = ax.plot(lons, lats, "-", color=full_colour,
                                  lw=1.0, alpha=0.95, transform=data_crs)
                    artists.append(ln)
                if pts[-1]["time"] == t_dt:
                    is_back = pts[-1]["time"] < standard_start_dt
                    sc = ax.scatter([pts[-1]["lon"]], [pts[-1]["lat"]],
                                    c=back_colour if is_back else full_colour,
                                    s=32, edgecolor="k", lw=0.5, zorder=5,
                                    transform=data_crs)
                    artists.append(sc)

        _draw_set(tracks_C,  full_colour="blue",   back_colour="cyan")
        if tracks_AC:
            _draw_set(tracks_AC, full_colour="orange", back_colour="gold")

        std_day = CFG.WINDOW_BY_LC[lc][0] / 24.0
        marker = "← BACKWARD" if day < std_day else "FORWARD"
        title.set_text(
            f"{lc.upper()}  {METHOD_TAG}  day {day:5.2f}  ({marker})\n"
            "C blue (cyan=back), AC orange (gold=back); std onset day "
            f"{std_day:.0f}")
        return ()

    anim = FuncAnimation(fig, _draw, frames=frames, interval=120, blit=False)
    out_plots = ROOT / "outputs" / lc / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)
    mp4 = out_plots / "zeta250_tracked_extended_backward.mp4"
    print(f"[{lc}] writing {mp4}  ({len(frames)} frames)")
    anim.save(mp4, writer=FFMpegWriter(fps=CFG.ANIM_FPS, bitrate=2400))
    plt.close(fig)
    ds.close()
    print(f"[{lc}] done → {mp4}")


# ---------------------------------------------------------------------------
# Step 6 — Lagrangian composites from extended tracks
# ---------------------------------------------------------------------------
def _pad_periodic_lon(da: xr.DataArray) -> xr.DataArray:
    lon = da["lon"].values
    left = da.isel(lon=[-1]).assign_coords(lon=[float(lon[-1]) - 360.0])
    right = da.isel(lon=[0]).assign_coords(lon=[float(lon[0]) + 360.0])
    return xr.concat([left, da, right], dim="lon")


def _pole_reflect(lat_q, lon_q):
    lr, ll = lat_q.copy(), lon_q.copy()
    over_n = lr > 90.0;  lr[over_n] = 180.0 - lr[over_n]; ll[over_n] = (ll[over_n] + 180.0) % 360.0
    over_s = lr < -90.0; lr[over_s] = -180.0 - lr[over_s]; ll[over_s] = (ll[over_s] + 180.0) % 360.0
    return lr, ll


def build_composites(lc: str, polarity: str = "C"):
    """Read CSV, sample 81×61 patches around each track centre per hour."""
    csv_path = (ROOT / "outputs" / lc / "tracks" / METHOD_TAG
                / f"track_centers_{polarity}.csv")
    if not csv_path.exists():
        print(f"[{lc}:{polarity}] CSV missing; skipping composites")
        return None
    df = pd.read_csv(csv_path, parse_dates=["time_iso"])

    # Composite window: from the earliest track point overall to the latest.
    t_first = pd.to_datetime(df["time_iso"].min()).to_pydatetime()
    t_last  = pd.to_datetime(df["time_iso"].max()).to_pydatetime()
    n_hours = int((t_last - t_first).total_seconds() / 3600.0) + 1
    win_start = t_first
    print(f"[{lc}:{polarity}] composite window: {t_first} → {t_last}  ({n_hours} h)")

    ds, anom, total = _load_extended_field(lc)
    times = anom["time"].values
    total_pad = _pad_periodic_lon(total)
    anom_pad  = _pad_periodic_lon(anom)

    x_rel = np.arange(-CFG.PATCH_HALF_LON, CFG.PATCH_HALF_LON + CFG.DX / 2, CFG.DX)
    y_rel = np.arange(-CFG.PATCH_HALF_LAT, CFG.PATCH_HALF_LAT + CFG.DX / 2, CFG.DX)
    nx, ny = len(x_rel), len(y_rel)
    Xr, Yr = np.meshgrid(x_rel, y_rel)

    shape_m = (CFG.TOP_N, n_hours, ny, nx)
    members_total = np.full(shape_m, np.nan, dtype="float32")
    members_anom  = np.full(shape_m, np.nan, dtype="float32")
    track_lon = np.full((CFG.TOP_N, n_hours), np.nan, dtype="float32")
    track_lat = np.full((CFG.TOP_N, n_hours), np.nan, dtype="float32")

    for tid, grp in df.groupby("track_id"):
        tid_i = int(tid)
        if tid_i >= CFG.TOP_N:
            continue
        grp = grp.sort_values("time_iso").reset_index(drop=True)
        for _, row in grp.iterrows():
            tt = pd.to_datetime(row["time_iso"]).to_pydatetime()
            ihour = int((tt - win_start).total_seconds() / 3600)
            if ihour < 0 or ihour >= n_hours:
                continue
            lat_samp = float(row["lat"]) + Yr
            lon_samp = (float(row["lon"]) + Xr) % 360.0
            lat_r, lon_r = _pole_reflect(lat_samp.ravel(), lon_samp.ravel())
            lat_da = xr.DataArray(lat_r, dims="p")
            lon_da = xr.DataArray(lon_r, dims="p")
            ti = int(np.argmin(np.abs(times - np.datetime64(tt))))
            try:
                pt = total_pad.isel(time=ti).interp(
                    lat=lat_da, lon=lon_da, method="linear"
                ).values.reshape(ny, nx).astype("float32")
                pa = anom_pad.isel(time=ti).interp(
                    lat=lat_da, lon=lon_da, method="linear"
                ).values.reshape(ny, nx).astype("float32")
            except Exception as exc:
                print(f"  [{tid_i} ihour={ihour}] interp fail: {exc}")
                continue
            members_total[tid_i, ihour] = pt
            members_anom[tid_i, ihour]  = pa
            track_lon[tid_i, ihour] = float(row["lon"])
            track_lat[tid_i, ihour] = float(row["lat"])

    ds.close()
    comp_total = np.nanmean(members_total, axis=0).astype("float32")
    comp_anom  = np.nanmean(members_anom,  axis=0).astype("float32")
    n_per_t = np.sum(~np.isnan(members_total[..., 0, 0]), axis=0).astype("int8")

    win_h0_sim = int((win_start - datetime(2000, 1, 1)).total_seconds() / 3600.0)
    out = xr.Dataset(
        {
            "total_composite": (("t", "y", "x"), comp_total),
            "anom_composite":  (("t", "y", "x"), comp_anom),
            "total_members":   (("member", "t", "y", "x"), members_total),
            "anom_members":    (("member", "t", "y", "x"), members_anom),
            "track_lon":       (("member", "t"), track_lon),
            "track_lat":       (("member", "t"), track_lat),
            "n_members":       (("t",), n_per_t),
        },
        coords={
            "t":      ("t", np.arange(n_hours, dtype="int16")),
            "y":      ("y", y_rel.astype("float32")),
            "x":      ("x", x_rel.astype("float32")),
            "member": ("member", np.arange(CFG.TOP_N, dtype="int8")),
        },
        attrs={
            "lc": lc, "method": METHOD_TAG, "polarity": polarity,
            "patch_half_lon_deg": float(CFG.PATCH_HALF_LON),
            "patch_half_lat_deg": float(CFG.PATCH_HALF_LAT),
            "dx_deg": float(CFG.DX),
            "win_h0_sim": win_h0_sim,
            "standard_onset_h": int(CFG.WINDOW_BY_LC[lc][0]),
        },
    )
    dst_dir = ROOT / "outputs" / lc / "composites" / METHOD_TAG
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{polarity}_composite.nc"
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(dst, encoding=enc)
    print(f"[{lc}:{polarity}] composite → {dst}  shape {comp_total.shape}  "
          f"non-NaN frames {(n_per_t > 0).sum()}")
    return dst


# ---------------------------------------------------------------------------
# Step 7 — Tilt analysis on extended composite (same logic as tilt_evolution)
# ---------------------------------------------------------------------------
def run_tilt(lc: str, polarity: str = "C"):
    comp_nc = (ROOT / "outputs" / lc / "composites" / METHOD_TAG
               / f"{polarity}_composite.nc")
    if not comp_nc.exists():
        print(f"[{lc}:{polarity}] missing {comp_nc}; skipping tilt")
        return

    ds = xr.open_dataset(comp_nc)
    q  = ds["total_composite"].values.astype("float64")
    qa = ds["anom_composite"].values.astype("float64")
    t_hour = ds["t"].values
    x_rel  = ds["x"].values.astype("float64")
    y_rel  = ds["y"].values.astype("float64")
    win_h0 = int(ds.attrs.get("win_h0_sim", 0))
    std_onset_h = int(ds.attrs.get("standard_onset_h", win_h0))
    ds.close()

    X, Y = np.meshgrid(x_rel, y_rel)
    nt = q.shape[0]
    spec = CFG.METHOD[METHOD]
    fit_thr = float(spec["mask_thresh"])
    guard_r = float(CFG.PATCH_HALF_LAT - CFG.GUARD_PAD_DEG)
    mask_str = "> 0" if polarity == "C" else "< 0"
    dx_m = CFG.DX * M_PER_DEG * np.cos(np.deg2rad(CFG.CENTER_LAT))
    dy_m = CFG.DX * M_PER_DEG

    pv_dt = np.full_like(q, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2 * 3600.0)

    theta_obs = np.full(nt, np.nan)
    a_obs = np.full(nt, np.nan); b_obs = np.full(nt, np.nan)
    xc_obs = np.full(nt, np.nan); yc_obs = np.full(nt, np.nan)
    theta_pred = np.full(nt, np.nan)
    a_pred = np.full(nt, np.nan); b_pred = np.full(nt, np.nan)
    xc_pred = np.full(nt, np.nan); yc_pred = np.full(nt, np.nan)
    deform_field = np.full_like(q, np.nan)
    mask_frames  = np.zeros_like(q, dtype=bool)

    for i in range(nt):
        qai = qa[i]
        if not np.isfinite(qai).any():
            continue
        qai = _fill(qai); qi = _fill(q[i])
        m = _strict_mask(qai, polarity, fit_thr, X, Y, guard_r)
        if m.sum() < 8:
            continue
        mask_frames[i] = m
        th, a_, b_, xcc, ycc = _fit_ellipse(m, qai, X, Y)
        theta_obs[i] = th; a_obs[i] = a_; b_obs[i] = b_
        xc_obs[i] = xcc; yc_obs[i] = ycc

        if i in (0, nt - 1) or not np.isfinite(pv_dt[i]).any():
            continue
        qdx, qdy = safe_gradient(q[i], dy_m, dx_m)
        try:
            basis = pvtend.compute_orthogonal_basis(
                qai, qdx, qdy, x_rel, y_rel,
                mask=mask_str,
                apply_smoothing=True, smoothing_deg=CFG.SMOOTHING_DEG,
                grid_spacing=CFG.DX, center_lat=CFG.CENTER_LAT,
                include_lap=CFG.INCLUDE_LAP,
            )
            proj = pvtend.project_field(_fill(pv_dt[i]), basis,
                                        grid_spacing=CFG.DX)
            deform_field[i] = proj["def"]
        except Exception as exc:
            print(f"  [t={int(t_hour[i])}h] basis/proj failed: {exc}")
            continue
        dt_s = CFG.DT_PRED_HOURS * 3600.0
        qa_next = qai + dt_s * proj["recon"]
        m2 = _strict_mask(qa_next, polarity, fit_thr, X, Y, guard_r)
        if m2.sum() < 8:
            continue
        th2, a2, b2, xc2, yc2 = _fit_ellipse(m2, qa_next, X, Y)
        theta_pred[i] = th2; a_pred[i] = a2; b_pred[i] = b2
        xc_pred[i] = xc2; yc_pred[i] = yc2

    data_dir = ROOT / "outputs" / lc / "projections" / METHOD_TAG / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / f"tilt_{polarity}.npz",
             t_hour=t_hour, win_h0_sim=win_h0,
             theta_obs=theta_obs, theta_pred=theta_pred,
             a_obs=a_obs, b_obs=b_obs, a_pred=a_pred, b_pred=b_pred,
             xc_obs=xc_obs, yc_obs=yc_obs, xc_pred=xc_pred, yc_pred=yc_pred)

    plots = ROOT / "outputs" / lc / "projections" / METHOD_TAG / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    # v2.9.0: x-axis is now "hours since 2000-01-01 00:00" (day 5 → hour 120).
    hours_axis = (win_h0 + t_hour).astype("float64")
    std_hour = float(std_onset_h)

    theta_obs_uw  = _unwrap_mod180(theta_obs)
    theta_pred_uw = _unwrap_mod180(theta_pred)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours_axis, theta_obs_uw,  "g-",  lw=2,   label="obs")
    ax.plot(hours_axis, theta_pred_uw, "c--", lw=1.5, label="pred (+1 h)")
    ax.axvline(std_hour, color="k", lw=1.2, ls="--",
               label=f"std onset (h={std_hour:.0f}, day {std_hour/24:.0f})")
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(90, color="grey", lw=0.4, ls=":")
    ax.axhline(-90, color="grey", lw=0.4, ls=":")
    ax.set_xlabel("hours since simulation start (day 0)")
    ax.set_ylabel("θ tilt [deg, unwrapped]")
    ax.set_title(f"{lc.upper()} {METHOD_TAG}/{polarity}  "
                 f"ellipse tilt (extended backward)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / f"theta_tilt_{polarity}.png", dpi=140)
    plt.close(fig)

    dth_obs = np.zeros(nt); dth_pred = np.zeros(nt)
    for i in range(1, nt):
        if np.isfinite(theta_obs[i]) and np.isfinite(theta_obs[i - 1]):
            dth_obs[i] = dth_obs[i - 1] + _wrap_diff(theta_obs[i],
                                                    theta_obs[i - 1])
        else:
            dth_obs[i] = dth_obs[i - 1]
        if np.isfinite(theta_pred[i - 1]) and np.isfinite(theta_obs[i - 1]):
            dth_pred[i] = dth_pred[i - 1] + _wrap_diff(theta_pred[i - 1],
                                                      theta_obs[i - 1])
        else:
            dth_pred[i] = dth_pred[i - 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours_axis, dth_obs,  "g-",  lw=2,   label="∑Δθ obs")
    ax.plot(hours_axis, dth_pred, "c--", lw=1.5, label="∑Δθ pred (γ·Φ_def)")
    ax.axvline(std_hour, color="k", lw=1.2, ls="--",
               label=f"std onset (h={std_hour:.0f}, day {std_hour/24:.0f})")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("hours since simulation start (day 0)")
    ax.set_ylabel("accumulated tilt change [deg]")
    ax.set_title(f"{lc.upper()} {METHOD_TAG}/{polarity}  "
                 f"accumulated tilt (extended backward)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / f"theta_tilt_accum_{polarity}.png", dpi=140)
    plt.close(fig)

    # v2.9.0: animation x-axis hours-from-day-0 (sim start).
    t_hour_disp = (np.asarray(t_hour, dtype="float64") + float(win_h0))
    _make_animation(lc, METHOD_TAG, polarity,
                    q, qa, pv_dt, deform_field,
                    mask_frames, X, Y, x_rel, y_rel,
                    t_hour_disp, theta_obs, a_obs, b_obs, xc_obs, yc_obs,
                    theta_pred, a_pred, b_pred, xc_pred, yc_pred,
                    fit_thr, guard_r, mask_str, dx_m, dy_m, plots)
    print(f"[{lc}:{polarity}] tilt + 4-panel animation done.")


def run_idealized_back(lc: str, polarity: str = "C"):
    """Idealized 3-row decomposition figure on the backward-extended composite.

    Picks the projection frame at ``t_k = round(0.5 * 24) = 12`` h after
    the earliest backward composite frame (i.e. 0.5 day after the
    earliest timestamp of the full backward+forward track window).
    Output:  outputs/<lc>/projections/zeta250_back/idealized_plot/<lc>_idealized_3row_<pol>.png
    """
    comp_nc = (ROOT / "outputs" / lc / "composites" / METHOD_TAG
               / f"{polarity}_composite.nc")
    if not comp_nc.exists():
        print(f"[{lc}:{polarity}] {comp_nc} missing; skip idealized")
        return
    with xr.open_dataset(comp_nc) as ds:
        nt = ds.dims["t"]
    t_k = 12  # 0.5 day after the earliest backward frame
    if t_k > nt - 2:
        t_k = max(1, nt - 2)
    IDEAL.process(lc, polarity, method=METHOD_TAG, t_k=t_k)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _load_extended_tracks_from_disk(lc: str, polarity: str):
    """Load already-built extended top-6 tracks for a polarity from disk
    (so a single-polarity rerun can still draw both on the polar-cap MP4).
    Returns [] if missing.
    """
    p = (ROOT / "outputs" / lc / "tracks" / METHOD_TAG
         / f"tracks_{polarity}_top6.txt")
    if not p.exists():
        return []
    try:
        return parse_stitchnodes(p)
    except Exception:
        return []


def process(lc: str, polarities: list[str]):
    print(f"\n{'=' * 60}\n  {lc.upper()}  build extended tracks\n{'=' * 60}")
    extended_by_pol: dict[str, list] = {}
    for pol in polarities:
        tracks = build_extended_tracks(lc, pol)
        export_csv(lc, tracks, pol)
        extended_by_pol[pol] = tracks

    # Always include both polarities on the polar-cap MP4: pull the other
    # polarity's existing extended tracks from disk if it wasn't rebuilt now.
    tracks_C  = extended_by_pol.get("C")  or _load_extended_tracks_from_disk(lc, "C")
    tracks_AC = extended_by_pol.get("AC") or _load_extended_tracks_from_disk(lc, "AC")
    render_polar_cap_anim(lc, tracks_C, tracks_AC)

    # Composites + tilt per polarity actually requested this run
    for pol in polarities:
        build_composites(lc, pol)
        run_tilt(lc, pol)
        run_idealized_back(lc, pol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="C")
    args = ap.parse_args()
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        process(lc, pols)
