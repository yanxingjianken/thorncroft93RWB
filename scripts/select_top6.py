"""Select top-6 PV-anom tracks per LC on the 330 K isentrope.

Replaces the earlier vor250 selection.  Input file produced by
``run_tempest_pv330.sh`` is ``tracks_max_pv330.txt``.

Criteria:
  1. Track span (last.time - first.time) >= SPAN_REQ_H (default 90 h).
     Tracks may start or end anywhere inside the day-6 -> day-13
     window.
  2. Max hourly great-circle displacement <= JUMP_MAX_DEG (3.0 deg).
  3. Rank surviving tracks by mean |pv_anom_330| desc, keep top 6.

Output: outputs/<lc>/tracks/tracks_max_top6_pv330.txt
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from track_utils import parse_stitchnodes  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")

SPAN_REQ_H = 90.0
JUMP_MAX_DEG = 5.0   # deg/hr; matches StitchNodes --range in run_tempest_pv330.sh   # max great-circle deg between consecutive hourly nodes


def great_circle_deg(lon1, lat1, lon2, lat2):
    """Great-circle separation in degrees."""
    l1, l2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    cosd = np.sin(l1) * np.sin(l2) + np.cos(l1) * np.cos(l2) * np.cos(dlon)
    return float(np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0))))


def max_jump(track) -> float:
    if len(track) < 2:
        return 0.0
    m = 0.0
    for a, b in zip(track[:-1], track[1:]):
        dt_h = (b["time"] - a["time"]).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        d = great_circle_deg(a["lon"], a["lat"], b["lon"], b["lat"])
        # per-hour rate
        m = max(m, d / max(dt_h, 1.0))
    return m


def write_tracks(tracks, dst):
    with open(dst, "w") as fh:
        for t in tracks:
            fh.write(f"start\t{len(t)}\t{t[0]['time'].year}\t"
                     f"{t[0]['time'].month}\t{t[0]['time'].day}\t"
                     f"{t[0]['time'].hour}\n")
            for p in t:
                fh.write(f"\t0\t0\t{p['lon']:.6f}\t{p['lat']:.6f}\t"
                         f"{p['val']:.6e}\t{p['time'].year}\t"
                         f"{p['time'].month}\t{p['time'].day}\t"
                         f"{p['time'].hour}\n")


def select(lc):
    src = ROOT / lc / "tracks" / "tracks_max_pv330.txt"
    dst = ROOT / lc / "tracks" / "tracks_max_top6_pv330.txt"
    all_tr = parse_stitchnodes(src)
    qual, rejected_span, rejected_jump = [], 0, 0
    for tr in all_tr:
        if not tr:
            continue
        span_h = (tr[-1]["time"] - tr[0]["time"]).total_seconds() / 3600.0
        if span_h < SPAN_REQ_H:
            rejected_span += 1
            continue
        mj = max_jump(tr)
        if mj > JUMP_MAX_DEG:
            rejected_jump += 1
            continue
        qual.append((tr, span_h, mj))
    qual.sort(
        key=lambda x: sum(abs(p["val"]) for p in x[0]) / len(x[0]),
        reverse=True,
    )
    top = [t for t, _, _ in qual[:6]]
    write_tracks(top, dst)
    print(f"[{lc}] {len(all_tr)} raw -> span>={SPAN_REQ_H:.0f}h & "
          f"jump<={JUMP_MAX_DEG:.1f}\u00b0/h -> {len(qual)} qual -> "
          f"{len(top)} top6 (rejected: span={rejected_span}, "
          f"jump={rejected_jump})")
    for i, (tr, span, mj) in enumerate(qual[:6]):
        peak = max(abs(p["val"]) for p in tr)
        print(f"  t{i}: n={len(tr)} span={span:.0f}h peak={peak:.2e} "
              f"lat0={tr[0]['lat']:.1f} maxjump={mj:.2f}\u00b0/h "
              f"t0={tr[0]['time']:%m-%d %HZ} t1={tr[-1]['time']:%m-%d %HZ}")
    return top


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        select(lc)
