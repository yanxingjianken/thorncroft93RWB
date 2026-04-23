"""Select top-6 PV-anom tracks per LC on the 330 K isentrope.

Processes both polarities:
  C  (cyclonic):     tracks_max_pv330.txt -> tracks_max_top6_pv330.txt
  AC (anticyclonic): tracks_min_pv330.txt -> tracks_min_top6_pv330.txt

Criteria:
  1. Track span (last.time - first.time) >= SPAN_REQ_H (default 90 h).
  2. Max hourly great-circle displacement <= JUMP_MAX_DEG (5 deg/h).
  3. Rank surviving tracks by mean |pv_anom_330| desc, keep top 6.
  4. Trim selected tracks to a common time window so all top-6 tracks
     start/end at the same timestamps.
"""
from __future__ import annotations
import sys
import itertools
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from track_utils import parse_stitchnodes  # noqa
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")

SPAN_REQ_H = CFG.SPAN_REQ_H
JUMP_MAX_DEG = CFG.JUMP_MAX_DEG_PER_H   # deg/h; matches StitchNodes --range


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


def trim_to_common_window(tracks):
    """Trim tracks to common [max(start), min(end)] window."""
    if not tracks:
        return [], None, None
    t0 = max(tr[0]["time"] for tr in tracks if tr)
    t1 = min(tr[-1]["time"] for tr in tracks if tr)
    if t1 < t0:
        return [], t0, t1
    out = []
    for tr in tracks:
        clipped = [p for p in tr if t0 <= p["time"] <= t1]
        if clipped:
            out.append(clipped)
    return out, t0, t1


def mean_abs_val(track):
    return sum(abs(p["val"]) for p in track) / len(track)


def choose_best_six(scored_tracks):
    """Pick 6 tracks maximizing common overlap, then mean score.

    `scored_tracks` is a list of tuples: (track, span_h, max_jump, score).
    Preference order:
      1) common overlap >= SPAN_REQ_H (True beats False)
      2) larger common overlap span
      3) larger summed score
    """
    if len(scored_tracks) <= 6:
        return scored_tracks

    best_combo = None
    best_key = None
    for combo in itertools.combinations(scored_tracks, 6):
        tracks = [x[0] for x in combo]
        _, t0, t1 = trim_to_common_window(tracks)
        if t0 is None or t1 is None or t1 < t0:
            common_span = -1.0
        else:
            common_span = (t1 - t0).total_seconds() / 3600.0
        meets = common_span >= SPAN_REQ_H
        score_sum = sum(x[3] for x in combo)
        key = (1 if meets else 0, common_span, score_sum)
        if best_key is None or key > best_key:
            best_key = key
            best_combo = combo
    return list(best_combo)


def select(lc, polarity="C"):
    tag = "max" if polarity == "C" else "min"
    src = ROOT / lc / "tracks" / f"tracks_{tag}_pv330.txt"
    dst = ROOT / lc / "tracks" / f"tracks_{tag}_top6_pv330.txt"
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
        qual.append((tr, span_h, mj, mean_abs_val(tr)))
    qual.sort(
        key=lambda x: x[3],
        reverse=True,
    )
    chosen = choose_best_six(qual)
    top_raw = [t for t, _, _, _ in chosen]
    top, common_t0, common_t1 = trim_to_common_window(top_raw)
    write_tracks(top, dst)

    print(f"[{lc}:{polarity}] {len(all_tr)} raw -> span>={SPAN_REQ_H:.0f}h & "
          f"jump<={JUMP_MAX_DEG:.1f}\u00b0/h -> {len(qual)} qual -> "
          f"{len(top)} top6 (rejected: span={rejected_span}, "
          f"jump={rejected_jump})")
    if top and common_t0 is not None and common_t1 is not None:
        common_span_h = (common_t1 - common_t0).total_seconds() / 3600.0
        print(f"  common window: {common_t0:%m-%d %HZ} -> "
              f"{common_t1:%m-%d %HZ} (span={common_span_h:.0f}h)")

    for i, (tr, span, mj, _) in enumerate(chosen):
        peak = max(abs(p["val"]) for p in tr)
        print(f"  t{i}: n={len(tr)} span={span:.0f}h peak={peak:.2e} "
              f"lat0={tr[0]['lat']:.1f} maxjump={mj:.2f}\u00b0/h "
              f"t0={tr[0]['time']:%m-%d %HZ} t1={tr[-1]['time']:%m-%d %HZ}")
    return top


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for pol in pols:
            select(lc, polarity=pol)
