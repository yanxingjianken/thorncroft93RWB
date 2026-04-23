"""Select top-N tracks per LC for ONE method.

Usage:  python select_top6.py [--method M] [--polarity C|AC|both] [lcs ...]

Reads:  outputs/<lc>/tracks/<method>/tracks_{C,AC}.txt
Writes: outputs/<lc>/tracks/<method>/tracks_{C,AC}_top6.txt
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
JUMP_MAX_DEG = CFG.JUMP_MAX_DEG_PER_H
TOP_N = CFG.TOP_N


def great_circle_deg(lon1, lat1, lon2, lat2):
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


def choose_best_n(scored_tracks, n=TOP_N):
    if len(scored_tracks) <= n:
        return scored_tracks
    best_combo, best_key = None, None
    for combo in itertools.combinations(scored_tracks, n):
        tracks = [x[0] for x in combo]
        _, t0, t1 = trim_to_common_window(tracks)
        common_span = ((t1 - t0).total_seconds() / 3600.0
                       if (t0 is not None and t1 is not None and t1 >= t0)
                       else -1.0)
        meets = common_span >= SPAN_REQ_H
        score_sum = sum(x[3] for x in combo)
        key = (1 if meets else 0, common_span, score_sum)
        if best_key is None or key > best_key:
            best_key, best_combo = key, combo
    return list(best_combo)


def select(lc, method, polarity="C"):
    src = ROOT / lc / "tracks" / method / f"tracks_{polarity}.txt"
    dst = ROOT / lc / "tracks" / method / f"tracks_{polarity}_top6.txt"
    if not src.exists():
        print(f"[{lc}:{method}:{polarity}] missing {src}; skipping")
        return []
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
    qual.sort(key=lambda x: x[3], reverse=True)
    chosen = choose_best_n(qual)
    top_raw = [t for t, _, _, _ in chosen]
    top, t0c, t1c = trim_to_common_window(top_raw)
    write_tracks(top, dst)
    print(f"[{lc}:{method}:{polarity}] {len(all_tr)} raw -> "
          f"{len(qual)} qual -> {len(top)} top  (rej span={rejected_span}, "
          f"jump={rejected_jump})")
    if top and t0c is not None and t1c is not None:
        sp = (t1c - t0c).total_seconds() / 3600.0
        print(f"  common: {t0c:%m-%d %HZ} -> {t1c:%m-%d %HZ} ({sp:.0f}h)")
    return top


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--method", default=None,
                    help="one of pv330|zeta250|theta_pv2 (default: all)")
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    methods = [args.method] if args.method else CFG.METHODS
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for m in methods:
            for pol in pols:
                select(lc, m, polarity=pol)
