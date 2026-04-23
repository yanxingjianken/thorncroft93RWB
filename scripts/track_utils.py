"""Shared utilities: parse StitchNodes, pick top-N tracks per polarity.

Thorncroft LC runs have a wave-6 perturbation, so we expect six
cyclones (vor250 max) and six anticyclones (vor250 min). Ranking by
track length then peak |vor250| isolates the six dominant ridges/
troughs even when DetectNodes splits a ridge into multiple short
candidates.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_stitchnodes(path: Path):
    tracks = []
    cur = []
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
            try:
                lon = float(toks[2]); lat = float(toks[3]); val = float(toks[4])
                yr, mo, dy, hr = map(int, toks[5:9])
            except (ValueError, IndexError):
                continue
            cur.append({"time": datetime(yr, mo, dy, hr),
                        "lon": lon, "lat": lat, "val": val})
        if cur:
            tracks.append(cur)
    return tracks


def keep_top_n(tracks, n=6):
    def key(tr):
        if not tr:
            return (0, 0.0)
        peak = max(abs(p["val"]) for p in tr)
        return (len(tr), peak)
    return sorted(tracks, key=key, reverse=True)[:n]


def tracks_by_time(tracks):
    idx = defaultdict(list)
    for tid, tr in enumerate(tracks):
        for pt in tr:
            idx[pt["time"]].append((tid, pt))
    return idx


def write_top6(src: Path, dst: Path, n=6):
    tr = keep_top_n(parse_stitchnodes(src), n)
    with open(dst, "w") as fh:
        for t in tr:
            fh.write(f"start\t{len(t)}\t{t[0]['time'].year}\t"
                     f"{t[0]['time'].month}\t{t[0]['time'].day}\t"
                     f"{t[0]['time'].hour}\n")
            for p in t:
                # placeholder i j (0 0); not needed downstream
                fh.write(f"\t0\t0\t{p['lon']:.6f}\t{p['lat']:.6f}\t"
                         f"{p['val']:.6e}\t{p['time'].year}\t"
                         f"{p['time'].month}\t{p['time'].day}\t"
                         f"{p['time'].hour}\n")
    return tr
