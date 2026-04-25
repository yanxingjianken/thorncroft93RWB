"""Shared configuration for the Thorncroft RWB projection/tilt pipeline.

v2.3.0: three-method tracking comparison (zeta250, pv330, theta_pv2).
Per-method DetectNodes thresholds + closed-contour deltas live in
``METHOD``. Day-6..10 window. Basis anchored at first track frame
(most circular). No symmetrize.
"""
from __future__ import annotations

# --- Lagrangian patch geometry ---
PATCH_HALF = 40        # degrees; half-width in x/y (±40°) — wider so even
                       # elongated late-stage anomalies fit without truncation
DX = 1.0               # degrees; grid spacing
CENTER_LAT = 50.0      # approximate reference lat (for dx metric only)
LAT_MIN = 25.0         # degrees N; DetectNodes search domain (30-70 N + 5° slack)
LAT_MAX = 75.0

# --- Common stitch / selection ---
MERGE_DIST_DEG = 6.0   # DetectNodes --mergedist
STITCH_RANGE_DEG = 5.0 # StitchNodes --range
STITCH_MAXGAP = "6h"
SPAN_REQ_H = 36   # 1.5 days; window is only 96h, longer eats real tracks        # minimum track lifespan in hours (96h window)
JUMP_MAX_DEG_PER_H = 7.0
TOP_N = 6

# --- Per-LC composite window (hours since 2000-01-01 00 UTC) ---
# LC1 day 6..10 (hours 144..240), LC2 day 7..11 (hours 168..264).
# Each window covers 96 h => N_COMPOSITE_HOURS = 97 frames.
WINDOW_BY_LC = {
    "lc1": (6 * 24, 10 * 24),
    "lc2": (7 * 24, 11 * 24),
}
# Backward-compatible defaults (used when no LC tag is available; equal LC1).
WINDOW_START_HOUR = WINDOW_BY_LC["lc1"][0]
WINDOW_END_HOUR   = WINDOW_BY_LC["lc1"][1]
N_COMPOSITE_HOURS = WINDOW_END_HOUR - WINDOW_START_HOUR + 1   # 97 frames


def window_for(lc: str) -> tuple[int, int]:
    """Inclusive (start_hour, end_hour) window for a given LC tag."""
    return WINDOW_BY_LC.get(lc, (WINDOW_START_HOUR, WINDOW_END_HOUR))

# --- Method registry ---
# Sign convention for closedcontourcmd:
#   searchbymax requires NEGATIVE delta (field decreases outward),
#   searchbymin requires POSITIVE delta (field increases outward).
METHOD = {
    "pv330": {
        "var": "pv_anom_330",
        "total_var": "pv330",
        "input_nc": "pv330_anom.nc",
        "units": "PVU",
        "mask_thresh": 0.15,
        "C":  {"search": "max", "thresh": (">",  0.15),
               "ccdelta": -0.01, "ccdist": 5.0},
        "AC": {"search": "min", "thresh": ("<", -0.15),
               "ccdelta": 0.01,  "ccdist": 5.0},
    },
    "zeta250": {
        "var": "zeta_anom_250",
        "total_var": "zeta_250",
        "input_nc": "zeta250_anom.nc",
        "units": "1/s",
        "mask_thresh": 8.0e-6,
        "C":  {"search": "max", "thresh": (">",  8.0e-6),
               "ccdelta": -5.0e-7, "ccdist": 5.0},
        "AC": {"search": "min", "thresh": ("<", -8.0e-6),
               "ccdelta": 5.0e-7,  "ccdist": 5.0},
    },
    "theta_pv2": {
        # Low theta on the dynamical tropopause = high-PV cyclonic intrusion.
        # Keep "C" = physical cyclone => theta' negative => searchbymin.
        "var": "theta_anom_pv2",
        "total_var": "theta_pv2",
        "input_nc": "theta_pv2_anom.nc",
        "units": "K",
        "mask_thresh": 3.0,
        "C":  {"search": "min", "thresh": ("<", -3.0),
               "ccdelta": 0.25,  "ccdist": 5.0},
        "AC": {"search": "max", "thresh": (">",  3.0),
               "ccdelta": -0.25, "ccdist": 5.0},
    },
}
METHODS = list(METHOD.keys())
# Canonical / preferred method for all downstream analysis.
# pv330 and theta_pv2 are retained in METHOD for optional use but
# zeta250 is the single authoritative track from which composites,
# projections, and animations are produced by default.
CANONICAL_METHOD = "zeta250"
THETA_LEVEL = 330.0
PRESSURE_LEVEL_ZETA = 25000.0   # Pa (250 hPa)

# --- Basis / projection ---
SYMMETRIZE = False
N_ROT = 36
SMOOTHING_DEG = 3.0
INCLUDE_LAP = False
ANCHOR_FRAME = "first"  # "first" = most-circular onset; "peak" = legacy

# --- Tilt animation ---
ANIM_FPS = 8
PCTL_CBAR = 95.0
DT_PRED_HOURS = 1.0
GUARD_PAD_DEG = 5.0     # ellipse mask must lie within PATCH_HALF - GUARD_PAD_DEG
