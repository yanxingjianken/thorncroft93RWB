"""Shared configuration for the Thorncroft RWB projection/tilt pipeline.

One authoritative place for knobs that the tracking, composite, basis
projection, and tilt-diagnostics scripts all consume.  Changing a
value here propagates to every downstream script.

Key design points:

- **Fixed-latitude Lagrangian patch.**  The patch follows the tracked
  longitude, but its absolute latitude band is held fixed at
  ``[LAT_MIN, LAT_MAX]`` = 35°..75°N, centred on ``CENTER_LAT = 55°N``.
  At 1° resolution and ``PATCH_HALF = 20``, this gives a 41×41 grid
  with ``y_rel ∈ {-20..+20}°``, ``abs_lat = CENTER_LAT + y_rel``.
  Track points whose own latitude sits near the edges stay inside the
  patch and feed their features into the composite.

- **Optional rotation symmetrization.**  ``SYMMETRIZE = False`` runs
  the basis construction directly on the raw peak-hour q' field; set
  ``True`` to re-enable the 36×10° rotation averaging introduced in
  earlier rounds.

- **Tracking thresholds.**  Both polarities use |q'| > 0.1 PVU:
  Cyclonic `q' >  POS_THRESH` and Anticyclonic `q' < -NEG_THRESH`,
  with the DetectBlobs area floor at ``AREA_MIN_KM2 = 5e5 km^2``.
"""
from __future__ import annotations

# --- Lagrangian patch geometry ---
PATCH_HALF = 20        # degrees; half-width in x and y
DX = 1.0               # degrees; grid spacing
CENTER_LAT = 55.0      # degrees N; fixed patch centre latitude
LAT_MIN = 35.0         # degrees N; patch southern edge (= CENTER_LAT - PATCH_HALF)
LAT_MAX = 75.0         # degrees N; patch northern edge (= CENTER_LAT + PATCH_HALF)

# --- Tempest thresholds ---
POS_THRESH = 0.1       # PVU; C detection threshold q' > +POS_THRESH
NEG_THRESH = 0.1       # PVU; AC detection threshold q' < -NEG_THRESH
AREA_MIN_KM2 = 5.0e5   # DetectBlobs area floor, km^2
MERGE_DIST_DEG = 8.0   # DetectNodes --mergedist
STITCH_RANGE_DEG = 7.0 # StitchNodes --range (deg between hourly steps)
STITCH_MAXGAP   = "12h"  # StitchNodes --maxgap
SPAN_REQ_H = 90        # minimum track lifespan in hours

# --- Selection ---
JUMP_MAX_DEG_PER_H = 7.0  # max hourly great-circle jump (selector)
TOP_N = 6

# --- Time window (hours since 2000-01-01 00 UTC) ---
WINDOW_START_HOUR = 6 * 24     # day 6 = hour 144
WINDOW_END_HOUR = 13 * 24      # day 13 = hour 312 (inclusive)
N_COMPOSITE_HOURS = 145        # 145 hourly frames: 0..144 over 6 days

# --- Basis / projection ---
THETA_LEVEL = 330.0    # isentropic surface for q
SYMMETRIZE = True      # if True, average 36 rotations (10° step) at peak hour
N_ROT = 36             # used when SYMMETRIZE=True (360° / N_ROT step)
SMOOTHING_DEG = 3.0    # Gaussian smoothing width passed to pvtend
INCLUDE_LAP = False    # whether pvtend basis includes the Laplacian basis

# --- Tilt animation ---
ANIM_FPS = 8
PCTL_CBAR = 95.0       # lower-row colour bars clip at this percentile of |·|
DT_PRED_HOURS = 1.0    # forward-Euler prediction horizon for tilt_animation
