#!/usr/bin/env bash
# TempestExtremes tracking of θ-on-2-PVU anomaly extrema (RWB
# signatures) for the Thorncroft LC runs.
#
#   - DetectNodes + StitchNodes on θ' minima   → cyclonic-RWB tracks
#   - DetectNodes + StitchNodes on θ' maxima   → anticyclonic-RWB tracks
#   - DetectBlobs + StitchBlobs  on |θ'|>k K   → RWB wave bodies
#
# Usage:
#   bash scripts/run_tempest_pv2.sh lc1
#   bash scripts/run_tempest_pv2.sh lc2
#
set -euo pipefail

LC=${1:-lc1}
ROOT=/net/flood/data2/users/x_yan/barotropic_vorticity_model/thorncroft_rwb
TE_BIN=/net/flood/home/x_yan/tempestextremes/build/bin
IN=$ROOT/outputs/$LC/theta_pv2_anom.nc
TRK=$ROOT/outputs/$LC/tracks
mkdir -p "$TRK"

# ------------------------------------------------------------------
# DetectNodes — cyclonic (θ' minima) and anticyclonic (θ' maxima).
# ------------------------------------------------------------------
# closed-contour criteria:
#   cyclonic   : θ' must rise by >=4 K within 6° of the centre
#   anticyclonic: θ' must fall by <=-4 K within 6° of the centre
CAND_MIN=$TRK/cand_min.txt
CAND_MAX=$TRK/cand_max.txt
$TE_BIN/DetectNodes \
  --in_data "$IN" \
  --out "$CAND_MIN" \
  --searchbymin theta_pv2_anom \
  --closedcontourcmd "theta_pv2_anom,4.0,6.0,0" \
  --mergedist 6.0 \
  --minlat 20.0 --maxlat 85.0 \
  --outputcmd "theta_pv2_anom,min,0"

$TE_BIN/DetectNodes \
  --in_data "$IN" \
  --out "$CAND_MAX" \
  --searchbymax theta_pv2_anom \
  --closedcontourcmd "theta_pv2_anom,-4.0,6.0,0" \
  --mergedist 6.0 \
  --minlat 20.0 --maxlat 85.0 \
  --outputcmd "theta_pv2_anom,max,0"

# ------------------------------------------------------------------
# StitchNodes — form time-continuous tracks.
# ------------------------------------------------------------------
TRK_MIN=$TRK/tracks_min.txt
TRK_MAX=$TRK/tracks_max.txt
# Hourly data, wave-6 perturbation => expect ~6 stitched tracks per
# polarity. Loose maxgap/range; we post-filter to top-6 by length in
# the plotting stage.
$TE_BIN/StitchNodes \
  --in "$CAND_MIN" --out "$TRK_MIN" \
  --in_fmt "lon,lat,theta_min" \
  --range 8.0 --mintime "48h" --maxgap "3h"

$TE_BIN/StitchNodes \
  --in "$CAND_MAX" --out "$TRK_MAX" \
  --in_fmt "lon,lat,theta_max" \
  --range 8.0 --mintime "48h" --maxgap "3h"

# ------------------------------------------------------------------
# DetectBlobs — closed regions with |θ'| > 8 K.
# ------------------------------------------------------------------
BLOB_NEG=$TRK/blobs_neg.nc
BLOB_POS=$TRK/blobs_pos.nc
$TE_BIN/DetectBlobs \
  --in_data "$IN" \
  --out "$BLOB_NEG" \
  --thresholdcmd "theta_pv2_anom,<=,-8.0,0" \
  --minlat 20.0 --maxlat 85.0 \
  --geofiltercmd "area,>=,3.0e5km2"

$TE_BIN/DetectBlobs \
  --in_data "$IN" \
  --out "$BLOB_POS" \
  --thresholdcmd "theta_pv2_anom,>=,8.0,0" \
  --minlat 20.0 --maxlat 85.0 \
  --geofiltercmd "area,>=,3.0e5km2"

echo "[tempest] $LC tracks/blobs written to $TRK/"
ls -la "$TRK"
