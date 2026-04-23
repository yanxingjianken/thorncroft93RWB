#!/usr/bin/env bash
# TempestExtremes tracking of 250 hPa relative vorticity (cyclone only).
# Tracks 250 hPa positive ζ maxima (cyclones, threshold vor250 > 2e-5)
# over 20-80°N during days 6-13 (hourly).  StitchNodes relaxed to
# --mintime 90h so tracks do NOT have to span the full window; jumpy
# segments are filtered out downstream in select_top6.py.
#
# Usage: bash scripts/run_tempest_vor250.sh lc1
set -euo pipefail

LC=${1:-lc1}
ROOT=/net/flood/data2/users/x_yan/barotropic_vorticity_model/thorncroft_rwb
TE_BIN=/net/flood/home/x_yan/tempestextremes/build/bin
IN=$ROOT/outputs/$LC/vor250.nc
TRK=$ROOT/outputs/$LC/tracks
mkdir -p "$TRK"

EPS_NODE=${EPS_NODE:-2.0e-5}
EPS_BLOB=${EPS_BLOB:-2.0e-5}

CAND_MAX=$TRK/cand_max_vor250.txt

$TE_BIN/DetectNodes \
  --in_data "$IN" \
  --out "$CAND_MAX" \
  --searchbymax vor250 \
  --thresholdcmd "vor250,>,${EPS_NODE},0" \
  --mergedist 20.0 \
  --minlat 20.0 --maxlat 80.0 \
  --outputcmd "vor250,max,0"

TRK_MAX=$TRK/tracks_max_vor250.txt

# Tight range (<= 3 deg / hour) so physical cyclones stay intact and
# jumpy spurious stitches are suppressed.  mintime relaxed to 90h.
$TE_BIN/StitchNodes \
  --in "$CAND_MAX" --out "$TRK_MAX" \
  --in_fmt "lon,lat,vor" \
  --range 3.0 --mintime "90h" --maxgap "6h"

BLOB_POS=$TRK/blobs_vor250_pos.nc
$TE_BIN/DetectBlobs \
  --in_data "$IN" \
  --out "$BLOB_POS" \
  --thresholdcmd "vor250,>,${EPS_BLOB},0" \
  --minlat 20.0 --maxlat 80.0 \
  --geofiltercmd "area,>=,1.2e5km2"

echo "[tempest-vor250] $LC tracks/blobs written to $TRK/"
ls -la "$TRK"
