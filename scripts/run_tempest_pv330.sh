#!/usr/bin/env bash
# TempestExtremes tracking of PV-anomaly extrema on the 330 K isentrope.
#
# Tracks BOTH polarities:
#   - Cyclonic    (C):   pv_anom_330 >  +0.1 PVU  (cyclonic tip of a C streamer)
#   - Anticyclonic (AC): pv_anom_330 <  -0.1 PVU  (anticyclonic tongue/cut-off)
#
# Latitude gate 35..75 N, DetectBlobs area floor 5e5 km^2.
#
# Usage: bash scripts/run_tempest_pv330.sh lc1
set -euo pipefail

LC=${1:-lc1}
ROOT=/net/flood/data2/users/x_yan/barotropic_vorticity_model/thorncroft_rwb
TE_BIN=/net/flood/home/x_yan/tempestextremes/build/bin
IN=$ROOT/outputs/$LC/pv330_anom.nc
TRK=$ROOT/outputs/$LC/tracks
mkdir -p "$TRK"

POS_THRESH=${POS_THRESH:-0.1}   # PVU, C:  q' >  +POS_THRESH
NEG_THRESH=${NEG_THRESH:-0.1}   # PVU, AC: q' <  -NEG_THRESH  (absolute value)
MIN_LAT=${MIN_LAT:-35.0}
MAX_LAT=${MAX_LAT:-75.0}
MERGE_DIST=${MERGE_DIST:-8.0}
STITCH_RANGE=${STITCH_RANGE:-7.0}
STITCH_MAXGAP=${STITCH_MAXGAP:-6h}
SPAN_REQ_H=${SPAN_REQ_H:-90}
AREA_MIN=${AREA_MIN:-5e5km2}

# -----------------------------------------------------------------------
# Cyclonic (positive anomaly) branch
# -----------------------------------------------------------------------
CAND_POS=$TRK/cand_max_pv330.txt
TRK_POS=$TRK/tracks_max_pv330.txt
BLOB_POS=$TRK/blobs_pv330_pos.nc

echo "[tempest-pv330] $LC C:  q' > +${POS_THRESH} PVU, lat ${MIN_LAT}..${MAX_LAT}"

$TE_BIN/DetectNodes \
  --in_data "$IN" \
  --out "$CAND_POS" \
  --searchbymax pv_anom_330 \
  --thresholdcmd "pv_anom_330,>,${POS_THRESH},0" \
  --mergedist "${MERGE_DIST}" \
  --minlat "${MIN_LAT}" --maxlat "${MAX_LAT}" \
  --outputcmd "pv_anom_330,max,0" > "$TRK/detect_pv330_pos.log" 2>&1

$TE_BIN/StitchNodes \
  --in "$CAND_POS" --out "$TRK_POS" \
  --in_fmt "lon,lat,pv_anom" \
  --range "${STITCH_RANGE}" --mintime "${SPAN_REQ_H}h" --maxgap "${STITCH_MAXGAP}" \
  > "$TRK/stitch_pv330_pos.log" 2>&1

$TE_BIN/DetectBlobs \
  --in_data "$IN" \
  --out "$BLOB_POS" \
  --thresholdcmd "pv_anom_330,>,${POS_THRESH},0" \
  --minlat "${MIN_LAT}" --maxlat "${MAX_LAT}" \
  --geofiltercmd "area,>=,${AREA_MIN}" > "$TRK/blobs_pv330_pos.log" 2>&1

N_POS=$( (grep -c '^start' "$TRK_POS" 2>/dev/null) || echo 0)
echo "[tempest-pv330] $LC C  => ${N_POS} tracks"

# -----------------------------------------------------------------------
# Anticyclonic (negative anomaly) branch
# -----------------------------------------------------------------------
CAND_NEG=$TRK/cand_min_pv330.txt
TRK_NEG=$TRK/tracks_min_pv330.txt
BLOB_NEG=$TRK/blobs_pv330_neg.nc

echo "[tempest-pv330] $LC AC: q' < -${NEG_THRESH} PVU, lat ${MIN_LAT}..${MAX_LAT}"

$TE_BIN/DetectNodes \
  --in_data "$IN" \
  --out "$CAND_NEG" \
  --searchbymin pv_anom_330 \
  --thresholdcmd "pv_anom_330,<,-${NEG_THRESH},0" \
  --mergedist "${MERGE_DIST}" \
  --minlat "${MIN_LAT}" --maxlat "${MAX_LAT}" \
  --outputcmd "pv_anom_330,min,0" > "$TRK/detect_pv330_neg.log" 2>&1

$TE_BIN/StitchNodes \
  --in "$CAND_NEG" --out "$TRK_NEG" \
  --in_fmt "lon,lat,pv_anom" \
  --range "${STITCH_RANGE}" --mintime "${SPAN_REQ_H}h" --maxgap "${STITCH_MAXGAP}" \
  > "$TRK/stitch_pv330_neg.log" 2>&1

$TE_BIN/DetectBlobs \
  --in_data "$IN" \
  --out "$BLOB_NEG" \
  --thresholdcmd "pv_anom_330,<,-${NEG_THRESH},0" \
  --minlat "${MIN_LAT}" --maxlat "${MAX_LAT}" \
  --geofiltercmd "area,>=,${AREA_MIN}" > "$TRK/blobs_pv330_neg.log" 2>&1

N_NEG=$( (grep -c '^start' "$TRK_NEG" 2>/dev/null) || echo 0)
echo "[tempest-pv330] $LC AC => ${N_NEG} tracks"

# Record thresholds for downstream scripts.
printf "C=%s AC=%s\n" "${POS_THRESH}" "${NEG_THRESH}" > "$TRK/pv330_thresh.txt"
ls -la "$TRK"
