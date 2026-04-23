#!/usr/bin/env bash
# TempestExtremes tracking of PV-anomaly maxima on the 330 K isentrope
# (positive q' blobs that correspond to Thorncroft RWB cyclonic
# streamer tips).
#
# Fixed threshold tracking: pv_anom_330 > 0.1 PVU.
#
# Usage: bash scripts/run_tempest_pv330.sh lc1
set -euo pipefail

LC=${1:-lc1}
ROOT=/net/flood/data2/users/x_yan/barotropic_vorticity_model/thorncroft_rwb
TE_BIN=/net/flood/home/x_yan/tempestextremes/build/bin
IN=$ROOT/outputs/$LC/pv330_anom.nc
TRK=$ROOT/outputs/$LC/tracks
mkdir -p "$TRK"

PV_THRESH_START=${PV_THRESH_START:-0.1}
PV_THRESH_MIN=${PV_THRESH_MIN:-0.1}
MIN_TRACKS=${MIN_TRACKS:-6}
SPAN_REQ_H=${SPAN_REQ_H:-90}

# Single-pass run at fixed threshold.
THRESH=$PV_THRESH_START
ATTEMPT=0
MAX_ATTEMPT=1

CAND=$TRK/cand_max_pv330.txt
TRK_OUT=$TRK/tracks_max_pv330.txt
ACCEPTED=""

while (( ATTEMPT < MAX_ATTEMPT )); do
  ATTEMPT=$((ATTEMPT + 1))
  echo "[tempest-pv330] $LC attempt=$ATTEMPT thresh=$THRESH PVU"

  $TE_BIN/DetectNodes \
    --in_data "$IN" \
    --out "$CAND" \
    --searchbymax pv_anom_330 \
    --thresholdcmd "pv_anom_330,>,${THRESH},0" \
    --mergedist 8.0 \
    --minlat 20.0 --maxlat 80.0 \
    --outputcmd "pv_anom_330,max,0" > "$TRK/detect_pv330.log" 2>&1

  $TE_BIN/StitchNodes \
    --in "$CAND" --out "$TRK_OUT" \
    --in_fmt "lon,lat,pv_anom" \
    --range "${STITCH_RANGE:-5.0}" --mintime "${SPAN_REQ_H}h" --maxgap "${STITCH_MAXGAP:-6h}" \
    > "$TRK/stitch_pv330.log" 2>&1

  # Count tracks in the StitchNodes output.
  if [[ -s "$TRK_OUT" ]]; then
    N=$(grep -c "^start" "$TRK_OUT" || true)
  else
    N=0
  fi
  echo "[tempest-pv330] $LC attempt=$ATTEMPT => $N tracks with span>=${SPAN_REQ_H}h"

  if (( N >= MIN_TRACKS )); then
    ACCEPTED="thresh=${THRESH} tracks=${N}"
    break
  fi

  break
done

# Also run a single DetectBlobs pass at the final threshold for
# visualisation of the PV-anom blobs on the tracked GIF.
BLOB_POS=$TRK/blobs_pv330_pos.nc
$TE_BIN/DetectBlobs \
  --in_data "$IN" \
  --out "$BLOB_POS" \
  --thresholdcmd "pv_anom_330,>,${THRESH},0" \
  --minlat 20.0 --maxlat 80.0 \
  --geofiltercmd "area,>=,1.2e5km2" > "$TRK/blobs_pv330.log" 2>&1

echo "[tempest-pv330] $LC accepted=${ACCEPTED:-NONE} final_thresh=${THRESH}"
# Record the final threshold for downstream scripts.
echo "$THRESH" > "$TRK/pv330_thresh.txt"
ls -la "$TRK"
