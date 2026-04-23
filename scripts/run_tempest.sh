#!/usr/bin/env bash
# Three-way TempestExtremes tracking for the Thorncroft RWB pipeline.
#
# Usage:  bash scripts/run_tempest.sh <lc> <method>
#   <lc>     : lc1 | lc2
#   <method> : pv330 | zeta250 | theta_pv2
#
# Outputs go under outputs/<lc>/tracks/<method>/.
# Localised closed-contour DetectNodes + StitchNodes for both polarities.
# No fallback — a method that needs fallback is not a contender.
set -euo pipefail

LC=${1:?lc1 or lc2}
METHOD=${2:?pv330|zeta250|theta_pv2}
ROOT=/net/flood/data2/users/x_yan/barotropic_vorticity_model/thorncroft_rwb
TE_BIN=/net/flood/home/x_yan/tempestextremes/build/bin

# Pull per-method config from _config.py (single source of truth).
read VAR INPUT_NC \
     C_SEARCH C_OP C_THRESH C_DELTA C_DIST \
     AC_SEARCH AC_OP AC_THRESH AC_DELTA AC_DIST \
     MIN_LAT MAX_LAT MERGE STITCH_RANGE STITCH_MAXGAP SPAN_REQ_H \
  < <(micromamba run -n speedy_weather python - <<PY
import sys
sys.path.insert(0, "${ROOT}/scripts")
from _config import (METHOD, LAT_MIN, LAT_MAX, MERGE_DIST_DEG,
                     STITCH_RANGE_DEG, STITCH_MAXGAP, SPAN_REQ_H)
m = METHOD["${METHOD}"]
print(m["var"], m["input_nc"],
      m["C"]["search"],  m["C"]["thresh"][0],  m["C"]["thresh"][1],
      m["C"]["ccdelta"], m["C"]["ccdist"],
      m["AC"]["search"], m["AC"]["thresh"][0], m["AC"]["thresh"][1],
      m["AC"]["ccdelta"], m["AC"]["ccdist"],
      LAT_MIN, LAT_MAX, MERGE_DIST_DEG,
      STITCH_RANGE_DEG, STITCH_MAXGAP, SPAN_REQ_H)
PY
)

IN=$ROOT/outputs/$LC/$INPUT_NC
TRK=$ROOT/outputs/$LC/tracks/$METHOD
mkdir -p "$TRK"

run_branch () {
  local POL=$1 SEARCH=$2 OP=$3 THRESH=$4 DELTA=$5 DIST=$6
  local CAND=$TRK/cand_${POL}.txt
  local TRACKS=$TRK/tracks_${POL}.txt
  echo "[tempest] $LC $METHOD $POL  searchby${SEARCH} ${VAR} ${OP} ${THRESH}  cc=${DELTA}/${DIST}°"
  $TE_BIN/DetectNodes \
    --in_data "$IN" \
    --out "$CAND" \
    --searchby${SEARCH} "$VAR" \
    --thresholdcmd "${VAR},${OP},${THRESH},0" \
    --closedcontourcmd "${VAR},${DELTA},${DIST},0" \
    --mergedist "${MERGE}" \
    --minlat "${MIN_LAT}" --maxlat "${MAX_LAT}" \
    --outputcmd "${VAR},${SEARCH},0" > "$TRK/detect_${POL}.log" 2>&1

  $TE_BIN/StitchNodes \
    --in "$CAND" --out "$TRACKS" \
    --in_fmt "lon,lat,${VAR}" \
    --range "${STITCH_RANGE}" \
    --mintime "${SPAN_REQ_H}h" \
    --maxgap  "${STITCH_MAXGAP}" \
    > "$TRK/stitch_${POL}.log" 2>&1

  local N
  N=$( (grep -c '^start' "$TRACKS") 2>/dev/null || echo 0)
  N=${N//[^0-9]/}; N=${N:-0}
  echo "[tempest] $LC $METHOD $POL => ${N} tracks"
}

run_branch C  "$C_SEARCH"  "$C_OP"  "$C_THRESH"  "$C_DELTA"  "$C_DIST"
run_branch AC "$AC_SEARCH" "$AC_OP" "$AC_THRESH" "$AC_DELTA" "$AC_DIST"

ls -la "$TRK"
