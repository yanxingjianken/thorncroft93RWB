"""Filter a TempestExtremes candidate file to keep only nodes inside a
qualifying DetectBlobs mask (area>=AREA_MIN_KM2 coherent region) AND
replace the reported (lon, lat, value) with the ``|pv_anom|``-weighted
mass centroid of the connected blob component that contains the node.

Usage:
    python filter_cand_by_blob.py <cand_in.txt> <blob.nc> <cand_out.txt> <pv_anom.nc>

The candidate file format is the TempestExtremes DetectNodes text output::

    <year> <mon> <day> <count> <hour>
        <ix> <iy> <lon> <lat> <value>
        ...

The fourth argument is the signed pv_anom NetCDF used as weighting for
the centroid (weight = |pv_anom| * cos(lat)).  Longitude is averaged on
the unit circle to handle the 0/360 wrap.  For each kept node we:

  1. Label connected components of the DetectBlobs mask at the matching
     timestep (``scipy.ndimage.label``).
  2. Find the component containing the node.
  3. Compute the area-weighted |pv_anom| centroid over that component.
  4. Emit a new line ``ix iy lon_c lat_c value_c`` where ``value_c`` is
     the max of |pv_anom| within the component.

An empty-count header is emitted for timesteps with no kept nodes, so
StitchNodes sees the full time axis and does not segfault.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import label as nd_label


def _load_mask(nc_path: Path) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)
    for v in ds.data_vars:
        arr = ds[v]
        if arr.ndim == 3 and "time" in arr.dims:
            return arr.astype("float32")
    raise RuntimeError(f"No 3-D mask variable found in {nc_path}")


def _load_pv_anom(nc_path: Path) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)
    if "pv_anom_330" in ds.data_vars:
        return ds["pv_anom_330"]
    for v in ds.data_vars:
        arr = ds[v]
        if arr.ndim == 3 and "time" in arr.dims:
            return arr
    raise RuntimeError(f"No 3-D pv_anom variable found in {nc_path}")


def _weighted_centroid(
    lbl_arr: np.ndarray,
    label_val: int,
    pv2d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> tuple[float, float, float]:
    sel = lbl_arr == label_val
    pv_sel = np.abs(pv2d[sel])
    lat_sel = lat2d[sel]
    lon_sel = lon2d[sel]
    w = pv_sel * np.cos(np.deg2rad(lat_sel))
    w_sum = float(w.sum())
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        return float("nan"), float("nan"), float("nan")
    lon_rad = np.deg2rad(lon_sel)
    cx = float(np.sum(w * np.cos(lon_rad)))
    cy = float(np.sum(w * np.sin(lon_rad)))
    lon_c = float(np.rad2deg(np.arctan2(cy, cx))) % 360.0
    lat_c = float(np.sum(w * lat_sel) / w_sum)
    value_max = float(pv_sel.max())
    return lon_c, lat_c, value_max


def filter_file(
    cand_in: Path,
    blob_nc: Path,
    cand_out: Path,
    pv_nc: Path,
) -> tuple[int, int]:
    mask = _load_mask(blob_nc)
    pv = _load_pv_anom(pv_nc)

    lats = mask["lat"].values
    lons = mask["lon"].values
    lons_wrap = lons % 360.0
    n_t = mask.sizes["time"]

    lon2d, lat2d = np.meshgrid(lons_wrap, lats)

    with open(cand_in) as fh:
        lines = fh.readlines()

    out_lines: list[str] = []
    n_in = 0
    n_out = 0
    i = 0
    block_idx = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        toks = line.split()
        if len(toks) >= 5 and toks[0].isdigit() and len(toks[0]) == 4:
            n_nodes = int(toks[3])
            ti = block_idx if block_idx < n_t else n_t - 1
            block_idx += 1
            node_lines = lines[i + 1:i + 1 + n_nodes]
            i += 1 + n_nodes

            mask2 = mask.values[ti] > 0.5
            pv2 = pv.values[ti]
            lbl_arr, _ = nd_label(mask2)

            kept: list[str] = []
            seen_labels: set[int] = set()
            for nl in node_lines:
                ntoks = nl.split()
                if len(ntoks) < 5:
                    continue
                n_in += 1
                ix = ntoks[0]
                iy = ntoks[1]
                lon = float(ntoks[2])
                lat = float(ntoks[3])
                jx = int(np.argmin(np.abs(lons_wrap - (lon % 360.0))))
                iy_g = int(np.argmin(np.abs(lats - lat)))
                lv = int(lbl_arr[iy_g, jx])
                if lv <= 0:
                    continue
                # emit one centroid per connected blob per timestep
                if lv in seen_labels:
                    continue
                seen_labels.add(lv)
                lon_c, lat_c, val_c = _weighted_centroid(
                    lbl_arr, lv, pv2, lat2d, lon2d)
                if not (np.isfinite(lon_c) and np.isfinite(lat_c)):
                    continue
                kept.append(
                    f"\t{ix}\t{iy}\t{lon_c:.6f}\t{lat_c:.6f}\t{val_c:.6e}\n"
                )

            if kept:
                new_head = (f"{toks[0]}\t{toks[1]}\t{toks[2]}"
                            f"\t{len(kept)}\t{toks[4]}\n")
                out_lines.append(new_head)
                out_lines.extend(kept)
                n_out += len(kept)
            else:
                new_head = (f"{toks[0]}\t{toks[1]}\t{toks[2]}"
                            f"\t0\t{toks[4]}\n")
                out_lines.append(new_head)
        else:
            i += 1

    cand_out.write_text("".join(out_lines))
    return n_in, n_out


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    n_in, n_out = filter_file(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        Path(sys.argv[4]),
    )
    print(f"[filter_cand_by_blob] kept {n_out}/{n_in} nodes  "
          f"(centroid-replaced) -> {sys.argv[3]}")


if __name__ == "__main__":
    main()
