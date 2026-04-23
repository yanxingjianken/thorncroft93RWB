"""Filter a TempestExtremes candidate file to keep only nodes inside a
qualifying DetectBlobs mask (i.e. an area>=AREA_MIN_KM2 coherent region).

Usage:
    python filter_cand_by_blob.py <cand_in.txt> <blob.nc> <cand_out.txt>

The candidate file format is the TempestExtremes DetectNodes text output:

    <year> <mon> <day> <hour> <N>
        <i> <j> <lon> <lat> <value>
        ...

We read the blob NetCDF (variable name "binary_tag" or "object_id";
whatever 3-D mask variable exists) and for each node keep it only if
the blob mask at the nearest (time, lat, lon) is > 0.5.

Safe to call in the tracking shell script between DetectNodes and
StitchNodes.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr


def _load_mask(nc_path: Path) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)
    for v in ds.data_vars:
        arr = ds[v]
        if arr.ndim == 3 and "time" in arr.dims:
            return arr.astype("float32")
    raise RuntimeError(f"No 3-D mask variable found in {nc_path}")


def _parse_time_header(line: str) -> datetime:
    toks = line.split()
    # DetectNodes header: "YYYY MM DD HH N"
    return datetime(int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3]))


def filter_file(cand_in: Path, blob_nc: Path, cand_out: Path) -> tuple[int, int]:
    mask = _load_mask(blob_nc)
    times = mask["time"].values
    lats = mask["lat"].values
    lons = mask["lon"].values
    lons_wrap = lons % 360.0

    with open(cand_in) as fh:
        lines = fh.readlines()

    out_lines: list[str] = []
    n_in = 0
    n_out = 0
    i = 0
    block_idx = 0  # sequential header index == time index
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        toks = line.split()
        if len(toks) >= 5 and toks[0].isdigit() and len(toks[0]) == 4:
            # header: YYYY MM DD COUNT HOUR  (hour may be >=24, not a wall
            # clock value — we don't parse as datetime; use block index
            # to map onto the blob netcdf time axis)
            n_nodes = int(toks[3])
            if block_idx < len(times):
                ti = block_idx
            else:
                ti = len(times) - 1
            block_idx += 1
            node_lines = lines[i + 1:i + 1 + n_nodes]
            i += 1 + n_nodes
            kept: list[str] = []
            for nl in node_lines:
                ntoks = nl.split()
                if len(ntoks) < 5:
                    continue
                n_in += 1
                lon = float(ntoks[2]); lat = float(ntoks[3])
                lon_wrap = lon % 360.0
                jx = int(np.argmin(np.abs(lons_wrap - lon_wrap)))
                iy = int(np.argmin(np.abs(lats - lat)))
                v = float(mask.values[ti, iy, jx])
                if v > 0.5:
                    kept.append(nl)
            if kept:
                new_head = ("%s\t%s\t%s\t%d\t%s\n"
                            % (toks[0], toks[1], toks[2], len(kept), toks[4]))
                out_lines.append(new_head)
                out_lines.extend(kept)
                n_out += len(kept)
            else:
                # always emit header so StitchNodes sees the full time axis
                new_head = ("%s\t%s\t%s\t0\t%s\n"
                            % (toks[0], toks[1], toks[2], toks[4]))
                out_lines.append(new_head)
        else:
            i += 1

    cand_out.write_text("".join(out_lines))
    return n_in, n_out


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    n_in, n_out = filter_file(Path(sys.argv[1]),
                              Path(sys.argv[2]),
                              Path(sys.argv[3]))
    print(f"[filter_cand_by_blob] kept {n_out}/{n_in} nodes  "
          f"-> {sys.argv[3]}")


if __name__ == "__main__":
    main()
