"""5-basis orthogonal decomposition anchored at the FIRST tracked frame.

For each (lc, method, polarity) the basis is built from the q (total) and
q' (anomaly) at the *earliest* valid composite hour (most circular onset),
not from the peak-amplitude frame. ``CFG.SYMMETRIZE`` is False in v2.3
so the raw onset field is used directly (matches what the tilt-animation
ellipse fit operates on).

Reads:  outputs/<lc>/composites/<method>/{C,AC}_composite.nc
Writes: outputs/<lc>/projections/<method>/plots/{decomp,decomp_bases}_{C,AC}.png
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pvtend

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa
from _grad_safe import safe_gradient, mask_vmax  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")
M_PER_DEG = 111_000.0


def _first_valid_frame(qa: np.ndarray, frac: float = 0.9) -> int:
    """First hour where ≥ ``frac`` of the patch is finite."""
    fr = np.isfinite(qa).reshape(qa.shape[0], -1).mean(axis=1)
    where = np.where(fr >= frac)[0]
    return int(where[0]) if where.size else 0


def _fill_nans(F: np.ndarray) -> np.ndarray:
    """Replace NaN with 0 for stable basis/projection arithmetic."""
    out = F.copy()
    out[~np.isfinite(out)] = 0.0
    return out


def process(lc: str, method: str, polarity: str = "C"):
    src = ROOT / lc / "composites" / method / f"{polarity}_composite.nc"
    if not src.exists():
        print(f"[{lc}:{method}:{polarity}] missing {src}; skipping")
        return
    ds = xr.open_dataset(src)
    q = ds["total_composite"].values.astype("float64")
    qa = ds["anom_composite"].values.astype("float64")
    t_hour = ds["t"].values
    x_rel = ds["x"].values.astype("float64")
    y_rel = ds["y"].values.astype("float64")

    # pv_dt (centred difference)
    pv_dt = np.full_like(qa, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2 * 3600.0)

    # Anchor frame: earliest valid hour (most circular onset)
    i0 = _first_valid_frame(qa)
    # Need finite pv_dt around i0; bump until pv_dt[i0] is mostly finite.
    while i0 < qa.shape[0] - 1 and (
        np.isfinite(pv_dt[i0]).mean() < 0.5
    ):
        i0 += 1
    if not np.isfinite(pv_dt[i0]).any():
        print(f"[{lc}:{method}:{polarity}] no valid pv_dt frame; skipping")
        return

    q_anchor = _fill_nans(q[i0])
    qa_anchor = _fill_nans(qa[i0])
    pv_dt_anchor = _fill_nans(pv_dt[i0])
    mask_str = "> 0" if polarity == "C" else "< 0"

    dx_m = CFG.DX * M_PER_DEG * np.cos(np.deg2rad(CFG.CENTER_LAT))
    dy_m = CFG.DX * M_PER_DEG
    qdx_anchor, qdy_anchor = safe_gradient(q[i0], dy_m, dx_m)

    basis = pvtend.compute_orthogonal_basis(
        qa_anchor, qdx_anchor, qdy_anchor, x_rel, y_rel,
        mask=mask_str,
        apply_smoothing=True, smoothing_deg=CFG.SMOOTHING_DEG,
        grid_spacing=CFG.DX, center_lat=CFG.CENTER_LAT,
        include_lap=CFG.INCLUDE_LAP,
    )

    proj = pvtend.project_field(pv_dt_anchor, basis,
                                grid_spacing=CFG.DX)
    total = pv_dt_anchor
    recon = proj["recon"]; resid = proj["resid"]
    integ = proj["int"]; prop = proj["prop"]; deform = proj["def"]

    # Build the central-component mask first; cbar limits are computed
    # inside the mask region only so basis edge artefacts (np.gradient
    # one-sided stencils on the patch rim) do not dominate the colour
    # range.
    from scipy import ndimage as ndi
    fit_thr = float(CFG.METHOD[method]["mask_thresh"])
    guard_r = float(CFG.PATCH_HALF - CFG.GUARD_PAD_DEG)
    X, Y = np.meshgrid(x_rel, y_rel)
    if polarity == "C":
        raw_m = qa_anchor > fit_thr
    else:
        raw_m = qa_anchor < -fit_thr
    raw_m &= (np.sqrt(X * X + Y * Y) <= guard_r)
    if raw_m.any():
        labs, nlab = ndi.label(raw_m)
        if nlab > 1:
            iy0 = int(np.argmin(np.abs(y_rel)))
            ix0 = int(np.argmin(np.abs(x_rel)))
            centre_lbl = labs[iy0, ix0]
            if centre_lbl == 0:
                cs = ndi.center_of_mass(raw_m, labs, range(1, nlab + 1))
                ds_ = [np.hypot(cy - iy0, cx - ix0) for cy, cx in cs]
                centre_lbl = int(np.argmin(ds_)) + 1
            raw_m = labs == centre_lbl

    r1 = max(mask_vmax(total, raw_m), mask_vmax(recon, raw_m),
             mask_vmax(resid, raw_m))
    r2 = max(mask_vmax(integ, raw_m), mask_vmax(prop, raw_m),
             mask_vmax(deform, raw_m))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8),
                             constrained_layout=True)

    def _panel(ax, F, vmax, title):
        im = ax.pcolormesh(X, Y, F, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [deg]"); ax.set_ylabel("y [deg]")
        ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.axvline(0, color="k", lw=0.4, alpha=0.5)
        return im

    im1 = _panel(axes[0, 0], total, r1, r"$\partial q/\partial t$")
    # overlay the central-component ellipse mask.
    axes[0, 0].contour(X, Y, raw_m.astype(float), levels=[0.5],
                       colors="k", linewidths=1.2, linestyles="--")
    _panel(axes[0, 1], recon, r1, "reconstruction")
    _panel(axes[0, 2], resid, r1, "residual")
    fig.colorbar(im1, ax=axes[0, :], shrink=0.8, label="tendency")
    im2 = _panel(axes[1, 0], integ, r2, r"int:  $\beta\,\phi_1$")
    _panel(axes[1, 1], prop, r2, r"prop: $-a_x\phi_2 - a_y\phi_3$")
    _panel(axes[1, 2], deform, r2,
           r"def:  $-\gamma_1\phi_4 - \gamma_2\phi_5$")
    fig.colorbar(im2, ax=axes[1, :], shrink=0.8, label="component")

    coef = (f"β={proj['beta']:.2e}  ax={proj['ax']:.2e}  "
            f"ay={proj['ay']:.2e}  γ1={proj['gamma1']:.2e}  "
            f"γ2={proj['gamma2']:.2e}  RMSE={proj.get('rmse', np.nan):.2e}")
    fig.suptitle(
        f"{lc.upper()}  {method}/{polarity}  Lagrangian composite  "
        f"anchor t={int(t_hour[i0])}h (first valid, most circular)\n{coef}",
        fontsize=11)

    plots = ROOT / lc / "projections" / method / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    out = plots / f"decomp_{polarity}.png"
    fig.savefig(out, dpi=140); plt.close(fig)

    # bases panel
    phis = [basis.phi_int, basis.phi_dx, basis.phi_dy,
            basis.phi_def, basis.phi_strain]
    labels = [r"$\phi_1$ int", r"$\phi_2$ prop$_x$", r"$\phi_3$ prop$_y$",
              r"$\phi_4$ def shear", r"$\phi_5$ def strain"]
    rb = max(mask_vmax(p, raw_m) for p in phis)
    fig, axs = plt.subplots(1, 5, figsize=(18, 4.2),
                            constrained_layout=True)
    last = None
    for ax, F, lab in zip(axs, phis, labels):
        last = ax.pcolormesh(X, Y, F, cmap="RdBu_r",
                             vmin=-rb, vmax=rb, shading="auto")
        ax.set_aspect("equal"); ax.set_title(lab, fontsize=11)
        ax.set_xlabel("x [deg]")
    axs[0].set_ylabel("y [deg]")
    fig.colorbar(last, ax=axs, shrink=0.85, label="basis amplitude")
    fig.suptitle(
        f"{lc.upper()}  {method}/{polarity} basis at anchor t*"
        f"={int(t_hour[i0])}h  (mask q' {mask_str})", fontsize=12)
    out_b = plots / f"decomp_bases_{polarity}.png"
    fig.savefig(out_b, dpi=140); plt.close(fig)
    print(f"[{lc}:{method}:{polarity}] wrote {out.name} (anchor t={int(t_hour[i0])}h)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--method", default=None)
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    methods = [args.method] if args.method else [CFG.CANONICAL_METHOD]
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for m in methods:
            for pol in pols:
                process(lc, m, polarity=pol)
