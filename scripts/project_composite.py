"""5-basis orthogonal decomposition of the cyclone/anticyclone composite.

For each polarity (C or AC), the peak hour t* is the frame that
maximises the appropriate mass integral:

    C:  t* = argmax_t   sum( max(+q'(t), 0) )   (strongest positive lobe)
    AC: t* = argmax_t   sum( max(-q'(t), 0) )   (strongest negative lobe)

If ``SYMMETRIZE`` is True (``_config.SYMMETRIZE``) both the full PV
and the anomaly at t* are symmetrized via ``N_ROT = 36`` rotation
averages before the basis is computed; otherwise the raw fields at
t* are used directly.

The pvtend orthogonal basis is computed with
    mask = "> 0"    for C
    mask = "< 0"    for AC
and ``center_lat = 55 N``.  The un-symmetrized pv_dt at t* is then
projected to produce the 5-component decomposition.

Outputs (per LC, per polarity):
  projections/plots/decomp_{C,AC}.png        2x3 panel
  projections/plots/decomp_bases_{C,AC}.png  1x5 basis panel
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.ndimage import rotate as nd_rotate
import matplotlib.pyplot as plt
import pvtend

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")
M_PER_DEG = 111_000.0


def rot_avg(field: np.ndarray, n_rot: int = CFG.N_ROT) -> np.ndarray:
    """Average over ``n_rot`` equally-spaced rotations of ``field``."""
    out = np.zeros_like(field, dtype="float64")
    for k in range(n_rot):
        ang = k * (360.0 / n_rot)
        out += nd_rotate(field, ang, reshape=False, order=1,
                         mode="nearest")
    return out / n_rot


def process(lc: str, polarity: str = "C"):
    ds = xr.open_dataset(ROOT / lc / "composites" /
                         f"{polarity}_composite.nc")
    q = ds["pv_composite"].values.astype("float64")
    t_hour = ds["t"].values
    x_rel = ds["x"].values.astype("float64")
    y_rel = ds["y"].values.astype("float64")

    qa = q - np.nanmean(q, axis=-1, keepdims=True)

    # pv_dt via centered difference on raw composite
    pv_dt = np.full_like(qa, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2 * CFG.DT_PRED_HOURS * 3600.0)
    # note DT_PRED_HOURS=1 → 2*3600 s stencil for centered difference

    # Peak hour: polarity-specific mass integral
    if polarity == "C":
        mass = np.nansum(np.where(qa > 0, qa, 0.0), axis=(1, 2))
        mask_str = "> 0"
    else:
        mass = np.nansum(np.where(qa < 0, -qa, 0.0), axis=(1, 2))
        mask_str = "< 0"
    mass[~np.isfinite(mass)] = -np.inf
    mass[0] = mass[-1] = -np.inf
    valid_dt = np.isfinite(pv_dt).all(axis=(1, 2))
    if not valid_dt.any():
        print(f"[{lc}:{polarity}] no fully finite pv_dt frame; skipping")
        return
    mass_sel = mass.copy()
    mass_sel[~valid_dt] = -np.inf
    i0 = int(np.argmax(mass_sel))

    # q and qa at peak hour (optionally rotation-symmetrized)
    if CFG.SYMMETRIZE:
        q_peak = rot_avg(q[i0])
        qa_peak = rot_avg(qa[i0])
        basis_note = f"rotation-symmetrized ({CFG.N_ROT}×10°)"
    else:
        q_peak = q[i0]
        qa_peak = qa[i0]
        basis_note = "raw peak-hour field"

    dx_m = CFG.DX * M_PER_DEG * np.cos(np.deg2rad(CFG.CENTER_LAT))
    dy_m = CFG.DX * M_PER_DEG
    qdy_peak, qdx_peak = np.gradient(q_peak, dy_m, dx_m, edge_order=2)

    basis = pvtend.compute_orthogonal_basis(
        qa_peak, qdx_peak, qdy_peak, x_rel, y_rel,
        mask=mask_str,
        apply_smoothing=True, smoothing_deg=CFG.SMOOTHING_DEG,
        grid_spacing=CFG.DX, center_lat=CFG.CENTER_LAT,
        include_lap=CFG.INCLUDE_LAP,
    )

    proj = pvtend.project_field(pv_dt[i0], basis, grid_spacing=CFG.DX)

    total = pv_dt[i0]
    recon = proj["recon"]; resid = proj["resid"]
    integ = proj["int"]; prop = proj["prop"]; deform = proj["def"]

    r1 = np.nanmax(np.abs([total, recon, resid]))
    r2 = np.nanmax(np.abs([integ, prop, deform]))

    # ---- decomp_{C,AC}.png (2x3) ----
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8),
                             constrained_layout=True)
    X, Y = np.meshgrid(x_rel, y_rel)

    def _panel(ax, F, vmax, title):
        im = ax.pcolormesh(X, Y, F, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [deg]"); ax.set_ylabel("y [deg]")
        ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.axvline(0, color="k", lw=0.4, alpha=0.5)
        return im

    im1 = _panel(axes[0, 0], total, r1, r"$\partial q/\partial t$")
    _panel(axes[0, 1], recon, r1, "reconstruction")
    _panel(axes[0, 2], resid, r1, "residual")
    fig.colorbar(im1, ax=axes[0, :], shrink=0.8,
                 label=r"PV tendency [K m$^2$ kg$^{-1}$ s$^{-2}$]")

    im2 = _panel(axes[1, 0], integ, r2, r"int:  $\beta\,\phi_1$")
    _panel(axes[1, 1], prop, r2, r"prop: $-a_x\phi_2 - a_y\phi_3$")
    _panel(axes[1, 2], deform, r2,
           r"def:  $-\gamma_1\phi_4 - \gamma_2\phi_5$")
    fig.colorbar(im2, ax=axes[1, :], shrink=0.8,
                 label=r"component tendency [same units]")

    coef = (f"β={proj['beta']:.2e}  ax={proj['ax']:.2e}  "
            f"ay={proj['ay']:.2e}  γ1={proj['gamma1']:.2e}  "
            f"γ2={proj['gamma2']:.2e}  RMSE={proj.get('rmse', np.nan):.2e}")
    fig.suptitle(
        f"{lc.upper()}  {polarity}  Lagrangian composite  "
        f"t={int(t_hour[i0])}h — {basis_note}\n{coef}",
        fontsize=11)

    plots = ROOT / lc / "projections" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    out = plots / f"decomp_{polarity}.png"
    fig.savefig(out, dpi=140); plt.close(fig)

    # ---- decomp_bases_{C,AC}.png (1x5) : show Φ1..Φ5 ----
    phis = [basis.phi_int, basis.phi_dx, basis.phi_dy,
            basis.phi_def, basis.phi_strain]
    labels = [r"$\phi_1$  (int, circular)",
              r"$\phi_2$  (prop $x$, dipole)",
              r"$\phi_3$  (prop $y$, dipole)",
              r"$\phi_4$  (def shear, quadrupole)",
              r"$\phi_5$  (def strain, quadrupole)"]
    rb = max(np.nanmax(np.abs(p)) for p in phis)
    fig, axs = plt.subplots(1, 5, figsize=(18, 4.2),
                            constrained_layout=True)
    last = None
    for ax, F, lab in zip(axs, phis, labels):
        last = ax.pcolormesh(X, Y, F, cmap="RdBu_r",
                             vmin=-rb, vmax=rb, shading="auto")
        ax.set_aspect("equal"); ax.set_title(lab, fontsize=11)
        ax.set_xlabel("x [deg]")
        ax.axhline(0, color="k", lw=0.3, alpha=0.5)
        ax.axvline(0, color="k", lw=0.3, alpha=0.5)
    axs[0].set_ylabel("y [deg]")
    fig.colorbar(last, ax=axs, shrink=0.85,
                 label="basis amplitude (prenorm units)")
    fig.suptitle(
        f"{lc.upper()}  {polarity} orthogonal basis Φ1..Φ5 at "
        f"t*={int(t_hour[i0])}h ({basis_note}, mask q' {mask_str}, "
        f"include_lap={CFG.INCLUDE_LAP})", fontsize=12)
    out_b = plots / f"decomp_bases_{polarity}.png"
    fig.savefig(out_b, dpi=140); plt.close(fig)

    print(f"[{lc}:{polarity}] wrote {out} and {out_b}  "
          f"(peak t={int(t_hour[i0])}h)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for pol in pols:
            process(lc, polarity=pol)
