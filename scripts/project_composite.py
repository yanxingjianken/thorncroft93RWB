"""Rotation-symmetrized 5-basis decomposition of the cyclone composite.

Peak hour t* = argmax_t sum(q'(t)>0).

At t*, both q and q' are symmetrized by averaging 36 rotations
(10 deg step) about the patch center (0,0), giving a near
axisymmetric field.  PV gradients are recomputed from the
symmetrized q via np.gradient (so prenorm / smoothing / Gram-
Schmidt all consume the symmetric version). The resulting basis
has Φ1 circular, Φ2/Φ3 dipoles, Φ4/Φ5 quadrupoles.

The un-symmetrized pv_dt at t* is projected onto this symmetric
basis to produce the 5-component decomposition.

Outputs (per LC, cyclone only):
  decomp_C.png        2x3 panel (top: total pv_dt | recon | resid;
                                 bot: int | prop | def).
  decomp_bases_C.png  1x5 panel showing the symmetric Φ1..Φ5.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.ndimage import rotate as nd_rotate
import matplotlib.pyplot as plt
import pvtend

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")
DX = 1.0
CENTER_LAT = 50.0
M_PER_DEG = 111_000.0
N_ROT = 36  # 360/10 deg


def rot_avg(field: np.ndarray) -> np.ndarray:
    """Average over 36 rotations of 10 deg about patch center."""
    out = np.zeros_like(field, dtype="float64")
    for k in range(N_ROT):
        ang = k * (360.0 / N_ROT)
        out += nd_rotate(field, ang, reshape=False, order=1,
                         mode="nearest")
    return out / N_ROT


def process(lc: str):
    ds = xr.open_dataset(ROOT / lc / "composites" / "C_composite.nc")
    q = ds["pv_composite"].values.astype("float64")
    t_hour = ds["t"].values
    x_rel = ds["x"].values.astype("float64")
    y_rel = ds["y"].values.astype("float64")

    qa = q - np.nanmean(q, axis=-1, keepdims=True)

    # pv_dt via centered difference on raw composite
    pv_dt = np.full_like(qa, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2 * 3600.0)

    # Peak hour by total positive anomaly mass
    pos_sum = np.nansum(np.where(qa > 0, qa, 0.0), axis=(1, 2))
    pos_sum[~np.isfinite(pos_sum)] = -np.inf
    pos_sum[0] = pos_sum[-1] = -np.inf
    i0 = int(np.argmax(pos_sum))

    # Symmetrize BOTH q and q' at the peak hour
    q_sym = rot_avg(q[i0])
    qa_sym = rot_avg(qa[i0])

    # Recompute gradients from the symmetrized q (axisymmetric ->
    # gradients are radial/orthoradial)
    dx_m = DX * M_PER_DEG * np.cos(np.deg2rad(CENTER_LAT))
    dy_m = DX * M_PER_DEG
    qdy_sym, qdx_sym = np.gradient(q_sym, dy_m, dx_m, edge_order=2)

    basis = pvtend.compute_orthogonal_basis(
        qa_sym, qdx_sym, qdy_sym, x_rel, y_rel,
        mask="> 0", apply_smoothing=True, smoothing_deg=3.0,
        grid_spacing=DX, center_lat=CENTER_LAT, include_lap=False,
    )

    # Project the un-symmetrized pv_dt at t* onto the symmetric basis
    proj = pvtend.project_field(pv_dt[i0], basis, grid_spacing=DX)

    total = pv_dt[i0]
    recon = proj["recon"]; resid = proj["resid"]
    integ = proj["int"]; prop = proj["prop"]; deform = proj["def"]

    r1 = np.nanmax(np.abs([total, recon, resid]))
    r2 = np.nanmax(np.abs([integ, prop, deform]))

    # ---- decomp_C.png (2x3) ----
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
    fig.suptitle(f"{lc.upper()}  cyclone Lagrangian composite  "
                 f"t={int(t_hour[i0])}h — rotation-symmetrized "
                 f"basis ({N_ROT}x10°)\n{coef}", fontsize=11)

    plots = ROOT / lc / "projections" / "plots"
    scripts_dir = ROOT / lc / "projections" / "scripts"
    plots.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    out = plots / "decomp_C.png"
    fig.savefig(out, dpi=140); plt.close(fig)

    # ---- decomp_bases_C.png (1x5) : show symmetric Φ1..Φ5 ----
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
        f"{lc.upper()}  rotation-symmetrized orthogonal basis Φ1..Φ5 "
        f"at t*={int(t_hour[i0])}h ({N_ROT}×10°, mask q'>0, "
        f"include_lap=False)", fontsize=12)
    out_b = plots / "decomp_bases_C.png"
    fig.savefig(out_b, dpi=140); plt.close(fig)

    shutil.copy2(Path(__file__).resolve(),
                 scripts_dir / Path(__file__).name)
    print(f"[{lc}] wrote {out} and {out_b}  (peak t={int(t_hour[i0])}h)")


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        process(lc)
