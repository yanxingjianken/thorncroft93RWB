"""Idealized 3-row decomposition figure for the paper.

For each LC (default lc1, lc2) at a fixed day:
  lc1 → day 6.5 (i.e. middle of the 6..10 d window, t_k = 12 frames)
  lc2 → day 8   (start of the 8..12 d window, t_k = 0 frames)

Produces a 3 × 5 figure:
  Row 1 — pv_dt, recon, resid (cols 1..3 only; cols 4,5 hidden) sharing
          a single colour bar based on the mask region of pv_dt.
          The central-component mask is shown as a dashed contour and
          the fitted ellipse + major axis is drawn on every panel.
  Row 2 — Φ₁..Φ₅ post smoothing & Gram-Schmidt orthogonalisation
          (basis.phi_int, phi_dx, phi_dy, phi_def, phi_strain).
  Row 3 — 3 combined panels, each with its own colour bar:
            β·Φ₁                             (intensification)
            (−aₓ·Φ₂) + (−a_y·Φ₃)            (advection sum)
            (−γ₁·Φ₄) + (−γ₂·Φ₅)            (deformation sum)
          The deformation panel carries:
            • C-axis (compression, α = ½·atan2(γ₁,γ₂)):
              inward arrows toward centre, darkorange.
            • S-axis (stretching, α+90°):
              outward arrows away from centre, royalblue.
          Source: 2026-04-25_gamma_alpha_axes session findings.

The figure is saved to
  outputs/<lc>/projections/zeta250/idealized_plot/<lc>_idealized_3row.png
and a copy of this script is dropped alongside it.
"""
from __future__ import annotations
import sys, argparse, shutil
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import pvtend

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa
from _grad_safe import safe_gradient, mask_vmax  # noqa
from tilt_evolution import _strict_mask, _fit_ellipse, _ellipse_axis_endpoints  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")
M_PER_DEG = 111_000.0

# Frame index (within the windowed composite) at which to do the projection.
# lc1 window 6..10d → 96 hourly frames; day 6.5 → 12h after start → t_k=12.
# lc2 window 8..12d → 96 hourly frames; day 8.0 → t_k=0 (1st valid central diff
# slot is t_k=1, but we want day 8 itself; pv_dt at the very first frame is
# NaN, so we use t_k=1 in practice).
DAY_TK = {"lc1": 12, "lc2": 1}


def _draw_axis_arrows(ax, xc, yc, alpha_deg, arrow_len):
    """Draw compression (C) and stretching (S) axes for the deformation quadrupole.

    Per 2026-04-25_gamma_alpha_axes session findings:
      α = ½·atan2(γ₁, γ₂)   →  compressing axis: inward arrows (darkorange).
      α + 90°                →  stretching  axis: outward arrows (royalblue).
    """
    a = np.deg2rad(alpha_deg)
    b = a + np.pi / 2          # α + 90° = stretching axis
    # ── compression axis (α): arrows pointing inward toward centre ──
    dx_c, dy_c = arrow_len * np.cos(a), arrow_len * np.sin(a)
    for sx, sy in [(dx_c, dy_c), (-dx_c, -dy_c)]:
        ax.add_patch(FancyArrowPatch(
            (xc + sx, yc + sy), (xc, yc),
            arrowstyle="->", mutation_scale=14, color="darkorange", lw=1.6))
    # ── stretching axis (α+90°): arrows pointing outward from centre ──
    dx_s, dy_s = arrow_len * np.cos(b), arrow_len * np.sin(b)
    for sx, sy in [(dx_s, dy_s), (-dx_s, -dy_s)]:
        ax.add_patch(FancyArrowPatch(
            (xc, yc), (xc + sx, yc + sy),
            arrowstyle="->", mutation_scale=14, color="royalblue", lw=1.4))


def _panel(ax, X, Y, F, vmax, title, cmap="RdBu_r"):
    im = ax.pcolormesh(X, Y, F, cmap=cmap, vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-CFG.PATCH_HALF_LON, CFG.PATCH_HALF_LON)
    ax.set_ylim(-CFG.PATCH_HALF_LAT, CFG.PATCH_HALF_LAT)
    ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    ax.axvline(0, color="k", lw=0.3, alpha=0.4)
    return im


def process(lc: str, polarity: str = "C"):
    method = CFG.CANONICAL_METHOD
    comp_nc = ROOT / lc / "composites" / method / f"{polarity}_composite.nc"
    if not comp_nc.exists():
        print(f"[{lc}:{polarity}] composite missing: {comp_nc}")
        return
    ds = xr.open_dataset(comp_nc)
    q   = ds["total_composite"].values.astype("float64")
    qa  = ds["anom_composite"].values.astype("float64")
    x   = ds["x"].values.astype("float64")
    y   = ds["y"].values.astype("float64")
    t_h = ds["t"].values.astype("float64")
    ds.close()

    t_k = DAY_TK.get(lc, 1)
    if not (1 <= t_k <= q.shape[0] - 2):
        print(f"[{lc}] t_k={t_k} out of range")
        return

    # Centred difference for ∂q/∂t (per second)
    dt_s = (t_h[t_k + 1] - t_h[t_k - 1]) * 3600.0
    pv_dt = (q[t_k + 1] - q[t_k - 1]) / dt_s
    qa_k  = qa[t_k]
    q_k   = q[t_k]

    X, Y = np.meshgrid(x, y)
    dx_m = CFG.DX * M_PER_DEG * np.cos(np.deg2rad(CFG.CENTER_LAT))
    dy_m = CFG.DX * M_PER_DEG

    # gradients with safe stencil
    qdx, qdy = safe_gradient(q_k, dy_m, dx_m)

    # basis (with smoothing + Gram-Schmidt as inside pvtend)
    basis = pvtend.compute_orthogonal_basis(
        qa_k, qdx, qdy, x, y,
        center_lat=CFG.CENTER_LAT, grid_spacing=CFG.DX,
        apply_smoothing=True, smoothing_deg=CFG.SMOOTHING_DEG,
        include_lap=CFG.INCLUDE_LAP,
    )
    proj = pvtend.project_field(pv_dt, basis, grid_spacing=CFG.DX)
    recon = proj["recon"]; resid = proj["resid"]

    beta = float(proj["beta"]); ax_ = float(proj["ax"]); ay_ = float(proj["ay"])
    g1 = float(proj["gamma1"]); g2 = float(proj["gamma2"])
    # raw coefficients used to scale individual phi_dx, phi_dy panels
    ax_raw = float(proj.get("ax_raw", ax_))
    ay_raw = float(proj.get("ay_raw", ay_))
    g1_raw = float(proj.get("gamma1_raw", g1))
    g2_raw = float(proj.get("gamma2_raw", g2))

    # α = ½·atan2(γ₁, γ₂)   (source: 2026-04-25_gamma_alpha_axes session findings)
    alpha_deg = 0.5 * np.degrees(np.arctan2(g1, g2))
    alpha_deg = ((alpha_deg + 90.0) % 180.0) - 90.0   # wrap to (-90°, 90°]

    # central-component mask
    fit_thr = float(CFG.METHOD[method]["mask_thresh"])
    guard_r = float(CFG.PATCH_HALF_LAT - CFG.GUARD_PAD_DEG)
    raw_m = _strict_mask(qa_k, polarity, fit_thr, X, Y, guard_r)

    # ellipse fit (on |qa| weighted, no lat weighting)
    theta_e, a_e, b_e, xc, yc = _fit_ellipse(raw_m, qa_k, X, Y)

    # ---- figure: rows 0,1 = 5 panels; row 2 = 3 wider panels ----------
    from matplotlib import gridspec
    fig = plt.figure(figsize=(17, 11))
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                              top=0.92, bottom=0.04, left=0.04, right=0.98)
    inner_r0 = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[0], wspace=0.28)
    inner_r1 = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1], wspace=0.28)
    inner_r2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2], wspace=0.40)
    axes_r0 = [fig.add_subplot(inner_r0[0, j]) for j in range(5)]
    axes_r1 = [fig.add_subplot(inner_r1[0, j]) for j in range(5)]
    axes_r2 = [fig.add_subplot(inner_r2[0, j]) for j in range(3)]

    # row 0 — pv_dt / recon / resid + shared cbar
    r1 = max(mask_vmax(pv_dt, raw_m),
             mask_vmax(recon, raw_m),
             mask_vmax(resid, raw_m))
    fields_r0  = [pv_dt, recon, resid, None, None]
    titles_r0  = [r"$\partial q/\partial t$", "reconstruction",
                  "residual", "", ""]
    im0 = None
    for j, (F, title_) in enumerate(zip(fields_r0, titles_r0)):
        ax = axes_r0[j]
        if F is None:
            ax.set_visible(False)
            continue
        im0 = _panel(ax, X, Y, F, r1, title_)
        ax.contour(X, Y, raw_m.astype(float), levels=[0.5],
                   colors="k", linewidths=1.0, linestyles="--")
        if np.isfinite(theta_e):
            e = Ellipse((xc, yc), 2 * a_e, 2 * b_e, angle=theta_e,
                        fill=False, edgecolor="green", lw=1.8)
            ax.add_patch(e)
            xs, ys = _ellipse_axis_endpoints(xc, yc, a_e, theta_e)
            ax.plot(xs, ys, "g-", lw=1.4)
    if im0 is not None:
        fig.colorbar(im0, ax=axes_r0[:3], shrink=0.8,
                     label=r"$\partial q/\partial t$ [s$^{-2}$]")

    # row 1 — orthogonal bases Φ₁..Φ₅ + shared cbar
    phis = [basis.phi_int, basis.phi_dx, basis.phi_dy,
            basis.phi_def, basis.phi_strain]
    phi_lbls = [r"$\Phi_1$ (intensify)", r"$\Phi_2$ (prop$_x$)",
                r"$\Phi_3$ (prop$_y$)", r"$\Phi_4$ (def $q_{xy}$)",
                r"$\Phi_5$ (def $q_{xx}\!-\!q_{yy}$)"]
    rb = max(mask_vmax(p, raw_m) for p in phis)
    im_b = None
    for j, (P, lab) in enumerate(zip(phis, phi_lbls)):
        im_b = _panel(axes_r1[j], X, Y, P, rb, lab)
    if im_b is not None:
        fig.colorbar(im_b, ax=axes_r1, shrink=0.8, label="basis")

    # row 2 — 3 combined panels, each with its own colour bar
    #   [0] β·Φ₁                             intensification
    #   [1] (−aₓ·Φ₂) + (−a_y·Φ₃)           advection sum
    #   [2] (−γ₁·Φ₄) + (−γ₂·Φ₅)           deformation sum + C/S arrows
    s_deg = ((alpha_deg + 90.0) % 180.0) - 90.0  # S-axis angle
    F_intens = beta    * basis.phi_int
    F_adv    = (-ax_raw * basis.phi_dx) + (-ay_raw * basis.phi_dy)
    F_def    = (-g1_raw * basis.phi_def) + (-g2_raw * basis.phi_strain)
    lbl_intens = rf"$\beta\,\Phi_1$  ($\beta={beta:+.2e}$)"
    lbl_adv    = (rf"$(-a_x\Phi_2)+(-a_y\Phi_3)$"
                  rf"  $a_x\!=\!{ax_:+.2e},\;a_y\!=\!{ay_:+.2e}$")
    lbl_def    = (rf"$(-\gamma_1\Phi_4)+(-\gamma_2\Phi_5)$"
                  rf"  $\gamma_1\!=\!{g1:+.2e},\;\gamma_2\!=\!{g2:+.2e}$"
                  rf"  $\alpha\!=\!{alpha_deg:+.1f}^\circ$")
    arrow_len = 0.8 * (a_e if np.isfinite(a_e) else 10.0)
    for j, (F, lab) in enumerate([(F_intens, lbl_intens),
                                   (F_adv,    lbl_adv),
                                   (F_def,    lbl_def)]):
        ax2 = axes_r2[j]
        vj = mask_vmax(F, raw_m)
        im2 = _panel(ax2, X, Y, F, vj, lab)
        fig.colorbar(im2, ax=ax2, shrink=0.85, fraction=0.046, pad=0.04,
                     label="scaled basis")
        ax2.contour(X, Y, raw_m.astype(float), levels=[0.5],
                    colors="k", linewidths=0.8, linestyles="--")
        if np.isfinite(theta_e):
            e2 = Ellipse((xc, yc), 2 * a_e, 2 * b_e, angle=theta_e,
                         fill=False, edgecolor="green", lw=1.4)
            ax2.add_patch(e2)
            xs2, ys2 = _ellipse_axis_endpoints(xc, yc, a_e, theta_e)
            ax2.plot(xs2, ys2, "g-", lw=1.2)
        if j == 2 and np.isfinite(theta_e):
            _draw_axis_arrows(ax2, xc, yc, alpha_deg, arrow_len)
            ax2.text(0.03, 0.97,
                     f"C (compress) α={alpha_deg:+.1f}°\n"
                     f"S (stretch) α+90°={s_deg:+.1f}°",
                     transform=ax2.transAxes, fontsize=7.5,
                     va="top", ha="left", color="k",
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", alpha=0.75))

    fig.suptitle(
        f"{lc.upper()} {method}/{polarity}  "
        f"day {(t_h[t_k]) / 24:.2f}  "
        rf"$\alpha={alpha_deg:+.1f}^\circ$",
        fontsize=12)

    out_dir = ROOT / lc / "projections" / method / "idealized_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{lc}_idealized_3row_{polarity}.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # drop a copy of this script
    shutil.copy2(Path(__file__).resolve(), out_dir / Path(__file__).name)
    print(f"[{lc}:{polarity}] wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="C")
    args = ap.parse_args()
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for pol in pols:
            process(lc, pol)
