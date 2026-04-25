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
  Row 3 — Scaled bases:
            β·Φ₁,  −aₓ·Φ₂,  −a_y·Φ₃,  −γ₁·Φ₄,  −γ₂·Φ₅
          with numeric annotations of the fitted coefficients and the
          stretching angle  α = 90° − ½·atan2(γ₂, γ₁).
          On the deformation panels (cols 4,5) the stretching axis (α)
          is drawn as outward-pointing arrows and the compressing axis
          (α+90°) as inward-pointing arrows.

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


def _draw_axis_arrows(ax, xc, yc, alpha_deg, length, color):
    """Outward stretching arrow at angle α and inward compressing arrow at α+90°."""
    a = np.deg2rad(alpha_deg)
    # stretching: outward both ways from centre
    dx, dy = length * np.cos(a), length * np.sin(a)
    for sx, sy in [(dx, dy), (-dx, -dy)]:
        ax.add_patch(FancyArrowPatch((xc, yc), (xc + sx, yc + sy),
                                     arrowstyle="->", mutation_scale=14,
                                     color=color, lw=1.6))
    # compressing: inward toward centre from α+90° endpoints
    b = a + np.pi / 2
    dx2, dy2 = length * np.cos(b), length * np.sin(b)
    for sx, sy in [(dx2, dy2), (-dx2, -dy2)]:
        ax.add_patch(FancyArrowPatch((xc + sx, yc + sy), (xc, yc),
                                     arrowstyle="->", mutation_scale=14,
                                     color="purple", lw=1.4))


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

    alpha_deg = 90.0 - 0.5 * np.degrees(np.arctan2(g2, g1))
    # wrap to [-90,90)
    while alpha_deg >= 90.0:
        alpha_deg -= 180.0
    while alpha_deg < -90.0:
        alpha_deg += 180.0

    # central-component mask
    fit_thr = float(CFG.METHOD[method]["mask_thresh"])
    guard_r = float(CFG.PATCH_HALF_LAT - CFG.GUARD_PAD_DEG)
    raw_m = _strict_mask(qa_k, polarity, fit_thr, X, Y, guard_r)

    # ellipse fit (on |qa| weighted, no lat weighting)
    theta_e, a_e, b_e, xc, yc = _fit_ellipse(raw_m, qa_k, X, Y)

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(3, 5, figsize=(17, 10),
                             constrained_layout=True)

    # row 1 - share cbar
    r1 = max(mask_vmax(pv_dt, raw_m),
             mask_vmax(recon, raw_m),
             mask_vmax(resid, raw_m))
    titles_r1 = [r"$\partial q/\partial t$", "reconstruction",
                 "residual", "", ""]
    fields_r1 = [pv_dt, recon, resid, None, None]
    for j, (F, t) in enumerate(zip(fields_r1, titles_r1)):
        ax = axes[0, j]
        if F is None:
            ax.set_visible(False)
            continue
        im = _panel(ax, X, Y, F, r1, t)
        ax.contour(X, Y, raw_m.astype(float), levels=[0.5],
                   colors="k", linewidths=1.0, linestyles="--")
        if np.isfinite(theta_e):
            e = Ellipse((xc, yc), 2 * a_e, 2 * b_e, angle=theta_e,
                        fill=False, edgecolor="green", lw=1.8)
            ax.add_patch(e)
            xs, ys = _ellipse_axis_endpoints(xc, yc, a_e, theta_e)
            ax.plot(xs, ys, "g-", lw=1.4)
    fig.colorbar(im, ax=axes[0, :3].tolist(), shrink=0.8,
                 label=r"$\partial q/\partial t$ [s$^{-2}$]")

    # row 2 — bases
    phis = [basis.phi_int, basis.phi_dx, basis.phi_dy,
            basis.phi_def, basis.phi_strain]
    phi_lbls = [r"$\Phi_1$ (intensify)", r"$\Phi_2$ (prop$_x$)",
                r"$\Phi_3$ (prop$_y$)", r"$\Phi_4$ (def shear $q_{xy}$)",
                r"$\Phi_5$ (def strain $q_{xx}-q_{yy}$)"]
    rb = max(mask_vmax(p, raw_m) for p in phis)
    for j, (P, lab) in enumerate(zip(phis, phi_lbls)):
        ax = axes[1, j]
        im_b = _panel(ax, X, Y, P, rb, lab)
    fig.colorbar(im_b, ax=axes[1, :].tolist(), shrink=0.8, label="basis")

    # row 3 — scaled bases (use *_raw coefficients on the raw smoothed bases)
    scaled = [
        ( beta    * basis.phi_int,    rf"$\beta\,\Phi_1\;(\beta={beta:+.2e})$"),
        (-ax_raw  * basis.phi_dx,     rf"$-a_x\,\Phi_2\;(a_x={ax_:+.2e})$"),
        (-ay_raw  * basis.phi_dy,     rf"$-a_y\,\Phi_3\;(a_y={ay_:+.2e})$"),
        (-g1_raw  * basis.phi_def,    rf"$-\gamma_1\,\Phi_4\;(\gamma_1={g1:+.2e})$"),
        (-g2_raw  * basis.phi_strain, rf"$-\gamma_2\,\Phi_5\;(\gamma_2={g2:+.2e})$"),
    ]
    rs = max(mask_vmax(F, raw_m) for F, _ in scaled)
    arrow_len = 0.8 * (a_e if np.isfinite(a_e) else 10.0)
    for j, (F, lab) in enumerate(scaled):
        ax = axes[2, j]
        im_s = _panel(ax, X, Y, F, rs, lab)
        if j in (3, 4) and np.isfinite(theta_e):
            _draw_axis_arrows(ax, xc, yc, alpha_deg, arrow_len, "darkorange")
    fig.colorbar(im_s, ax=axes[2, :].tolist(), shrink=0.8,
                 label="scaled basis")

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
