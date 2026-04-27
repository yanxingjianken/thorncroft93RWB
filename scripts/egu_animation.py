"""4×3 EGU animation per LC × polarity (back-extended composites).

Layout per frame:

  Row 0 — patch fields
    [0] zeta_250 (q' anomaly only, raw shading)
    [1] total q with the q'=0 contour overlay (mask outline)
    [2] q' anomaly shading + central-component mask + fitted ellipse
        + green major-axis line

  Row 1 — tendency decomposition
    [0] real ∂q/∂t at the frame
    [1] reconstruction = β·Φ₁ − aₓ·Φ₂ − a_y·Φ₃ − γ₁·Φ₄ − γ₂·Φ₅
    [2] residual (real − recon) with mask contour

  Row 2 — basis projections (combined)
    [0] β·Φ₁  (intensification)
    [1] −aₓ·Φ₂ − a_y·Φ₃  (propagation sum)
    [2] −γ₁·Φ₄ − γ₂·Φ₅  (deformation sum) + ellipse + major axis

  Row 3 — time-series predictions (cropped from --start_hour onward)
    [0] mask area:  obs (km²) vs A(t0)·exp(∫β dt)        (e-folding rate)
    [1] centroid travel:
          obs (xc_obs−xc_obs[t0], yc_obs−yc_obs[t0]) [°]
          pred = ∫aₓ dt / (M_PER_DEG·cos(lat))   [°]
                 ∫a_y dt / M_PER_DEG             [°]
    [2] accumulated tilt:
          obs ∑Δθ_obs  vs  pred ∑Δθ_pred (wrap-aware, both rebased to
          0 at the start hour).

References:
  • β e-folding integration:
    /net/flood/data2/users/x_yan/pvtend/outputs/egu_plots/beta_projection/
      plot_beta_projection_onset.py
  • aₓ propagation integration:
    /net/flood/data2/users/x_yan/pvtend/outputs/egu_plots/ax_projection/
      plot_ax_projection_peak.py
  • Tilt accumulator:
    outputs/lc?/projections/zeta250_back/plots/theta_tilt_accum_*.png
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pvtend

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa
from _grad_safe import safe_gradient, mask_vmax  # noqa
from tilt_evolution import (  # noqa
    _strict_mask, _fit_ellipse, _ellipse_axis_endpoints,
    _wrap_diff, _unwrap_mod180, _fill,
)

ROOT = Path(
    "/net/flood/data2/users/x_yan/literature_review/rwb/"
    "thorncroft93_baroclinic/thorncroft_rwb/outputs"
)
EGU = Path(
    "/net/flood/data2/users/x_yan/literature_review/rwb/"
    "thorncroft93_baroclinic/thorncroft_rwb/egu"
)
M_PER_DEG = 111_000.0
DT_S = 3600.0   # 1-hour composite cadence


# ---------------------------------------------------------------------------
def _scan_frames(lc, method, polarity):
    """Per-frame mask/ellipse/projection pass. Returns dict of arrays."""
    spec_key = method if method in CFG.METHOD else CFG.CANONICAL_METHOD
    spec = CFG.METHOD[spec_key]
    src = ROOT / lc / "composites" / method / f"{polarity}_composite.nc"
    if not src.exists():
        raise FileNotFoundError(src)
    ds = xr.open_dataset(src)
    q  = ds["total_composite"].values.astype("float64")
    qa = ds["anom_composite"].values.astype("float64")
    t_h = ds["t"].values.astype("float64")
    win_h0 = ds.attrs.get("win_h0_sim", None)
    win_h0 = int(win_h0) if win_h0 is not None else int(CFG.window_for(lc)[0])
    x_rel = ds["x"].values.astype("float64")
    y_rel = ds["y"].values.astype("float64")
    ds.close()

    nt = q.shape[0]
    X, Y = np.meshgrid(x_rel, y_rel)
    dx_m = CFG.DX * M_PER_DEG * np.cos(np.deg2rad(CFG.CENTER_LAT))
    dy_m = CFG.DX * M_PER_DEG
    fit_thr = float(spec["mask_thresh"])
    guard_r = float(CFG.PATCH_HALF_LAT - CFG.GUARD_PAD_DEG)
    cell_area_km2 = (CFG.DX * M_PER_DEG / 1000.0) ** 2 * np.cos(
        np.deg2rad(CFG.CENTER_LAT)
    )

    # tendency from centred difference of TOTAL field (units: q per second)
    pv_dt = np.full_like(q, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2.0 * DT_S)

    # outputs
    mask_frames = np.zeros_like(q, dtype=bool)
    recon = np.full_like(q, np.nan)
    resid = np.full_like(q, np.nan)
    inten = np.full_like(q, np.nan)
    prop  = np.full_like(q, np.nan)
    defm  = np.full_like(q, np.nan)
    beta  = np.full(nt, np.nan)
    ax_   = np.full(nt, np.nan)
    ay_   = np.full(nt, np.nan)
    g1    = np.full(nt, np.nan)
    g2    = np.full(nt, np.nan)
    ax_raw = np.full(nt, np.nan)
    ay_raw = np.full(nt, np.nan)
    g1_raw = np.full(nt, np.nan)
    g2_raw = np.full(nt, np.nan)
    theta_obs = np.full(nt, np.nan)
    a_obs = np.full(nt, np.nan); b_obs = np.full(nt, np.nan)
    xc_obs = np.full(nt, np.nan); yc_obs = np.full(nt, np.nan)
    theta_pred = np.full(nt, np.nan)
    area_obs = np.full(nt, np.nan)

    print(f"  scanning {nt} frames ...", flush=True)
    for i in range(nt):
        qai = _fill(qa[i])
        qi  = _fill(q[i])
        m = _strict_mask(qai, polarity, fit_thr, X, Y, guard_r)
        if m.sum() < 8:
            continue
        mask_frames[i] = m
        th, a_, b_, xcc, ycc = _fit_ellipse(m, qai, X, Y)
        theta_obs[i] = th; a_obs[i] = a_; b_obs[i] = b_
        xc_obs[i] = xcc; yc_obs[i] = ycc
        area_obs[i] = m.sum() * cell_area_km2

        if i == 0 or i == nt - 1 or not np.isfinite(pv_dt[i]).any():
            continue

        qdx, qdy = safe_gradient(q[i], dy_m, dx_m)
        try:
            basis = pvtend.compute_orthogonal_basis(
                qai, qdx, qdy, x_rel, y_rel,
                center_lat=CFG.CENTER_LAT, grid_spacing=CFG.DX,
                apply_smoothing=True, smoothing_deg=CFG.SMOOTHING_DEG,
                include_lap=CFG.INCLUDE_LAP,
            )
            proj = pvtend.project_field(
                _fill(pv_dt[i]), basis, grid_spacing=CFG.DX
            )
        except Exception as exc:
            print(f"    [t={int(t_h[i])}h] basis/proj failed: {exc}",
                  flush=True)
            continue

        recon[i] = proj["recon"]
        # full-domain residual (pvtend masks resid to basis.mask;
        # we want to visualise the residual everywhere)
        resid[i] = _fill(pv_dt[i]) - proj["recon"]
        beta[i] = float(proj["beta"])
        ax_[i]  = float(proj["ax"])
        ay_[i]  = float(proj["ay"])
        g1[i]   = float(proj["gamma1"])
        g2[i]   = float(proj["gamma2"])
        ax_raw[i] = float(proj.get("ax_raw", ax_[i]))
        ay_raw[i] = float(proj.get("ay_raw", ay_[i]))
        g1_raw[i] = float(proj.get("gamma1_raw", g1[i]))
        g2_raw[i] = float(proj.get("gamma2_raw", g2[i]))

        # combined component fields (already computed inside project_field)
        inten[i] = proj["int"]
        prop[i]  = proj["prop"]
        defm[i]  = proj["def"]

        # 1-h Euler prediction of tilt
        qa_next = qai + DT_S * proj["recon"]
        m2 = _strict_mask(qa_next, polarity, fit_thr, X, Y, guard_r)
        if m2.sum() >= 8:
            th2, *_ = _fit_ellipse(m2, qa_next, X, Y)
            theta_pred[i] = th2

        if i % 25 == 0:
            print(f"    [{i:3d}/{nt}] t={int(t_h[i]):3d}h  "
                  f"β={beta[i]:+.2e} aₓ={ax_[i]:+.2e} a_y={ay_[i]:+.2e}",
                  flush=True)

    return dict(
        q=q, qa=qa, pv_dt=pv_dt, recon=recon, resid=resid,
        inten=inten, prop=prop, defm=defm,
        beta=beta, ax=ax_, ay=ay_, g1=g1, g2=g2,
        mask_frames=mask_frames,
        theta_obs=theta_obs, theta_pred=theta_pred,
        a_obs=a_obs, b_obs=b_obs, xc_obs=xc_obs, yc_obs=yc_obs,
        area_obs=area_obs,
        x_rel=x_rel, y_rel=y_rel, X=X, Y=Y, t_h=t_h, win_h0=win_h0,
        cell_area_km2=cell_area_km2,
    )


# ---------------------------------------------------------------------------
def _integrate_predictions(d, idx0, idx_max):
    """Compute time-integrated predictions starting from ``idx0``.

    A_pred  = A_obs[idx0] * exp(∫β dt)
    x_pred  = ∫aₓ dt / (M_PER_DEG·cos(CENTER_LAT))   [°]
    y_pred  = ∫a_y dt / M_PER_DEG                     [°]
    Δθ_pred = ∑_{j=idx0+1..i} wrap(theta_pred[j-1] - theta_obs[j-1])
              (same convention as tilt_evolution accumulator)
    Δθ_obs  = ∑_{j=idx0+1..i} wrap(theta_obs[j] - theta_obs[j-1])
    """
    nt = len(d["beta"])
    cosL = np.cos(np.deg2rad(CFG.CENTER_LAT))
    A_pred = np.full(nt, np.nan)
    xp_pred = np.full(nt, np.nan)
    yp_pred = np.full(nt, np.nan)
    dth_obs_disp = np.full(nt, np.nan)
    dth_pred_disp = np.full(nt, np.nan)

    if not np.isfinite(d["area_obs"][idx0]):
        # fall back to first finite area at/after idx0
        finite_idx = np.where(np.isfinite(d["area_obs"][idx0:]))[0]
        if not finite_idx.size:
            return A_pred, xp_pred, yp_pred, dth_obs_disp, dth_pred_disp
        idx0 = idx0 + int(finite_idx[0])
    A0 = float(d["area_obs"][idx0])
    xc0 = float(d["xc_obs"][idx0]) if np.isfinite(d["xc_obs"][idx0]) else 0.0
    yc0 = float(d["yc_obs"][idx0]) if np.isfinite(d["yc_obs"][idx0]) else 0.0

    # cumulative integrals
    int_beta = 0.0
    int_ax = 0.0
    int_ay = 0.0
    int_dth_obs = 0.0
    int_dth_pred = 0.0
    A_pred[idx0] = A0
    xp_pred[idx0] = 0.0
    yp_pred[idx0] = 0.0
    dth_obs_disp[idx0] = 0.0
    dth_pred_disp[idx0] = 0.0
    for i in range(idx0 + 1, min(idx_max + 1, nt)):
        # trapezoidal integration over 1-h step
        b1 = d["beta"][i - 1]; b2 = d["beta"][i]
        ax1 = d["ax"][i - 1];  ax2 = d["ax"][i]
        ay1 = d["ay"][i - 1];  ay2 = d["ay"][i]
        if np.isfinite(b1) and np.isfinite(b2):
            int_beta += 0.5 * (b1 + b2) * DT_S
        elif np.isfinite(b2):
            int_beta += b2 * DT_S
        if np.isfinite(ax1) and np.isfinite(ax2):
            int_ax += 0.5 * (ax1 + ax2) * DT_S
        elif np.isfinite(ax2):
            int_ax += ax2 * DT_S
        if np.isfinite(ay1) and np.isfinite(ay2):
            int_ay += 0.5 * (ay1 + ay2) * DT_S
        elif np.isfinite(ay2):
            int_ay += ay2 * DT_S
        A_pred[i] = A0 * np.exp(int_beta)
        xp_pred[i] = int_ax / (M_PER_DEG * cosL)
        yp_pred[i] = int_ay / M_PER_DEG

        # tilt accumulators (wrap-aware mod-180)
        if (np.isfinite(d["theta_obs"][i]) and
                np.isfinite(d["theta_obs"][i - 1])):
            int_dth_obs += _wrap_diff(d["theta_obs"][i],
                                      d["theta_obs"][i - 1])
        if (np.isfinite(d["theta_pred"][i - 1]) and
                np.isfinite(d["theta_obs"][i - 1])):
            int_dth_pred += _wrap_diff(d["theta_pred"][i - 1],
                                       d["theta_obs"][i - 1])
        dth_obs_disp[i] = int_dth_obs
        dth_pred_disp[i] = int_dth_pred

    return A_pred, xp_pred, yp_pred, dth_obs_disp, dth_pred_disp


# ---------------------------------------------------------------------------
def _vmax_mask(arr, mask, pct=99.0):
    """Robust symmetric colour limit: ``pct``-th percentile of |arr| within mask.

    Falls back to global percentile if mask is empty. Returns 1.0 as a last
    resort for all-NaN inputs.
    """
    a = np.asarray(arr)
    sel = mask & np.isfinite(a)
    if sel.any():
        v = float(np.nanpercentile(np.abs(a[sel]), pct))
    else:
        finite = np.isfinite(a)
        if not finite.any():
            return 1.0
        v = float(np.nanpercentile(np.abs(a[finite]), pct))
    return v if v > 0 else 1.0


def make_animation(lc, polarity, start_hour_abs, method="zeta250_back"):
    print(f"[{lc}:{polarity}] start_hour={start_hour_abs}")
    d = _scan_frames(lc, method, polarity)
    nt = d["q"].shape[0]
    win_h0 = d["win_h0"]
    t_abs = d["t_h"] + win_h0

    idx0 = int(np.searchsorted(t_abs, float(start_hour_abs)))
    if idx0 >= nt:
        print(f"  start_hour {start_hour_abs} past last frame; abort")
        return
    # drop the last hours (frames are hourly) — the very-end frames are
    # noisy because pv_dt uses centred differences and basis projections
    # become unstable as the vortex breaks down. LC1 needs an extra 10 h
    # trimmed because its breakdown phase is more chaotic.
    tail_drop = 20 if lc == "lc1" else 10
    idx_end = max(idx0 + 1, nt - tail_drop)
    valid = list(range(idx0, idx_end))

    A_pred, xp_pred, yp_pred, dth_obs, dth_pred = _integrate_predictions(
        d, idx0, idx_end - 1
    )

    # ── colour ranges (mask-region based, max over valid frames) ──
    mf = d["mask_frames"]
    sub = slice(idx0, idx_end)
    vm_q   = _vmax_mask(d["q"][sub],     mf[sub])
    vm_qa  = _vmax_mask(d["qa"][sub],    mf[sub])
    # rows 1 & 2 use 95th percentile for tighter colour scales
    vm_dt  = _vmax_mask(d["pv_dt"][sub], mf[sub], pct=95.0)
    vm_rec = max(_vmax_mask(d["recon"][sub], mf[sub], pct=95.0),
                 _vmax_mask(d["resid"][sub], mf[sub], pct=95.0),
                 vm_dt)
    vm_int = _vmax_mask(d["inten"][sub], mf[sub], pct=95.0)
    vm_prp = _vmax_mask(d["prop"][sub],  mf[sub], pct=95.0)
    vm_def = _vmax_mask(d["defm"][sub],  mf[sub], pct=95.0)

    # ── figure ──
    fig = plt.figure(figsize=(15.5, 14))
    outer = gridspec.GridSpec(
        4, 1, figure=fig, hspace=0.55,
        top=0.94, bottom=0.05, left=0.05, right=0.97,
    )
    rows = [gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=outer[r], wspace=0.40)
            for r in range(4)]
    axes = [[fig.add_subplot(rows[r][0, c]) for c in range(3)]
            for r in range(4)]
    for r in range(3):
        for ax in axes[r]:
            ax.set_xlim(-CFG.PATCH_HALF_LON, CFG.PATCH_HALF_LON)
            ax.set_ylim(-CFG.PATCH_HALF_LAT, CFG.PATCH_HALF_LAT)
            ax.set_aspect("auto")

    # static colour bars (one per panel, except row 3)
    from matplotlib import cm, colors as mcolors
    def _cbar(ax, vmax, label):
        sm = cm.ScalarMappable(norm=mcolors.Normalize(-vmax, vmax),
                               cmap="RdBu_r")
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # row 0
    _cbar(axes[0][0], vm_qa, r"$q'$ (zeta250)")
    _cbar(axes[0][1], vm_q,  r"$q$ total")
    _cbar(axes[0][2], vm_qa, r"$q'$")
    # row 1
    for c in range(3):
        _cbar(axes[1][c], vm_rec, r"$\partial q/\partial t$")
    # row 2
    _cbar(axes[2][0], vm_int, r"$\beta\Phi_1$")
    _cbar(axes[2][1], vm_prp, r"$-a_x\Phi_2-a_y\Phi_3$")
    _cbar(axes[2][2], vm_def, r"$-\gamma_1\Phi_4-\gamma_2\Phi_5$")

    X, Y = d["X"], d["Y"]

    # ── row-3 line plots (pre-drawn; per-frame we only update the now-line)
    ax_area = axes[3][0]; ax_cen = axes[3][1]; ax_til = axes[3][2]

    # area
    A_obs_disp = d["area_obs"].copy()
    if np.isfinite(A_obs_disp[idx0]):
        # plot relative to start (km²)
        ax_area.plot(t_abs[valid], A_obs_disp[valid], "g-", lw=2,
                     label=r"$A_\mathrm{obs}$")
        ax_area.plot(t_abs[valid], A_pred[valid], "c--", lw=1.5,
                     label=r"$A_0\cdot e^{\int\beta\,dt}$")
    ax_area.set_xlabel("simulation hour"); ax_area.set_ylabel(r"area [km$^2$]")
    ax_area.set_title("mask area vs e-folding β")
    ax_area.legend(fontsize=8); ax_area.grid(alpha=0.3)
    ax_area.set_xlim(left=float(start_hour_abs))

    # centroid: total displacement |Δr| = sqrt(Δx² + Δy²)
    disp_obs = np.full(nt, np.nan)
    disp_pred = np.full(nt, np.nan)
    if np.isfinite(d["xc_obs"][idx0]) and np.isfinite(d["yc_obs"][idx0]):
        dx_obs = d["xc_obs"] - d["xc_obs"][idx0]
        dy_obs = d["yc_obs"] - d["yc_obs"][idx0]
        disp_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2)
        disp_pred = np.sqrt(xp_pred ** 2 + yp_pred ** 2)
        ax_cen.plot(t_abs[valid], disp_obs[valid], "g-", lw=2,
                    label=r"$|\Delta\mathbf{r}|_\mathrm{obs}$")
        ax_cen.plot(t_abs[valid], disp_pred[valid], "c--", lw=1.5,
                    label=r"$|\int\!\!(a_x,a_y)\,dt|$")
    ax_cen.axhline(0, color="k", lw=0.4)
    ax_cen.set_xlabel("simulation hour")
    ax_cen.set_ylabel("total displacement [°]")
    ax_cen.set_title("centroid travel distance (obs vs ∫|aₓ,a_y| dt)")
    ax_cen.legend(fontsize=8, loc="best"); ax_cen.grid(alpha=0.3)
    ax_cen.set_xlim(left=float(start_hour_abs))

    # tilt
    ax_til.plot(t_abs[valid], dth_obs[valid], "g-",  lw=2,
                label=r"$\sum\Delta\theta_\mathrm{obs}$")
    ax_til.plot(t_abs[valid], dth_pred[valid], "c--", lw=1.5,
                label=r"$\sum\Delta\theta_\mathrm{pred}$")
    ax_til.axhline(0, color="k", lw=0.4)
    ax_til.set_xlabel("simulation hour"); ax_til.set_ylabel("tilt change [°]")
    ax_til.set_title("∑Δθ obs vs def-basis pred")
    ax_til.legend(fontsize=8); ax_til.grid(alpha=0.3)
    ax_til.set_xlim(left=float(start_hour_abs))

    # red "current time" marker on each row-3 panel: a vertical line at
    # the current absolute hour plus a dot at the obs curve value.
    now_v_lines = [ax.axvline(t_abs[idx0], color="red", lw=1.4, alpha=0.85)
                   for ax in (ax_area, ax_cen, ax_til)]
    now_dot_area, = ax_area.plot([], [], "ro", ms=6, zorder=5)
    now_dot_cen,  = ax_cen.plot([],  [], "ro", ms=6, zorder=5)
    now_dot_til,  = ax_til.plot([],  [], "ro", ms=6, zorder=5)
    # cache obs curves keyed by axis for the per-frame update
    now_curves = {
        id(ax_area): (now_dot_area, A_obs_disp),
        id(ax_cen):  (now_dot_cen,  disp_obs),
        id(ax_til):  (now_dot_til,  dth_obs),
    }

    # ── per-frame draw ──
    def _frame(i):
        for r in range(3):
            for c in range(3):
                axes[r][c].clear()
        # row 0
        a00, a01, a02 = axes[0]
        a00.pcolormesh(X, Y, d["qa"][i], cmap="RdBu_r",
                       vmin=-vm_qa, vmax=vm_qa, shading="auto")
        a00.set_title(r"$\zeta_{250}$ (q')")
        a01.pcolormesh(X, Y, d["q"][i], cmap="RdBu_r",
                       vmin=-vm_q, vmax=vm_q, shading="auto")
        a01.contour(X, Y, d["mask_frames"][i].astype(float),
                    levels=[0.5], colors="k", linewidths=1.0,
                    linestyles="--")
        if np.isfinite(d["theta_obs"][i]):
            e_mid = Ellipse((d["xc_obs"][i], d["yc_obs"][i]),
                            2 * d["a_obs"][i], 2 * d["b_obs"][i],
                            angle=d["theta_obs"][i],
                            fill=False, edgecolor="green", lw=1.6)
            a01.add_patch(e_mid)
            xs_m, ys_m = _ellipse_axis_endpoints(
                d["xc_obs"][i], d["yc_obs"][i],
                d["a_obs"][i], d["theta_obs"][i],
            )
            a01.plot(xs_m, ys_m, "g-", lw=1.2)
        a01.set_title("total PV")
        a02.pcolormesh(X, Y, d["qa"][i], cmap="RdBu_r",
                       vmin=-vm_qa, vmax=vm_qa, shading="auto")
        a02.contour(X, Y, d["mask_frames"][i].astype(float),
                    levels=[0.5], colors="k", linewidths=1.0,
                    linestyles="--")
        if np.isfinite(d["theta_obs"][i]):
            e = Ellipse((d["xc_obs"][i], d["yc_obs"][i]),
                        2 * d["a_obs"][i], 2 * d["b_obs"][i],
                        angle=d["theta_obs"][i],
                        fill=False, edgecolor="green", lw=1.8)
            a02.add_patch(e)
            xs, ys = _ellipse_axis_endpoints(
                d["xc_obs"][i], d["yc_obs"][i],
                d["a_obs"][i], d["theta_obs"][i],
            )
            a02.plot(xs, ys, "g-", lw=1.4)
        a02.set_title(r"$q'$ + mask + ellipse")

        # row 1
        a10, a11, a12 = axes[1]
        a10.pcolormesh(X, Y, d["pv_dt"][i], cmap="RdBu_r",
                       vmin=-vm_rec, vmax=vm_rec, shading="auto")
        a10.set_title(r"real $\partial q/\partial t$")
        if np.isfinite(d["recon"][i]).any():
            a11.pcolormesh(X, Y, d["recon"][i], cmap="RdBu_r",
                           vmin=-vm_rec, vmax=vm_rec, shading="auto")
        a11.set_title("reconstruction")
        if np.isfinite(d["resid"][i]).any():
            a12.pcolormesh(X, Y, d["resid"][i], cmap="RdBu_r",
                           vmin=-vm_rec, vmax=vm_rec, shading="auto")
        a12.set_title("residual")
        for ax in (a10, a11, a12):
            ax.contour(X, Y, d["mask_frames"][i].astype(float),
                       levels=[0.5], colors="k", linewidths=0.7,
                       linestyles="--")

        # row 2 — two-line titles: equation on line 1, diagnosed value+units
        # on line 2. β has units of s⁻¹; aₓ, a_y are m/s; γ₁, γ₂ are m².
        a20, a21, a22 = axes[2]
        if np.isfinite(d["inten"][i]).any():
            a20.pcolormesh(X, Y, d["inten"][i], cmap="RdBu_r",
                           vmin=-vm_int, vmax=vm_int, shading="auto")
        a20.set_title(rf"$\beta\,\Phi_1$" "\n"
                      rf"$\beta={d['beta'][i]:+.2e}\ \mathrm{{s^{{-1}}}}$",
                      fontsize=10)
        if np.isfinite(d["prop"][i]).any():
            a21.pcolormesh(X, Y, d["prop"][i], cmap="RdBu_r",
                           vmin=-vm_prp, vmax=vm_prp, shading="auto")
        a21.set_title(rf"$-a_x\Phi_2-a_y\Phi_3$" "\n"
                      rf"$a_x={d['ax'][i]:+.2f}$, $a_y={d['ay'][i]:+.2f}\ \mathrm{{m/s}}$",
                      fontsize=10)
        if np.isfinite(d["defm"][i]).any():
            a22.pcolormesh(X, Y, d["defm"][i], cmap="RdBu_r",
                           vmin=-vm_def, vmax=vm_def, shading="auto")
        a22.set_title(rf"$-\gamma_1\Phi_4-\gamma_2\Phi_5$" "\n"
                      rf"$\gamma_1={d['g1'][i]/1e6:+.2f}$, "
                      rf"$\gamma_2={d['g2'][i]/1e6:+.2f}\ \mathrm{{km^2/s}}$",
                      fontsize=10)
        # mask + ellipse on row-2 panels too
        for ax in (a20, a21, a22):
            ax.contour(X, Y, d["mask_frames"][i].astype(float),
                       levels=[0.5], colors="k", linewidths=0.7,
                       linestyles="--")
            if np.isfinite(d["theta_obs"][i]):
                e2 = Ellipse((d["xc_obs"][i], d["yc_obs"][i]),
                             2 * d["a_obs"][i], 2 * d["b_obs"][i],
                             angle=d["theta_obs"][i],
                             fill=False, edgecolor="green", lw=1.4)
                ax.add_patch(e2)
                xs2, ys2 = _ellipse_axis_endpoints(
                    d["xc_obs"][i], d["yc_obs"][i],
                    d["a_obs"][i], d["theta_obs"][i],
                )
                ax.plot(xs2, ys2, "g-", lw=1.0)

        # axes cosmetics for fields
        for r in range(3):
            for ax in axes[r]:
                ax.set_xlim(-CFG.PATCH_HALF_LON, CFG.PATCH_HALF_LON)
                ax.set_ylim(-CFG.PATCH_HALF_LAT, CFG.PATCH_HALF_LAT)
                ax.axhline(0, color="k", lw=0.3, alpha=0.4)
                ax.axvline(0, color="k", lw=0.3, alpha=0.4)

        # row 3 is static; update only the red "now" marker
        for ln in now_v_lines:
            ln.set_xdata([t_abs[i], t_abs[i]])
        for ax_obj, (dot, curve) in (
            (ax_area, now_curves[id(ax_area)]),
            (ax_cen,  now_curves[id(ax_cen)]),
            (ax_til,  now_curves[id(ax_til)]),
        ):
            yv = curve[i] if (0 <= i < len(curve) and np.isfinite(curve[i])) else np.nan
            if np.isfinite(yv):
                dot.set_data([t_abs[i]], [yv])
            else:
                dot.set_data([], [])

        fig.suptitle(
            f"{lc.upper()} {method}/{polarity}  "
            f"hour {t_abs[i]:.0f}  (day {t_abs[i] / 24:.2f})  "
            f"frame {i - idx0 + 1}/{len(valid)}",
            fontsize=12,
        )
        return []

    print(f"  rendering {len(valid)} frames ...", flush=True)
    EGU.mkdir(parents=True, exist_ok=True)
    out_mp4 = EGU / f"{lc}_egu_{polarity}.mp4"
    anim = FuncAnimation(fig, _frame, frames=valid, blit=False,
                         interval=1000 / CFG.ANIM_FPS)
    writer = FFMpegWriter(fps=CFG.ANIM_FPS, bitrate=2400)
    anim.save(out_mp4, writer=writer, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_mp4}", flush=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lc", choices=["lc1", "lc2"])
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    ap.add_argument("--start_hour", type=float, required=True,
                    help="Absolute simulation hour to start the animation "
                         "and time-series predictions (e.g. 150 for lc1, "
                         "180 for lc2).")
    ap.add_argument("--method", default="zeta250_back")
    args = ap.parse_args()
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for pol in pols:
        make_animation(args.lc, pol, args.start_hour, args.method)
