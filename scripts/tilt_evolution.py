"""Tilt evolution diagnostics for the PV Lagrangian composite.

Runs for both polarities:
  C  (cyclonic, positive q')   – mask "> 0", cov-ellipse weights max(+q', 0)
  AC (anticyclonic, negative q')– mask "< 0", cov-ellipse weights max(-q', 0)

Pipeline:
  1. Peak hour t* = argmax_t  sum(max(sgn·q'(t), 0)), sgn=+1 for C, −1 for AC.
  2. Basis computed from (q, q') at t*, with optional 36-rotation
     symmetrization (``CFG.SYMMETRIZE``), ``center_lat = 55 N``.
  3. For every valid hour, project pv_dt onto the basis:
        γ1, γ2, F_DEF, α = ½·atan(γ1/γ2)+90°, A=√(γ1²+γ2²)
     Also compute
        θ_obs  = cov-ellipse angle of (sgn·q')>0 weights,
        q_next = q + F_DEF·Δt_pred   (Δt_pred = 1 h)
        θ_pred = cov-ellipse angle of the same weights of q_next's q'.

Outputs (per LC × polarity):
  projections/plots/theta_tilt_{C,AC}.png
      2-panel: top θ_obs/θ_pred, bottom α + A.
  projections/plots/theta_tilt_accum_{C,AC}.png
      cumulative unwrapped Δθ_obs (green) and cumulative Δθ_pred
      (cyan) from hour 0 to hour N−1.
  projections/plots/tilt_animation_{C,AC}.(mp4|gif)
      2×2 panel animation.  Lower-row colorbars are clipped at the
      ``PCTL_CBAR`` percentile of |pv_dt| and |F_DEF|.  Falls back
      to GIF if ffmpeg is not available.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.ndimage import rotate as nd_rotate
import pvtend

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")
M_PER_DEG = 111_000.0


def _rot_avg(field, n_rot=CFG.N_ROT):
    out = np.zeros_like(field, dtype="float64")
    for k in range(n_rot):
        ang = k * (360.0 / n_rot)
        out += nd_rotate(field, ang, reshape=False, order=1,
                         mode="nearest")
    return out / n_rot


def wrap90(a):
    """Wrap angle(s) in degrees to (-90, 90]."""
    return ((a + 90.0) % 180.0) - 90.0


def cov_ellipse_angle(weights, X, Y):
    """Return (theta_deg in [-90,90], a, b, xc, yc) of ``weights`` mass ellipse.

    ``weights`` should already be clipped non-negative (e.g.
    max(q',0) for C or max(-q',0) for AC).
    """
    w = np.where(weights > 0, weights, 0.0)
    s = w.sum()
    if s <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    xc = (w * X).sum() / s
    yc = (w * Y).sum() / s
    dx = X - xc; dy = Y - yc
    Cxx = (w * dx * dx).sum() / s
    Cyy = (w * dy * dy).sum() / s
    Cxy = (w * dx * dy).sum() / s
    theta = 0.5 * np.degrees(np.arctan2(2.0 * Cxy, Cxx - Cyy))
    tr = Cxx + Cyy
    det = Cxx * Cyy - Cxy * Cxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = tr / 2.0 - np.sqrt(disc)
    a = 2.0 * np.sqrt(max(l1, 0.0))
    b = 2.0 * np.sqrt(max(l2, 0.0))
    return wrap90(theta), a, b, xc, yc


def _unwrap_deg(series):
    """Unwrap a (-90, 90] series to remove ±180° discontinuities."""
    x = np.array(series, dtype="float64")
    mask = np.isfinite(x)
    if mask.sum() < 2:
        return x
    out = x.copy()
    # Treat wrap90 as period 180° by scaling into the standard unwrap.
    idx = np.where(mask)[0]
    vals = np.deg2rad(2.0 * x[idx])
    un = np.unwrap(vals)
    out[idx] = 0.5 * np.rad2deg(un)
    return out


def _save_anim(anim, out_mp4: Path, fps: int):
    """Save with ffmpeg if possible, else fall back to GIF."""
    try:
        if FFMpegWriter.isAvailable():
            writer = FFMpegWriter(fps=fps, bitrate=3200,
                                  codec="libx264",
                                  extra_args=["-pix_fmt", "yuv420p"])
            anim.save(out_mp4, writer=writer)
            return out_mp4
    except Exception as e:
        print(f"  FFMpegWriter failed ({e!r}); falling back to GIF")
    out_gif = out_mp4.with_suffix(".gif")
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    return out_gif


def process(lc: str, polarity: str = "C"):
    ds = xr.open_dataset(ROOT / lc / "composites" /
                         f"{polarity}_composite.nc")
    q = ds["pv_composite"].values.astype("float64")
    theta_pv2 = ds["theta_pv2_composite"].values.astype("float64")
    pv_anom_p = ds["pv_anom_composite"].values.astype("float64")
    t_hour = ds["t"].values.astype(int)
    x_rel = ds["x"].values.astype("float64")
    y_rel = ds["y"].values.astype("float64")
    track_lon = ds["track_lon"].values
    track_lat = ds["track_lat"].values

    nt, ny, nx = q.shape
    X, Y = np.meshgrid(x_rel, y_rel)

    qa = q - np.nanmean(q, axis=-1, keepdims=True)
    sgn = +1.0 if polarity == "C" else -1.0
    mask_str = "> 0" if polarity == "C" else "< 0"

    dx_m = CFG.DX * M_PER_DEG * np.cos(np.deg2rad(CFG.CENTER_LAT))
    dy_m = CFG.DX * M_PER_DEG

    # Centered-difference pv_dt over 1 h (DT_PRED_HOURS·3600)
    DT = 3600.0
    DT_PRED = CFG.DT_PRED_HOURS * 3600.0
    pv_dt = np.full_like(qa, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2 * DT)

    # --- peak-hour basis ---
    mass = np.nansum(np.where(sgn * qa > 0, sgn * qa, 0.0), axis=(1, 2))
    mass[~np.isfinite(mass)] = -np.inf
    mass[0] = mass[-1] = -np.inf
    i_peak = int(np.argmax(mass))
    if CFG.SYMMETRIZE:
        q_peak = _rot_avg(q[i_peak])
        qa_peak = _rot_avg(qa[i_peak])
        basis_note = f"rot-sym ({CFG.N_ROT}×10°)"
    else:
        q_peak = q[i_peak]
        qa_peak = qa[i_peak]
        basis_note = "raw peak-hour"
    qdy_peak, qdx_peak = np.gradient(q_peak, dy_m, dx_m, edge_order=2)
    basis = pvtend.compute_orthogonal_basis(
        qa_peak, qdx_peak, qdy_peak, x_rel, y_rel,
        mask=mask_str,
        apply_smoothing=True, smoothing_deg=CFG.SMOOTHING_DEG,
        grid_spacing=CFG.DX, center_lat=CFG.CENTER_LAT,
        include_lap=CFG.INCLUDE_LAP,
    )

    # diagnostics
    F_DEF = np.full_like(qa, np.nan)
    g1 = np.full(nt, np.nan); g2 = np.full(nt, np.nan)
    A = np.full(nt, np.nan); alpha = np.full(nt, np.nan)
    theta_obs = np.full(nt, np.nan); theta_pred = np.full(nt, np.nan)
    a_obs_arr = np.full(nt, np.nan); b_obs_arr = np.full(nt, np.nan)
    xc_obs_arr = np.full(nt, np.nan); yc_obs_arr = np.full(nt, np.nan)
    a_pred_arr = np.full(nt, np.nan); b_pred_arr = np.full(nt, np.nan)
    q_next_anom = np.full_like(qa, np.nan)

    for i in range(1, nt - 1):
        if np.any(np.isnan(pv_dt[i])):
            continue
        proj = pvtend.project_field(pv_dt[i], basis, grid_spacing=CFG.DX)
        F_DEF[i] = proj["def"]
        g1[i] = proj["gamma1"]; g2[i] = proj["gamma2"]
        A[i] = float(np.hypot(g1[i], g2[i]))
        alpha[i] = wrap90(0.5 * np.degrees(np.arctan2(g1[i], g2[i])) + 90.0)
        # Cov-ellipse weights use polarity-signed anomaly
        w_obs = np.maximum(sgn * qa[i], 0.0)
        th_o, a_o, b_o, xc_o, yc_o = cov_ellipse_angle(w_obs, X, Y)
        theta_obs[i] = th_o
        a_obs_arr[i] = a_o; b_obs_arr[i] = b_o
        xc_obs_arr[i] = xc_o; yc_obs_arr[i] = yc_o
        q_next = q[i] + F_DEF[i] * DT_PRED
        q_next_a = q_next - np.nanmean(q_next, axis=-1, keepdims=True)
        q_next_anom[i] = q_next_a
        w_pred = np.maximum(sgn * q_next_a, 0.0)
        th_p, a_p, b_p, xc_p, yc_p = cov_ellipse_angle(w_pred, X, Y)
        theta_pred[i] = th_p
        a_pred_arr[i] = a_p; b_pred_arr[i] = b_p

    # ---- theta_tilt_{C,AC}.png : 2 panels ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   constrained_layout=True)
    ax1.plot(t_hour, theta_obs, color="green", lw=1.6,
             label=r"$\theta_{\rm obs}$")
    ax1.plot(t_hour, theta_pred, color="cyan", lw=1.6,
             label=r"$\theta_{\rm pred}$  ($q+F_{DEF}\Delta t$)")
    ax1.axhline(0, color="k", lw=0.4, alpha=0.5)
    ax1.set_ylabel("orientation [deg, (-90, 90]]")
    ax1.set_ylim(-90, 90)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"{lc.upper()}  {polarity}  Lagrangian composite "
                  f"— PV-axis tilt evolution  [{basis_note}]")

    ax2.plot(t_hour, alpha, color="magenta", lw=1.6,
             label=r"$\alpha = \frac{1}{2}\arctan(\gamma_1/\gamma_2)+90^\circ$")
    ax2.set_ylim(-90, 90)
    ax2.set_ylabel(r"$\alpha$ [deg]", color="magenta")
    ax2.tick_params(axis="y", labelcolor="magenta")
    ax2.axhline(0, color="k", lw=0.4, alpha=0.5)
    ax2b = ax2.twinx()
    ax2b.plot(t_hour, A, color="black", lw=1.4,
              label=r"$A=\sqrt{\gamma_1^2+\gamma_2^2}$")
    ax2b.set_ylabel(r"$A$ [PV s$^{-1}$ units]")
    ax2.set_xlabel("hour since 2000-01-07 00 UTC (day 6)")
    lns = ax2.get_legend_handles_labels()[0] + \
          ax2b.get_legend_handles_labels()[0]
    lbl = ax2.get_legend_handles_labels()[1] + \
          ax2b.get_legend_handles_labels()[1]
    ax2.legend(lns, lbl, loc="upper right", fontsize=9)

    plots = ROOT / lc / "projections" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    out_png = plots / f"theta_tilt_{polarity}.png"
    fig.savefig(out_png, dpi=140); plt.close(fig)

    # ---- theta_tilt_accum_{C,AC}.png ----
    # Cumulative observed tilt change:  Θ_obs_cum(t) = Σ Δθ_obs[k]
    # Cumulative predicted tilt change: Θ_pred_cum(t) = Σ (θ_pred[k]-θ_obs[k])
    # Wrap-aware increments via unwrapping with 180° period.
    th_o_un = _unwrap_deg(theta_obs)
    d_obs = np.full(nt, np.nan)
    mask_o = np.isfinite(th_o_un)
    idx_o = np.where(mask_o)[0]
    if idx_o.size >= 2:
        d_obs[idx_o[1:]] = np.diff(th_o_un[idx_o])
    d_obs_cum = np.full(nt, np.nan)
    d_obs_cum_vals = np.nancumsum(np.where(np.isfinite(d_obs), d_obs, 0.0))
    d_obs_cum[mask_o] = d_obs_cum_vals[mask_o]

    # Hourly predicted tilt increment: wrap90 difference θ_pred-θ_obs
    delta_pred = wrap90(theta_pred - theta_obs)
    d_pred_cum_vals = np.nancumsum(np.where(np.isfinite(delta_pred),
                                             delta_pred, 0.0))
    mask_p = np.isfinite(delta_pred)
    d_pred_cum = np.full(nt, np.nan)
    d_pred_cum[mask_p] = d_pred_cum_vals[mask_p]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t_hour, d_obs_cum, color="green", lw=1.8,
            label=r"$\sum \Delta\theta_{\rm obs}$ (unwrapped)")
    ax.plot(t_hour, d_pred_cum, color="cyan", lw=1.8,
            label=r"$\sum (\theta_{\rm pred}-\theta_{\rm obs})$  (1h F$_{DEF}$)")
    ax.axhline(0, color="k", lw=0.4, alpha=0.5)
    ax.set_xlabel("hour since 2000-01-07 00 UTC (day 6)")
    ax.set_ylabel("cumulative tilt change [deg]")
    ax.set_title(f"{lc.upper()}  {polarity}  cumulative PV-axis tilt "
                 f"change day 6 → day 13  [{basis_note}]")
    ax.legend(loc="best", fontsize=10)
    out_accum = plots / f"theta_tilt_accum_{polarity}.png"
    fig.savefig(out_accum, dpi=140); plt.close(fig)

    # ---- tilt_animation_{C,AC}.(mp4|gif) ----
    valid = [i for i in range(1, nt - 1) if np.isfinite(theta_obs[i])]
    if not valid:
        print(f"[{lc}:{polarity}] no valid frames; skipping animation")
        return
    # Colorbars clipped at 95th percentile of |·| (both rows)
    pv_dt_flat = np.abs(pv_dt[valid])
    pv_dt_flat = pv_dt_flat[np.isfinite(pv_dt_flat)]
    fdef_flat = np.abs(F_DEF[valid])
    fdef_flat = fdef_flat[np.isfinite(fdef_flat)]
    qa_flat = np.abs(qa[valid])
    qa_flat = qa_flat[np.isfinite(qa_flat)]
    rmax_pvdt = float(np.percentile(pv_dt_flat, CFG.PCTL_CBAR)) \
        if pv_dt_flat.size else 1e-10
    rmax_def = float(np.percentile(fdef_flat, CFG.PCTL_CBAR)) \
        if fdef_flat.size else 1e-10
    qa_abs_max = float(np.percentile(qa_flat, CFG.PCTL_CBAR)) \
        if qa_flat.size else 1e-10
    th_flat = theta_pv2[valid][np.isfinite(theta_pv2[valid])]
    if th_flat.size:
        th_lo, th_hi = np.nanpercentile(th_flat, [5, 95])
    else:
        th_lo, th_hi = 300.0, 340.0

    pv_levels = [-0.5, -0.1, 0.1, 0.5]
    pv_lsty = ["--", "--", "-", "-"]
    pv_contour_txt = (r"contours: $q'_{330K}$ at $\pm 0.1, \pm 0.5$ PVU "
                      r"(solid +, dashed −)")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11),
                             constrained_layout=True)
    axUL, axUR = axes[0]; axL, axR = axes[1]

    imUL = axUL.pcolormesh(X, Y, theta_pv2[valid[0]], cmap="Spectral_r",
                           vmin=th_lo, vmax=th_hi, shading="auto")
    imUR = axUR.pcolormesh(X, Y, qa[valid[0]], cmap="RdBu_r",
                           vmin=-qa_abs_max, vmax=qa_abs_max,
                           shading="auto")
    imL = axL.pcolormesh(X, Y, pv_dt[valid[0]], cmap="RdBu_r",
                         vmin=-rmax_pvdt, vmax=rmax_pvdt, shading="auto")
    imR = axR.pcolormesh(X, Y, F_DEF[valid[0]], cmap="RdBu_r",
                         vmin=-rmax_def, vmax=rmax_def, shading="auto")

    fig.colorbar(imUL, ax=axUL, label=r"$\theta$ on 2 PVU [K]")
    fig.colorbar(imUR, ax=axUR,
                 label=(fr"$q'$ on {CFG.THETA_LEVEL:.0f} K [PVU]  "
                        fr"(clipped at {CFG.PCTL_CBAR:.0f}%ile)"))
    fig.colorbar(imL, ax=axL,
                 label=(r"$\partial q/\partial t$ "
                        fr"(clipped at {CFG.PCTL_CBAR:.0f}%ile)"))
    fig.colorbar(imR, ax=axR,
                 label=(r"$F_{DEF}=-\gamma_1\phi_4-\gamma_2\phi_5$ "
                        fr"(clipped at {CFG.PCTL_CBAR:.0f}%ile)"))

    def _prep(ax):
        ax.set_aspect("equal")
        ax.set_xlabel("x [deg]"); ax.set_ylabel("y [deg]")
        ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        ax.axvline(0, color="k", lw=0.3, alpha=0.4)
        ax.plot(0, 0, marker="+", ms=12, mew=1.6, color="k")

    for ax in (axUL, axUR, axL, axR):
        _prep(ax)

    ell_obs_UR = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                         edgecolor="green", lw=1.8, ls="--")
    ell_pred_UR = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                          edgecolor="cyan", lw=1.8, ls="-")
    ell_pred_R = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                         edgecolor="cyan", lw=1.8, ls="-")
    line_alpha, = axR.plot([], [], color="magenta", lw=1.5)
    axUR.add_patch(ell_pred_UR); axUR.add_patch(ell_obs_UR)
    axR.add_patch(ell_pred_R)

    titUL = axUL.set_title("")
    titUR = axUR.set_title("")
    titL = axL.set_title("")
    titR = axR.set_title("")
    sup = fig.suptitle("")
    ctx_state = {"cUL": None, "cUR": None}

    def update(idx):
        i = valid[idx]
        imUL.set_array(theta_pv2[i].ravel())
        imUR.set_array(qa[i].ravel())
        imL.set_array(pv_dt[i].ravel())
        imR.set_array(F_DEF[i].ravel())
        for key in ("cUL", "cUR"):
            c = ctx_state[key]
            if c is not None:
                try:
                    c.remove()
                except Exception:
                    for coll in getattr(c, "collections", []):
                        try:
                            coll.remove()
                        except Exception:
                            pass
                ctx_state[key] = None
        pa = pv_anom_p[i]
        if np.any(np.isfinite(pa)):
            ctx_state["cUL"] = axUL.contour(
                X, Y, pa, levels=pv_levels, colors="k",
                linewidths=0.8, linestyles=pv_lsty)
            ctx_state["cUR"] = axUR.contour(
                X, Y, pa, levels=pv_levels, colors="k",
                linewidths=0.8, linestyles=pv_lsty)
        a_o = a_obs_arr[i]; b_o = b_obs_arr[i]
        xc_o = xc_obs_arr[i]; yc_o = yc_obs_arr[i]
        ell_obs_UR.set_center((xc_o, yc_o))
        ell_obs_UR.set_width(max(a_o, 1e-3))
        ell_obs_UR.set_height(max(b_o, 1e-3))
        ell_obs_UR.set_angle(theta_obs[i])
        a_p = a_pred_arr[i]; b_p = b_pred_arr[i]
        for e in (ell_pred_UR, ell_pred_R):
            e.set_center((xc_o, yc_o))
            e.set_width(max(a_p, 1e-3)); e.set_height(max(b_p, 1e-3))
            e.set_angle(theta_pred[i])
        Lx = 25.0
        ang = np.deg2rad(alpha[i])
        line_alpha.set_data([-Lx * np.cos(ang), Lx * np.cos(ang)],
                            [-Lx * np.sin(ang), Lx * np.sin(ang)])
        lon_c = float(track_lon[:, i][np.isfinite(track_lon[:, i])].mean()) \
            if np.any(np.isfinite(track_lon[:, i])) else np.nan
        lat_c = float(track_lat[:, i][np.isfinite(track_lat[:, i])].mean()) \
            if np.any(np.isfinite(track_lat[:, i])) else np.nan
        stretch = a_p / max(a_o, 1e-6)
        titUL.set_text(fr"$\theta$ on 2 PVU  t={t_hour[i]}h  "
                       fr"center=({lon_c:.1f}E, {lat_c:.1f}N)"
                       "\n" + pv_contour_txt)
        titUR.set_text(fr"$q'$ on {CFG.THETA_LEVEL:.0f} K  "
                       fr"(green dashed = $\theta_{{obs}}$;  "
                       fr"cyan = $\theta_{{pred}}$, stretched $\pm 1h$)"
                       "\n" + pv_contour_txt)
        titL.set_text(r"$\partial q/\partial t$ (centered diff)")
        titR.set_text(
            f"$F_{{DEF}}$ ({basis_note} basis, polarity={polarity})\n"
            f"$\\alpha$={alpha[i]:+5.1f}  "
            f"$\\theta_{{obs}}$={theta_obs[i]:+5.1f}  "
            f"$\\theta_{{pred}}$={theta_pred[i]:+5.1f}  "
            f"A={A[i]:.2e}  a/b stretch $\\times${stretch:.2f}")
        sup.set_text(f"{lc.upper()}  {polarity}  Lagrangian composite on "
                     fr"$\theta$={CFG.THETA_LEVEL:.0f} K "
                     f"(green=$\\theta_{{obs}}$, "
                     f"cyan=$\\theta_{{pred}}$, magenta=$\\alpha$)")
        return (imUL, imUR, imL, imR,
                ell_obs_UR, ell_pred_UR, ell_pred_R, line_alpha)

    anim = FuncAnimation(fig, update, frames=len(valid),
                         blit=False, interval=1000 // CFG.ANIM_FPS)
    out_mp4 = plots / f"tilt_animation_{polarity}.mp4"
    written = _save_anim(anim, out_mp4, fps=CFG.ANIM_FPS)
    plt.close(fig)
    print(f"[{lc}:{polarity}] wrote {out_png}, {out_accum}, {written}  "
          f"({len(valid)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for pol in pols:
            process(lc, polarity=pol)
