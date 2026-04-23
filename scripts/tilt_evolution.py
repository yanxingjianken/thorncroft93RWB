"""Tilt evolution diagnostics for the cyclone Lagrangian composite.

* Build a SINGLE orthogonal basis from the time-mean q' / q_dx / q_dy
  (mask='> 0', include_lap=False, grid_spacing=1.0, center_lat=50).
* For each hour t in 1..N-2:
      pv_dt = (q[t+1] - q[t-1]) / (2*3600)
      proj  = project_field(pv_dt, basis)
      gamma1, gamma2  -> A = sqrt(g1^2+g2^2)
      alpha = wrap90( 0.5*deg(atan2(gamma1, gamma2)) + 90 )
      theta_obs = cov-ellipse axis of q'(t)>0
      q_next   = q[t] + proj['def'] * 3600          # F_DEF forward-Euler
      q'_next  = q_next - mean_x(q_next)
      theta_pred = cov-ellipse axis of q'_next>0
* Outputs:
    theta_tilt_C.png  – 2-panel:
        top: theta_obs (green) and theta_pred (cyan) vs hour.
        bot: alpha (left axis, magenta) + amplitude A (right axis,
             black, twinx) vs hour.
    tilt_animation_C.gif – one row x two cols per frame:
        left  : pv_dt with green dashed mask ellipse at theta_obs;
                title shows real (lon,lat) of moving center.
        right : F_DEF = -gamma1*phi4 - gamma2*phi5 with green
                theta_obs ellipse, cyan theta_pred ellipse, magenta
                line at orientation alpha; title shows alpha,
                theta_obs, theta_pred, A.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import rotate as nd_rotate
import pvtend

N_ROT = 36  # 360 / 10 deg, matches project_composite.py


def _rot_avg(field):
    out = np.zeros_like(field, dtype="float64")
    for k in range(N_ROT):
        ang = k * (360.0 / N_ROT)
        out += nd_rotate(field, ang, reshape=False, order=1,
                         mode="nearest")
    return out / N_ROT

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")
DX = 1.0
CENTER_LAT = 50.0
M_PER_DEG = 111_000.0
DT = 3600.0  # 1 h
THETA_LEVEL = 330.0  # theta level of the composite (matches build_composites.py)


def wrap90(a):
    return ((a + 90.0) % 180.0) - 90.0


def cov_ellipse_angle(weights, X, Y):
    """Return (theta_deg in [-90,90], a, b, xc, yc) of the q'>0 mass ellipse.

    weights = q' where q'>0, else 0.
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
    # eigvalues for axis lengths
    tr = Cxx + Cyy
    det = Cxx * Cyy - Cxy * Cxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = tr / 2.0 - np.sqrt(disc)
    a = 2.0 * np.sqrt(max(l1, 0.0))
    b = 2.0 * np.sqrt(max(l2, 0.0))
    return wrap90(theta), a, b, xc, yc


def process(lc: str):
    ds = xr.open_dataset(ROOT / lc / "composites" / "C_composite.nc")
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

    dx_m = DX * M_PER_DEG * np.cos(np.deg2rad(CENTER_LAT))
    dy_m = DX * M_PER_DEG
    qdx = np.full_like(qa, np.nan)
    qdy = np.full_like(qa, np.nan)
    for i in range(nt):
        if np.any(np.isnan(qa[i])):
            continue
        qdy[i], qdx[i] = np.gradient(qa[i], dy_m, dx_m, edge_order=2)

    # Rotation-symmetrized peak-hour basis (same as project_composite.py).
    # Peak hour = argmax_t sum(q'(t) > 0).  At that hour we average 36
    # rotations (10 deg step) of q and q' about the patch center, then
    # recompute gradients from the symmetrized q.  The basis therefore
    # has Phi4/Phi5 as a clean quadrupole, identical to decomp_C.png.
    pos_sum = np.nansum(np.where(qa > 0, qa, 0.0), axis=(1, 2))
    pos_sum[~np.isfinite(pos_sum)] = -np.inf
    pos_sum[0] = pos_sum[-1] = -np.inf
    i_peak = int(np.argmax(pos_sum))
    q_sym = _rot_avg(q[i_peak])
    qa_sym = _rot_avg(qa[i_peak])
    qdy_sym, qdx_sym = np.gradient(q_sym, dy_m, dx_m, edge_order=2)
    basis = pvtend.compute_orthogonal_basis(
        qa_sym, qdx_sym, qdy_sym, x_rel, y_rel,
        mask="> 0", apply_smoothing=True, smoothing_deg=3.0,
        grid_spacing=DX, center_lat=CENTER_LAT, include_lap=False,
    )

    # diagnostics arrays
    pv_dt = np.full_like(qa, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2 * DT)

    F_DEF = np.full_like(qa, np.nan)
    g1 = np.full(nt, np.nan); g2 = np.full(nt, np.nan)
    A = np.full(nt, np.nan); alpha = np.full(nt, np.nan)
    theta_obs = np.full(nt, np.nan); theta_pred = np.full(nt, np.nan)
    # Analytical predicted ellipse axes (stretch along alpha, compress
    # across) and predicted center -- used for the cyan ellipse.
    a_obs_arr = np.full(nt, np.nan); b_obs_arr = np.full(nt, np.nan)
    xc_obs_arr = np.full(nt, np.nan); yc_obs_arr = np.full(nt, np.nan)
    a_pred_arr = np.full(nt, np.nan); b_pred_arr = np.full(nt, np.nan)
    q_next_anom = np.full_like(qa, np.nan)

    # Prediction horizon: forward-Euler over DT_PRED so that the
    # integrated compression / extension of the ellipse axes by F_DEF
    # is visually distinguishable from the observed ellipse.
    DT_PRED = 6.0 * DT

    for i in range(1, nt - 1):
        if np.any(np.isnan(pv_dt[i])):
            continue
        proj = pvtend.project_field(pv_dt[i], basis, grid_spacing=DX)
        F_DEF[i] = proj["def"]
        g1[i] = proj["gamma1"]; g2[i] = proj["gamma2"]
        A[i] = float(np.hypot(g1[i], g2[i]))
        # 08-notebook orthog form (90-deg shift relative to slides)
        alpha[i] = wrap90(0.5 * np.degrees(np.arctan2(g1[i], g2[i])) + 90.0)
        th_o, a_o, b_o, xc_o, yc_o = cov_ellipse_angle(qa[i], X, Y)
        theta_obs[i] = th_o
        a_obs_arr[i] = a_o; b_obs_arr[i] = b_o
        xc_obs_arr[i] = xc_o; yc_obs_arr[i] = yc_o
        # Forward-Euler prediction of q' over DT_PRED (6 h).  The full
        # tendency on the 5-basis is ~ F_INT + F_DEF + F_PROP but
        # F_DEF alone drives the ellipse axis stretch/compress, so we
        # integrate q + F_DEF * DT_PRED for the ellipse fit.
        q_next = q[i] + F_DEF[i] * DT_PRED
        q_next_anom_i = q_next - np.nanmean(q_next, axis=-1, keepdims=True)
        q_next_anom[i] = q_next_anom_i
        th_p, a_p, b_p, xc_p, yc_p = cov_ellipse_angle(q_next_anom_i, X, Y)
        theta_pred[i] = th_p
        a_pred_arr[i] = a_p; b_pred_arr[i] = b_p

    # ---- theta_tilt_C.png : 2 panels ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   constrained_layout=True)
    ax1.plot(t_hour, theta_obs, color="green", lw=1.6,
             label=r"$\theta_{\rm obs}$ (cov ellipse of $q'>0$)")
    ax1.plot(t_hour, theta_pred, color="cyan", lw=1.6,
             label=r"$\theta_{\rm pred}$ (after $q+F_{DEF}\Delta t$)")
    ax1.axhline(0, color="k", lw=0.4, alpha=0.5)
    ax1.set_ylabel("orientation [deg, wrapped to (-90, 90]]")
    ax1.set_ylim(-90, 90)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"{lc.upper()}  Lagrangian cyclone composite "
                  f"— PV-axis tilt evolution")

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
    scripts = ROOT / lc / "projections" / "scripts"
    plots.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    out_png = plots / "theta_tilt_C.png"
    fig.savefig(out_png, dpi=140); plt.close(fig)

    # ---- tilt_animation_C.gif  (2 x 2) ----
    valid = [i for i in range(1, nt - 1) if np.isfinite(theta_obs[i])]
    if not valid:
        print(f"[{lc}] no valid frames; skipping gif"); return
    rmax_pvdt = np.nanmax(np.abs(pv_dt[valid]))
    rmax_def = np.nanmax(np.abs(F_DEF[valid]))
    qa_abs_max = np.nanmax(np.abs(qa[valid]))
    th_flat = theta_pv2[valid][np.isfinite(theta_pv2[valid])]
    if th_flat.size:
        th_lo, th_hi = np.nanpercentile(th_flat, [5, 95])
    else:
        th_lo, th_hi = 300.0, 340.0
    # pv_anom contours (replacing the former zeta250 overlay).  Use
    # the fixed tracking threshold on q' (0.2 PVU) plus stronger
    # levels to visualise the streamer envelope.
    pv_levels = [-0.5, -0.2, 0.2, 0.5]
    pv_lsty = ["--", "--", "-", "-"]
    pv_contour_txt = (r"contours: $q'_{330K}$ at $\pm 0.2, \pm 0.5$ PVU "
                      r"(solid +, dashed −)")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11),
                             constrained_layout=True)
    axUL, axUR = axes[0]
    axL, axR = axes[1]

    imUL = axUL.pcolormesh(X, Y, theta_pv2[valid[0]], cmap="Spectral_r",
                           vmin=th_lo, vmax=th_hi, shading="auto")
    imUR = axUR.pcolormesh(X, Y, qa[valid[0]], cmap="RdBu_r",
                           vmin=-qa_abs_max, vmax=qa_abs_max,
                           shading="auto")
    imL = axL.pcolormesh(X, Y, pv_dt[valid[0]], cmap="RdBu_r",
                        vmin=-rmax_pvdt, vmax=rmax_pvdt, shading="auto")
    imR = axR.pcolormesh(X, Y, F_DEF[valid[0]], cmap="RdBu_r",
                        vmin=-rmax_def, vmax=rmax_def, shading="auto")

    cbUL = fig.colorbar(imUL, ax=axUL, label=r"$\theta$ on 2 PVU [K]")
    cbUR = fig.colorbar(imUR, ax=axUR,
                        label=fr"$q'$ on {THETA_LEVEL:.0f} K [PVU]")
    cbL = fig.colorbar(imL, ax=axL, label=r"$\partial q/\partial t$")
    cbR = fig.colorbar(imR, ax=axR,
                       label=r"$F_{DEF}=-\gamma_1\phi_4-\gamma_2\phi_5$")

    def _prep(ax):
        ax.set_aspect("equal")
        ax.set_xlabel("x [deg]"); ax.set_ylabel("y [deg]")
        ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        ax.axvline(0, color="k", lw=0.3, alpha=0.4)
        ax.plot(0, 0, marker="+", ms=12, mew=1.6, color="k")

    for ax in (axUL, axUR, axL, axR):
        _prep(ax)

    # Ellipses move to the UR panel (where the q' fit actually lives).
    # LR keeps the predicted ellipse + alpha line (deformation diagnostic).
    ell_obs_UR = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                         edgecolor="green", lw=1.8, ls="--")
    ell_pred_UR = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                          edgecolor="cyan", lw=1.8, ls="-")
    ell_pred_R = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                         edgecolor="cyan", lw=1.8, ls="-")
    line_alpha, = axR.plot([], [], color="magenta", lw=1.5)
    # UR: draw predicted (cyan solid) first, then observed (green dashed)
    # on top so both ellipses remain visible.
    axUR.add_patch(ell_pred_UR); axUR.add_patch(ell_obs_UR)
    axR.add_patch(ell_pred_R)

    # Titles
    titUL = axUL.set_title("")
    titUR = axUR.set_title("")
    titL = axL.set_title("")
    titR = axR.set_title("")
    sup = fig.suptitle("")

    # Holder for the transient contour artists (re-drawn every frame)
    ctx_state = {"cUL": None, "cUR": None}

    def update(idx):
        i = valid[idx]
        # Arrays only (colorbars are FIXED at init -- no per-frame rescale)
        imUL.set_array(theta_pv2[i].ravel())
        imUR.set_array(qa[i].ravel())
        imL.set_array(pv_dt[i].ravel())
        imR.set_array(F_DEF[i].ravel())
        # Remove previous overlay contours
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
        # Redraw pv_anom contours (replaces the former vor250 overlay)
        pa = pv_anom_p[i]
        if np.any(np.isfinite(pa)):
            ctx_state["cUL"] = axUL.contour(
                X, Y, pa, levels=pv_levels, colors="k",
                linewidths=0.8, linestyles=pv_lsty)
            ctx_state["cUR"] = axUR.contour(
                X, Y, pa, levels=pv_levels, colors="k",
                linewidths=0.8, linestyles=pv_lsty)
        # Observed ellipse from q'(t) -> UR
        a_o = a_obs_arr[i]; b_o = b_obs_arr[i]
        xc_o = xc_obs_arr[i]; yc_o = yc_obs_arr[i]
        ell_obs_UR.set_center((xc_o, yc_o))
        ell_obs_UR.set_width(max(a_o, 1e-3))
        ell_obs_UR.set_height(max(b_o, 1e-3))
        ell_obs_UR.set_angle(theta_obs[i])
        # Cyan predicted ellipse: analytical stretch/compress from F_DEF
        # over DT_PRED horizon, plus the covariance-based rotation angle.
        a_p = a_pred_arr[i]; b_p = b_pred_arr[i]
        for e in (ell_pred_UR, ell_pred_R):
            e.set_center((xc_o, yc_o))
            e.set_width(max(a_p, 1e-3)); e.set_height(max(b_p, 1e-3))
            e.set_angle(theta_pred[i])
        # alpha line
        Lx = 25.0
        ang = np.deg2rad(alpha[i])
        line_alpha.set_data([-Lx * np.cos(ang), Lx * np.cos(ang)],
                            [-Lx * np.sin(ang), Lx * np.sin(ang)])
        # Center (real-world)
        lon_c = float(track_lon[:, i][np.isfinite(track_lon[:, i])].mean()) \
            if np.any(np.isfinite(track_lon[:, i])) else np.nan
        lat_c = float(track_lat[:, i][np.isfinite(track_lat[:, i])].mean()) \
            if np.any(np.isfinite(track_lat[:, i])) else np.nan
        stretch = a_p / max(a_o, 1e-6)
        titUL.set_text(fr"$\theta$ on 2 PVU  t={t_hour[i]}h  "
                       fr"center=({lon_c:.1f}E, {lat_c:.1f}N)"
                       "\n" + pv_contour_txt)
        titUR.set_text(fr"$q'$ on {THETA_LEVEL:.0f} K  "
                       fr"(green dashed = $\theta_{{obs}}$, a,b observed;  "
                       fr"cyan = $\theta_{{pred}}$, a,b stretched $\pm 6h$)"
                       "\n" + pv_contour_txt)
        titL.set_text(r"$\partial q/\partial t$ (centered diff)")
        titR.set_text(
            "$F_{DEF}$ (rot-sym peak-hour basis, quadrupole)\n"
            f"$\\alpha$={alpha[i]:+5.1f}  "
            f"$\\theta_{{obs}}$={theta_obs[i]:+5.1f}  "
            f"$\\theta_{{pred}}$={theta_pred[i]:+5.1f}  "
            f"A={A[i]:.2e}  a/b stretch $\\times${stretch:.2f}")
        sup.set_text(f"{lc.upper()}  cyclone Lagrangian composite on "
                     fr"$\theta$={THETA_LEVEL:.0f} K "
                     f"(green=$\\theta_{{obs}}$, "
                     f"cyan=$\\theta_{{pred}}$, magenta=$\\alpha$)")
        return (imUL, imUR, imL, imR,
                ell_obs_UR, ell_pred_UR, ell_pred_R, line_alpha)

    anim = FuncAnimation(fig, update, frames=len(valid),
                         blit=False, interval=100)
    out_gif = plots / "tilt_animation_C.gif"
    anim.save(out_gif, writer=PillowWriter(fps=8))
    plt.close(fig)
    shutil.copy2(Path(__file__).resolve(), scripts / Path(__file__).name)
    print(f"[{lc}] wrote {out_png} and {out_gif}  ({len(valid)} frames)")


if __name__ == "__main__":
    for lc in sys.argv[1:] or ("lc1", "lc2"):
        process(lc)
