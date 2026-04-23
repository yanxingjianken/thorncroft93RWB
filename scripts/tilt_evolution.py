"""Track-following tilt evolution + 4-panel animation per LC/method/polarity.

For each composite hour i:
  • Build a strict ellipse mask:
        - signed anomaly polarity (C: anom>+thr, AC: anom<-thr)
        - inside guard circle of radius (PATCH_HALF - GUARD_PAD_DEG)
  • Fit a weighted-covariance ellipse over the masked points
        (a,b = √(λ),  θ_obs ∈ [-90,90)).
  • Build basis at frame i (no symmetrization in v2.3) and project pv_dt[i].
  • Predict (a_pred, b_pred, θ_pred) one hour later via β/ax/ay/γ1/γ2.

Writes per (lc, method, polarity):
  outputs/<lc>/projections/<method>/plots/theta_tilt_{C,AC}.png
  outputs/<lc>/projections/<method>/plots/theta_tilt_accum_{C,AC}.png
  outputs/<lc>/projections/<method>/plots/tilt_animation_{C,AC}.mp4
  outputs/<lc>/projections/<method>/data/tilt_{C,AC}.npz
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy import ndimage as ndi
import pvtend

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/barotropic_vorticity_model/"
            "thorncroft_rwb/outputs")
M_PER_DEG = 111_000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fill(F: np.ndarray) -> np.ndarray:
    out = F.copy()
    out[~np.isfinite(out)] = 0.0
    return out


def _strict_mask(anom: np.ndarray, polarity: str, thr: float,
                 X: np.ndarray, Y: np.ndarray,
                 guard_radius: float) -> np.ndarray:
    """Signed-anomaly mask AND inside guard circle, restricted to the
    connected component containing (or closest to) the patch centre.

    The simple signed threshold often lights up several disjoint blobs
    inside the patch — adjacent eddies and trough/ridge pairs are
    frequent along the jet. For tilt fitting we want the *central*
    coherent vortex only. Policy:
      1. signed threshold
      2. mask outside guard circle (radius ``guard_radius``)
      3. label connected components (4-connectivity)
      4. keep the component that contains the patch origin (0, 0); if
         none does, keep the component whose centroid is closest to
         the origin.
    """
    if polarity == "C":
        m = anom > thr
    else:
        m = anom < -thr
    rr = np.sqrt(X * X + Y * Y)
    m = m & (rr <= guard_radius)
    if not m.any():
        return m
    labels, n = ndi.label(m)
    if n <= 1:
        return m
    # find pixel index closest to (0,0) inside the mask
    ny, nx = m.shape
    # assume uniform grid centred on zero
    iy0 = int(np.argmin(np.abs(Y[:, 0])))
    ix0 = int(np.argmin(np.abs(X[0, :])))
    centre_lbl = labels[iy0, ix0]
    if centre_lbl != 0:
        keep = centre_lbl
    else:
        # no component covers origin → take nearest centroid
        centroids = ndi.center_of_mass(m, labels, range(1, n + 1))
        dists = [np.hypot(cy - iy0, cx - ix0) for cy, cx in centroids]
        keep = int(np.argmin(dists)) + 1
    return labels == keep


def _fit_ellipse(mask: np.ndarray, weights: np.ndarray,
                 X: np.ndarray, Y: np.ndarray):
    """Weighted-covariance ellipse over masked points.

    Returns (theta_deg ∈ [-90,90), a, b, xc, yc) or all NaN if too sparse.
    """
    w = np.where(mask, np.abs(weights), 0.0)
    s = float(w.sum())
    if s <= 0 or mask.sum() < 8:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    xc = float((w * X).sum() / s)
    yc = float((w * Y).sum() / s)
    dx = X - xc
    dy = Y - yc
    sxx = float((w * dx * dx).sum() / s)
    syy = float((w * dy * dy).sum() / s)
    sxy = float((w * dx * dy).sum() / s)
    cov = np.array([[sxx, sxy], [sxy, syy]])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    a = 2.0 * np.sqrt(max(vals[0], 0.0))   # semi-major (in deg)
    b = 2.0 * np.sqrt(max(vals[1], 0.0))   # semi-minor
    vx, vy = vecs[0, 0], vecs[1, 0]
    theta = np.degrees(np.arctan2(vy, vx))
    # wrap to [-90, 90)
    while theta >= 90.0:
        theta -= 180.0
    while theta < -90.0:
        theta += 180.0
    return theta, a, b, xc, yc


def _wrap_diff(a: float, b: float) -> float:
    """Signed mod-180 difference a-b in (-90,90]."""
    d = (a - b + 90.0) % 180.0 - 90.0
    return d


def _unwrap_mod180(theta: np.ndarray) -> np.ndarray:
    """Continuity-unwrap a mod-180 angle series so adjacent samples differ
    by at most 90°. NaN values are propagated; the running offset is
    held across NaN gaps so the next finite sample is connected to the
    last finite sample.
    """
    out = np.full_like(theta, np.nan, dtype="float64")
    last = None
    offset = 0.0
    for i, v in enumerate(theta):
        if not np.isfinite(v):
            continue
        if last is None:
            out[i] = v
            last = v
            continue
        cand = v + offset
        # bring within ±90 of the previous unwrapped value
        while cand - last > 90.0:
            cand -= 180.0
            offset -= 180.0
        while cand - last < -90.0:
            cand += 180.0
            offset += 180.0
        out[i] = cand
        last = cand
    return out


def _predict_one_step(qa_now, q_now, X_m, Y_m, x_rel, y_rel,
                      polarity: str, fit_thr: float,
                      guard_radius: float, dt_h: float):
    """Project pv_dt and integrate one hour to get predicted ellipse."""
    pv_dt_local = (q_now * 0)  # placeholder — not used here
    return None  # (computed in main loop where pv_dt is known)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def process(lc: str, method: str, polarity: str = "C"):
    spec = CFG.METHOD[method]
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
    X, Y = np.meshgrid(x_rel, y_rel)

    nt = q.shape[0]
    fit_thr = float(spec["mask_thresh"])
    guard_r = float(CFG.PATCH_HALF - CFG.GUARD_PAD_DEG)
    mask_str = "> 0" if polarity == "C" else "< 0"
    dx_m = CFG.DX * M_PER_DEG * np.cos(np.deg2rad(CFG.CENTER_LAT))
    dy_m = CFG.DX * M_PER_DEG

    # pv_dt centred difference of TOTAL field
    pv_dt = np.full_like(q, np.nan)
    pv_dt[1:-1] = (q[2:] - q[:-2]) / (2 * 3600.0)

    theta_obs = np.full(nt, np.nan)
    a_obs = np.full(nt, np.nan); b_obs = np.full(nt, np.nan)
    xc_obs = np.full(nt, np.nan); yc_obs = np.full(nt, np.nan)
    theta_pred = np.full(nt, np.nan)
    a_pred = np.full(nt, np.nan); b_pred = np.full(nt, np.nan)
    xc_pred = np.full(nt, np.nan); yc_pred = np.full(nt, np.nan)
    deform_field = np.full_like(q, np.nan)   # -gamma1*phi4 - gamma2*phi5
    mask_frames = np.zeros_like(q, dtype=bool)  # central-component mask

    for i in range(nt):
        qai = qa[i]
        if not np.isfinite(qai).any():
            continue
        qai = _fill(qai)
        qi = _fill(q[i])
        m = _strict_mask(qai, polarity, fit_thr, X, Y, guard_r)
        if m.sum() < 8:
            continue
        mask_frames[i] = m
        th, a_, b_, xcc, ycc = _fit_ellipse(m, qai, X, Y)
        theta_obs[i] = th; a_obs[i] = a_; b_obs[i] = b_
        xc_obs[i] = xcc; yc_obs[i] = ycc

        if i == 0 or i == nt - 1 or not np.isfinite(pv_dt[i]).any():
            continue
        # Build basis at i and project
        qdy, qdx = np.gradient(qi, dy_m, dx_m, edge_order=2)
        try:
            basis = pvtend.compute_orthogonal_basis(
                qai, qdx, qdy, x_rel, y_rel,
                mask=mask_str,
                apply_smoothing=True, smoothing_deg=CFG.SMOOTHING_DEG,
                grid_spacing=CFG.DX, center_lat=CFG.CENTER_LAT,
                include_lap=CFG.INCLUDE_LAP,
            )
            proj = pvtend.project_field(_fill(pv_dt[i]), basis,
                                        grid_spacing=CFG.DX)
            deform_field[i] = proj["def"]
        except Exception as exc:
            print(f"  [t={int(t_hour[i])}h] basis/proj failed: {exc}")
            continue
        # One-hour prediction via Euler step on q'
        dt_s = CFG.DT_PRED_HOURS * 3600.0
        qa_next = qai + dt_s * proj["recon"]
        m2 = _strict_mask(qa_next, polarity, fit_thr, X, Y, guard_r)
        if m2.sum() < 8:
            continue
        th2, a2, b2, xc2, yc2 = _fit_ellipse(m2, qa_next, X, Y)
        theta_pred[i] = th2; a_pred[i] = a2; b_pred[i] = b2
        xc_pred[i] = xc2; yc_pred[i] = yc2

    # ----- save NPZ sidecar
    data_dir = ROOT / lc / "projections" / method / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    npz = data_dir / f"tilt_{polarity}.npz"
    np.savez(npz,
             t_hour=t_hour, theta_obs=theta_obs, theta_pred=theta_pred,
             a_obs=a_obs, b_obs=b_obs, a_pred=a_pred, b_pred=b_pred,
             xc_obs=xc_obs, yc_obs=yc_obs, xc_pred=xc_pred, yc_pred=yc_pred)

    # ----- tilt time series
    plots = ROOT / lc / "projections" / method / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Continuity-unwrap mod 180 so the time series has no ±180° sign flips.
    theta_obs_cont = _unwrap_mod180(theta_obs)
    theta_pred_cont = _unwrap_mod180(theta_pred)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_hour, theta_obs_cont, "g-", lw=2, label="obs")
    ax.plot(t_hour, theta_pred_cont, "c--", lw=1.5, label="pred (+1 h)")
    ax.set_xlabel("composite hour"); ax.set_ylabel("θ tilt [deg, unwrapped]")
    ax.set_title(f"{lc.upper()} {method}/{polarity}  ellipse tilt")
    ax.axhline(0, color="k", lw=0.4); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / f"theta_tilt_{polarity}.png", dpi=140)
    plt.close(fig)

    # accumulated wrap-aware delta
    dth_obs = np.zeros(nt); dth_pred = np.zeros(nt)
    for i in range(1, nt):
        if np.isfinite(theta_obs[i]) and np.isfinite(theta_obs[i-1]):
            dth_obs[i] = dth_obs[i-1] + _wrap_diff(theta_obs[i],
                                                    theta_obs[i-1])
        else:
            dth_obs[i] = dth_obs[i-1]
        if np.isfinite(theta_pred[i-1]) and np.isfinite(theta_obs[i-1]):
            dth_pred[i] = dth_pred[i-1] + _wrap_diff(theta_pred[i-1],
                                                      theta_obs[i-1])
        else:
            dth_pred[i] = dth_pred[i-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_hour, dth_obs, "g-", lw=2, label="∑Δθ obs")
    ax.plot(t_hour, dth_pred, "c--", lw=1.5, label="∑Δθ pred")
    ax.set_xlabel("composite hour")
    ax.set_ylabel("accumulated tilt change [deg]")
    ax.set_title(f"{lc.upper()} {method}/{polarity}  accumulated tilt")
    ax.axhline(0, color="k", lw=0.4); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / f"theta_tilt_accum_{polarity}.png", dpi=140)
    plt.close(fig)

    # ----- 4-panel animation
    _make_animation(lc, method, polarity, q, qa, pv_dt, deform_field,
                    mask_frames, X, Y, x_rel, y_rel,
                    t_hour, theta_obs, a_obs, b_obs, xc_obs, yc_obs,
                    theta_pred, a_pred, b_pred, xc_pred, yc_pred,
                    fit_thr, guard_r, mask_str, dx_m, dy_m, plots)
    print(f"[{lc}:{method}:{polarity}] wrote tilt outputs in {plots}")


def _ellipse_axis_endpoints(xc, yc, a, theta_deg):
    """Major-axis endpoints (length 2a)."""
    th = np.deg2rad(theta_deg)
    dx = a * np.cos(th); dy = a * np.sin(th)
    return (xc - dx, xc + dx), (yc - dy, yc + dy)


def _make_animation(lc, method, polarity,
                    q, qa, pv_dt, deform_field, mask_frames,
                    X, Y, x_rel, y_rel, t_hour,
                    theta_obs, a_obs, b_obs, xc_obs, yc_obs,
                    theta_pred, a_pred, b_pred, xc_pred, yc_pred,
                    fit_thr, guard_r, mask_str, dx_m, dy_m, plots):
    nt = q.shape[0]
    # Robust colour limits
    pct = CFG.PCTL_CBAR
    vmax_q = float(np.nanpercentile(np.abs(q), pct))
    vmax_qa = float(np.nanpercentile(np.abs(qa), pct))
    vmax_dt = float(np.nanpercentile(np.abs(pv_dt[np.isfinite(pv_dt)]), pct)
                    if np.isfinite(pv_dt).any() else 1.0)
    vmax_def = float(np.nanpercentile(
        np.abs(deform_field[np.isfinite(deform_field)]), pct)
        if np.isfinite(deform_field).any() else vmax_dt)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), constrained_layout=True)
    (ax_ul, ax_ur), (ax_ll, ax_lr) = axes

    # Static colorbars (one per panel) — created from dummy ScalarMappables
    # so they don't get cleared when ax.clear() is called each frame.
    from matplotlib import cm, colors as mcolors
    norm_q   = mcolors.Normalize(-vmax_q,   vmax_q)
    norm_qa  = mcolors.Normalize(-vmax_qa,  vmax_qa)
    norm_dt  = mcolors.Normalize(-vmax_dt,  vmax_dt)
    norm_def = mcolors.Normalize(-vmax_def, vmax_def)
    sm_q   = cm.ScalarMappable(norm=norm_q,   cmap="RdBu_r")
    sm_qa  = cm.ScalarMappable(norm=norm_qa,  cmap="RdBu_r")
    sm_dt  = cm.ScalarMappable(norm=norm_dt,  cmap="RdBu_r")
    sm_def = cm.ScalarMappable(norm=norm_def, cmap="RdBu_r")
    fig.colorbar(sm_q,   ax=ax_ul, shrink=0.82, pad=0.02,
                 label=f"q ({method})")
    fig.colorbar(sm_qa,  ax=ax_ur, shrink=0.82, pad=0.02,
                 label="q'")
    fig.colorbar(sm_dt,  ax=ax_ll, shrink=0.82, pad=0.02,
                 label=r"$\partial q/\partial t$")
    fig.colorbar(sm_def, ax=ax_lr, shrink=0.82, pad=0.02,
                 label=r"$-\gamma_1\phi_4 - \gamma_2\phi_5$")

    def _frame(i):
        for ax in axes.flat:
            ax.clear()
        # UL: total field shading + obs ellipse + axis
        ax_ul.pcolormesh(X, Y, q[i], cmap="RdBu_r",
                         vmin=-vmax_q, vmax=vmax_q, shading="auto")
        ax_ul.set_title(f"q ({method}) t={int(t_hour[i])}h")
        # UR: anomaly shading, total contour
        ax_ur.pcolormesh(X, Y, qa[i], cmap="RdBu_r",
                         vmin=-vmax_qa, vmax=vmax_qa, shading="auto")
        try:
            lvs = np.linspace(np.nanpercentile(q[i], 10),
                              np.nanpercentile(q[i], 90), 7)
            ax_ur.contour(X, Y, q[i], levels=lvs, colors="grey",
                          linewidths=0.6, alpha=0.7)
        except Exception:
            pass
        ax_ur.set_title(f"q' shading + q contour")
        # LL: pv_dt with central-component mask as black dashed contour
        if np.isfinite(pv_dt[i]).any():
            ax_ll.pcolormesh(X, Y, pv_dt[i], cmap="RdBu_r",
                             vmin=-vmax_dt, vmax=vmax_dt, shading="auto")
        try:
            ax_ll.contour(X, Y, mask_frames[i].astype(float),
                          levels=[0.5], colors="k",
                          linewidths=1.2, linestyles="--")
        except Exception:
            pass
        ax_ll.set_title(r"$\partial q/\partial t$ + central-blob mask")
        # LR: deformation tendency  -gamma1*phi4 - gamma2*phi5
        if np.isfinite(deform_field[i]).any():
            ax_lr.pcolormesh(X, Y, deform_field[i], cmap="RdBu_r",
                             vmin=-vmax_def, vmax=vmax_def,
                             shading="auto")
        ax_lr.set_title(r"def: $-\gamma_1\phi_4 - \gamma_2\phi_5$")

        # ellipse + major axis on every panel
        for ax in axes.flat:
            ax.set_aspect("equal")
            ax.set_xlim(-CFG.PATCH_HALF, CFG.PATCH_HALF)
            ax.set_ylim(-CFG.PATCH_HALF, CFG.PATCH_HALF)
            ax.axhline(0, color="k", lw=0.3, alpha=0.4)
            ax.axvline(0, color="k", lw=0.3, alpha=0.4)
            # guard circle
            circ = plt.Circle((0, 0), guard_r, fill=False,
                              color="k", lw=0.4, ls=":")
            ax.add_patch(circ)
            if np.isfinite(theta_obs[i]):
                e = Ellipse((xc_obs[i], yc_obs[i]),
                            2 * a_obs[i], 2 * b_obs[i],
                            angle=theta_obs[i],
                            fill=False, edgecolor="green", lw=2.0)
                ax.add_patch(e)
                xs, ys = _ellipse_axis_endpoints(
                    xc_obs[i], yc_obs[i], a_obs[i], theta_obs[i])
                ax.plot(xs, ys, "g-", lw=1.6)
            if np.isfinite(theta_pred[i]):
                e = Ellipse((xc_pred[i], yc_pred[i]),
                            2 * a_pred[i], 2 * b_pred[i],
                            angle=theta_pred[i],
                            fill=False, edgecolor="cyan", lw=1.4,
                            linestyle="--")
                ax.add_patch(e)
                xs, ys = _ellipse_axis_endpoints(
                    xc_pred[i], yc_pred[i], a_pred[i], theta_pred[i])
                ax.plot(xs, ys, "c--", lw=1.2)
        fig.suptitle(
            f"{lc.upper()} {method}/{polarity}  "
            f"t={int(t_hour[i])}h  "
            f"θ_obs={theta_obs[i]:+.1f}° "
            f"θ_pred={theta_pred[i]:+.1f}°"
            if np.isfinite(theta_obs[i]) else
            f"{lc.upper()} {method}/{polarity}  t={int(t_hour[i])}h",
            fontsize=11)
        return []

    anim = FuncAnimation(fig, _frame, frames=range(nt), blit=False,
                         interval=1000 / CFG.ANIM_FPS)
    out_mp4 = plots / f"tilt_animation_{polarity}.mp4"
    writer = FFMpegWriter(fps=CFG.ANIM_FPS, bitrate=2200)
    anim.save(out_mp4, writer=writer, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--method", default=None)
    ap.add_argument("--polarity", choices=["C", "AC", "both"], default="both")
    args = ap.parse_args()
    methods = [args.method] if args.method else CFG.METHODS
    pols = ["C", "AC"] if args.polarity == "both" else [args.polarity]
    for lc in args.lcs:
        for m in methods:
            for pol in pols:
                process(lc, m, polarity=pol)
