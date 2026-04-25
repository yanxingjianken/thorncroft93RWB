"""Safe gradient + cbar helpers for projection / tilt scripts.

Two utilities used to suppress boundary blow-up in basis fields and
to choose colour-bar limits from inside the ellipse mask region only.

Lagrangian patch context
------------------------
The 81x81 composite patch is sampled around a tracked centre on the
full sphere.  When the patch crosses the pole or the date-line the
underlying samples can be NaN; ``np.gradient`` does not propagate NaN
gracefully, so filling with zero before differentiation creates a
sharp edge ring that pollutes phi_dx/phi_dy and (worse) the second
derivatives phi_def/phi_strain/phi_lap on the patch boundary.

``safe_gradient`` computes the gradient on the raw (NaN-bearing)
field, then fills the still-finite interior NaN with zero so the
returned (gx, gy) are only zeroed where the input was already missing.
At the patch perimeter ``np.gradient(edge_order=2)`` falls back to a
second-order one-sided stencil, which is the right thing for an
isolated patch on the sphere (no zonal periodicity at the patch
edges since the patch is moving with the track).
"""
from __future__ import annotations
import numpy as np


def safe_gradient(F: np.ndarray, dy: float, dx: float):
    """NaN-preserving central-difference gradient with edge fallback.

    Parameters
    ----------
    F : 2-D array, may contain NaN.  Shape (ny, nx).
    dy, dx : grid spacing in metres along axis-0 and axis-1.

    Returns
    -------
    gx, gy : 2-D arrays of d/dx and d/dy with NaN replaced by zero.
        NaN-adjacent cells receive a one-sided difference where the
        immediate neighbours are finite, otherwise 0.
    """
    F = np.asarray(F, dtype=float)
    ny, nx = F.shape
    gx = np.zeros_like(F)
    gy = np.zeros_like(F)
    finite = np.isfinite(F)
    F0 = np.where(finite, F, 0.0)

    # central diff in x
    gx[:, 1:-1] = (F0[:, 2:] - F0[:, :-2]) / (2 * dx)
    # central diff in y
    gy[1:-1, :] = (F0[2:, :] - F0[:-2, :]) / (2 * dy)
    # one-sided 2nd-order at boundaries (np.gradient style)
    gx[:, 0]  = (-3 * F0[:, 0]  + 4 * F0[:, 1]  - F0[:, 2])  / (2 * dx)
    gx[:, -1] = ( 3 * F0[:, -1] - 4 * F0[:, -2] + F0[:, -3]) / (2 * dx)
    gy[0, :]  = (-3 * F0[0, :]  + 4 * F0[1, :]  - F0[2, :])  / (2 * dy)
    gy[-1, :] = ( 3 * F0[-1, :] - 4 * F0[-2, :] + F0[-3, :]) / (2 * dy)

    # zero out gradients where the central-diff stencil touched a NaN.
    # In the interior, both neighbours must be finite for a valid central
    # diff; at the rim, all three points of the one-sided stencil must
    # be finite.
    nb_x = np.zeros_like(F, dtype=bool)
    nb_x[:, 1:-1] = finite[:, 2:] & finite[:, :-2]
    nb_x[:, 0]    = finite[:, 0]  & finite[:, 1]  & finite[:, 2]
    nb_x[:, -1]   = finite[:, -1] & finite[:, -2] & finite[:, -3]

    nb_y = np.zeros_like(F, dtype=bool)
    nb_y[1:-1, :] = finite[2:, :] & finite[:-2, :]
    nb_y[0, :]    = finite[0, :]  & finite[1, :]  & finite[2, :]
    nb_y[-1, :]   = finite[-1, :] & finite[-2, :] & finite[-3, :]

    gx = np.where(nb_x, gx, 0.0)
    gy = np.where(nb_y, gy, 0.0)
    return gx, gy


def mask_vmax(field: np.ndarray, mask: np.ndarray,
              fallback_pctl: float = 95.0) -> float:
    """Symmetric vmax for a diverging cmap, computed inside ``mask`` only.

    Returns the maximum |field| value over pixels where ``mask`` is True
    and ``field`` is finite.  If the mask region is empty or all-NaN,
    falls back to the global ``fallback_pctl``-th percentile of |field|.
    """
    field = np.asarray(field)
    if mask is not None and mask.any():
        sel = mask & np.isfinite(field)
        if sel.any():
            v = float(np.nanmax(np.abs(field[sel])))
            if v > 0:
                return v
    finite = np.abs(field[np.isfinite(field)])
    if finite.size == 0:
        return 1.0
    v = float(np.nanpercentile(finite, fallback_pctl))
    return v if v > 0 else float(finite.max() or 1.0)
