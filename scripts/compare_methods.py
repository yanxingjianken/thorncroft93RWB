"""Pick the best-aligned tracking method per LC from tilt NPZ sidecars.

For each (lc, method, polarity) load projections/<method>/data/tilt_<pol>.npz
and compute a cumulative mod-180 alignment score:

    score_{C,AC} = sum_i |  _wrap(θ_obs[i+1]-θ_obs[i])
                           - _wrap(θ_pred[i]-θ_obs[i])  |

(over finite samples only). The method with the SMALLEST combined
score = score_C + score_AC wins for that LC.

Writes:
  outputs/<lc>/projections/winner.json
  outputs/<lc>/projections/best -> projections/<winner_method>/plots
  (symlink; replaced atomically)
"""
from __future__ import annotations
import sys
import json
import math
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _config as CFG  # noqa

ROOT = Path("/net/flood/data2/users/x_yan/literature_review/rwb/thorncroft93_baroclinic/"
            "thorncroft_rwb/outputs")


def _wrap(d: float) -> float:
    """Wrap angle diff to (-90, 90]."""
    return (d + 90.0) % 180.0 - 90.0


def _score(npz_path: Path) -> tuple[float, int]:
    if not npz_path.exists():
        return math.inf, 0
    d = np.load(npz_path)
    theta_obs = d["theta_obs"]
    theta_pred = d["theta_pred"]
    s = 0.0; n = 0
    for i in range(len(theta_obs) - 1):
        if (np.isfinite(theta_obs[i]) and np.isfinite(theta_obs[i + 1])
                and np.isfinite(theta_pred[i])):
            d_obs = _wrap(theta_obs[i + 1] - theta_obs[i])
            d_pred = _wrap(theta_pred[i] - theta_obs[i])
            s += abs(_wrap(d_obs - d_pred))
            n += 1
    return (s / n if n else math.inf), n


def pick_winner(lc: str, force: str | None = None) -> dict:
    proj_dir = ROOT / lc / "projections"
    scores = {}
    for m in CFG.METHODS:
        sC, nC = _score(proj_dir / m / "data" / "tilt_C.npz")
        sAC, nAC = _score(proj_dir / m / "data" / "tilt_AC.npz")
        total = (sC if math.isfinite(sC) else 0) + \
                (sAC if math.isfinite(sAC) else 0)
        if not (math.isfinite(sC) or math.isfinite(sAC)):
            total = math.inf
        scores[m] = {
            "score_C_mean_abs_deg": sC,
            "score_AC_mean_abs_deg": sAC,
            "n_C": nC, "n_AC": nAC,
            "score_total": total,
        }
    auto_winner = min(scores, key=lambda m: scores[m]["score_total"])
    if force and force in scores:
        winner = force
        forced = True
    else:
        winner = auto_winner
        forced = False
    payload = {
        "lc": lc,
        "winner": winner,
        "auto_winner": auto_winner,
        "forced": forced,
        "methods": scores,
        "metric": "mean |Δθ_obs - Δθ_pred| per step, wrapped to (-90,90]",
    }
    winner_file = proj_dir / "winner.json"
    winner_file.write_text(json.dumps(payload, indent=2))

    tag = " (forced)" if forced else ""
    # NOTE: we deliberately do NOT symlink projections/best any more —
    # all 3 methods' outputs should be kept visible side-by-side for
    # manual comparison. The winner.json file is purely informational.
    print(f"[{lc}] winner={winner}{tag}  scores="
          f"{ {m: round(scores[m]['score_total'], 3) for m in scores} }")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lcs", nargs="*", default=["lc1", "lc2"])
    ap.add_argument("--force", default=None,
                    help="Force a winner method (e.g. zeta250) regardless "
                         "of comparator scores.")
    args = ap.parse_args()
    for lc in args.lcs:
        pick_winner(lc, force=args.force)
