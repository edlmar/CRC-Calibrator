"""Distribution-free threshold selection: conformal risk control (CRC, Angelopoulos et al., ICLR 2024) and
risk-controlling prediction sets (RCPS, Bates et al., J ACM 2021) with Hoeffding-Bentkus (HB) or
Waudby-Smith-Ramdas (WSR) bounds.

Taken unchanged from crc_app/core/crc.py (the bone-metastasis segmentation project), where it is covered by tests.

Threshold convention: an object counts as found when the score of its matched detection is >= lam. Losses are per
case, in [0, 1] (B = 1), non-decreasing in lam. CRC takes the strictest grid threshold with
n/(n+1) * Rhat_n(lam) + B/(n+1) <= alpha and gives E[loss_test(lam_hat)] <= alpha for exchangeable calibration and
test cases; nothing qualifies when alpha < B/(n+1). RCPS certifies P(R(lam_hat) <= alpha) >= 1 - delta over the
calibration draw; at Rhat = 0 HB needs n >= log(1/delta) / log(1/(1 - alpha)).
"""
import numpy as np
from scipy.stats import binom

B = 1.0
METHODS = ("crc", "hb", "wsr")


def lambda_hat(loss_cal, grid, alpha):
    """Index into grid of the CRC threshold, or None when alpha < B/(n+1).
    loss_cal: (n_cal, n_grid), each row non-decreasing in the grid (grid ascending)."""
    n = loss_cal.shape[0]
    adjusted = n / (n + 1) * loss_cal.mean(axis=0) + B / (n + 1)
    for j in range(len(grid) - 1, -1, -1):  # strict -> lenient: first hit is the strictest valid threshold
        if adjusted[j] <= alpha:
            return j
    return None


def hb_pvalue(rhat, n, alpha):
    """Hoeffding-Bentkus p-value for H0: risk > alpha from the mean of n losses in [0, 1] (Bates et al. 2021)."""
    a = np.minimum(rhat, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = np.where(a > 0, a * np.log(a / alpha), 0.0) + (1 - a) * np.log((1 - a) / (1 - alpha))
    return np.minimum(np.exp(-n * h1), np.e * binom.cdf(np.ceil(n * rhat - 1e-12), n, alpha))


def hb_ucb(rhat, n, delta, iters=50):
    """Upper confidence bound on the risk: the smallest alpha with hb_pvalue(rhat, n, alpha) <= delta, by bisection
    (the p-value falls as alpha rises). RCPS-HB certifies lam exactly when hb_ucb(lam) <= alpha, so this is the curve
    the RCPS-HB threshold is read from, as the inflated curve is for CRC."""
    lo = np.asarray(rhat, float).copy()
    hi = np.full_like(lo, 1 - 1e-12)
    for _ in range(iters):
        mid = (lo + hi) / 2
        rejects = hb_pvalue(rhat, n, mid) <= delta
        hi = np.where(rejects, mid, hi)
        lo = np.where(rejects, lo, mid)
    return hi


def wsr_rejects(losses, alpha, delta):
    """Waudby-Smith-Ramdas betting test of H0: risk >= alpha for every column of losses (n, G), taken in row order
    (shuffle first). Predictable bets as in the RCPS reference code; True where H0 is rejected at level delta."""
    n = losses.shape[0]
    t = np.arange(1, n + 1)[:, None]
    mu = (np.cumsum(losses, axis=0) + 0.5) / (t + 1)
    var = (np.cumsum((losses - mu) ** 2, axis=0) + 0.25) / (t + 1)
    var = np.vstack([np.full((1, losses.shape[1]), 0.25), var[:-1]])   # bet i only sees losses 1..i-1
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / (n * var)), 1.0)
    return np.cumsum(np.log1p(-nu * (losses - alpha)), axis=0).max(axis=0) >= np.log(1 / delta)


def select(loss_cal, grid, alpha, method="crc", delta=0.1, rng=None):
    """Grid index of the chosen threshold, or None. RCPS certifies thresholds from the lenient end (lam = 0) up
    and stops at the first failure: the loss is monotone, so the certified set is a prefix and needs no
    multiplicity correction."""
    if method == "crc":
        return lambda_hat(loss_cal, grid, alpha)
    n = loss_cal.shape[0]
    if method == "hb":
        ok = hb_pvalue(loss_cal.mean(axis=0), n, alpha) <= delta
    elif method == "wsr":
        ok = wsr_rejects(loss_cal[rng.permutation(n)] if rng is not None else loss_cal, alpha, delta)
    else:
        raise ValueError(f"unknown method {method!r}")
    bad = np.flatnonzero(~ok)
    j = (bad[0] if bad.size else len(grid)) - 1
    return j if j >= 0 else None


def min_alpha(n):
    return B / (n + 1)
