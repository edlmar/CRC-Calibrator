"""From Cases to thresholds with guarantees, their price, and an empirical check.

Loss of case i at threshold lam: share of its annotated objects that are missed: never detected (score 0), or
detected with a score below lam. It is in [0, 1] and non-decreasing in lam, as CRC and RCPS require. Never-detected
objects are missed at every threshold, lam = 0 included, so the loss at lam = 0 is the share no threshold can recover.
Cost: false detections with score >= lam.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import crc

METHOD_NAMES = {"crc": "CRC", "hb": "RCPS-HB", "wsr": "RCPS-WSR"}
PROMISE = {"crc": "mean share missed on new cases ≤ α",
           "hb": "true mean share missed ≤ α with probability ≥ 1 − δ",
           "wsr": "true mean share missed ≤ α with probability ≥ 1 − δ"}


@dataclass
class Settings:
    alphas: list = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    methods: list = field(default_factory=lambda: ["crc", "hb"])
    delta: float = 0.10
    n_splits: int = 2000
    cal_fraction: float = 0.75
    seed: int = 0
    grid_step: float = 0.001

    def __post_init__(self):
        """Reject settings the methods are not defined for, with a message a user can act on."""
        bad = [a for a in self.alphas if not 0 < a < 1]
        if bad:
            raise ValueError(f"budgets α must lie between 0 and 1 (0% and 100%), got {', '.join(map(str, bad))}")
        if not self.alphas:
            raise ValueError("choose at least one budget α")
        unknown = set(self.methods) - set(METHOD_NAMES)
        if unknown or not self.methods:
            raise ValueError(f"methods must be among {', '.join(METHOD_NAMES)}, got {', '.join(self.methods)}")
        if not 0 < self.delta < 1:
            raise ValueError(f"δ must lie between 0 and 1, got {self.delta}")
        if not 0 < self.cal_fraction < 1:
            raise ValueError(f"the calibration share must lie between 0 and 1, got {self.cal_fraction}")
        if self.n_splits < 0:
            raise ValueError(f"the number of check splits cannot be negative, got {self.n_splits}")


def grid_for(cases, step=0.001):
    top = max(1.0, cases.max_score)
    return np.round(np.arange(0, top + step / 2, step), 6)


def curves(cases, grid):
    """L (n, G) share missed, M (n, G) objects missed, F (n, G) false detections, per case and threshold."""
    n_obj = cases.n_objects
    # below lam, or never detected: a score-0 object is not found even at lam = 0, where "score >= lam" holds
    M = np.stack([np.maximum(np.searchsorted(np.sort(o), grid, side="left"), (o <= 0).sum())
                  for o in cases.objects]).astype(float)
    L = np.divide(M, n_obj[:, None], out=np.zeros_like(M), where=n_obj[:, None] > 0)
    F = np.stack([len(f) - np.searchsorted(np.sort(f), grid, side="left") for f in cases.false]).astype(float)
    return L, M, F


def _row(method, alpha, j, grid, L, M, F, n, delta):
    base = dict(method=METHOD_NAMES[method], key=method, alpha=alpha, delta=delta if method != "crc" else np.nan,
                promise=PROMISE[method], n=n)
    if j is None:
        return base | dict(achievable=False, threshold=np.nan, share_missed=np.nan, missed_per_case=np.nan,
                           false_per_case=np.nan, cases_above_alpha=np.nan)
    return base | dict(achievable=True, threshold=float(grid[j]), share_missed=L[:, j].mean(),
                       missed_per_case=M[:, j].mean(), false_per_case=F[:, j].mean(),
                       cases_above_alpha=(L[:, j] > alpha).mean())


def calibrate(cases, s: Settings):
    """Thresholds from all cases (the ones to use), with in-sample price."""
    grid = grid_for(cases, s.grid_step)
    L, M, F = curves(cases, grid)
    rng = np.random.default_rng(s.seed)
    rows = [_row(m, a, crc.select(L, grid, a, m, s.delta, rng), grid, L, M, F, len(cases), s.delta)
            for m in s.methods for a in s.alphas]
    return pd.DataFrame(rows)


def natural_key(x):
    """Numeric ids in numeric order (2 before 10), then the rest alphabetically."""
    s = str(x)
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def split_groups(groups, cal_fraction, rng):
    """Calibration mask over cases; all cases of one group land on the same side. Groups are put in natural order
    before the permutation, so with one case per slide the splits equal those of the original ICC analysis script."""
    uniq = np.array(sorted(pd.unique(np.asarray(groups, dtype=object)), key=natural_key), dtype=object)
    k = min(max(1, int(round(cal_fraction * len(uniq)))), len(uniq) - 1)
    cal = set(uniq[rng.permutation(len(uniq))[:k]].tolist())
    return np.array([g in cal for g in groups])


def validate(cases, s: Settings, progress=None):
    """Repeated random calibration/test splits (by group). Per method and alpha: how often a threshold exists, the
    realised mean share missed on the test cases (its mean is what CRC bounds), and the test cost."""
    grid = grid_for(cases, s.grid_step)
    L, M, F = curves(cases, grid)
    rng = np.random.default_rng(s.seed)
    keys = [(m, a) for m in s.methods for a in s.alphas]
    risk = {k: np.full(s.n_splits, np.nan) for k in keys}
    cost = {k: np.full(s.n_splits, np.nan) for k in keys}
    above = {k: np.full(s.n_splits, np.nan) for k in keys}
    lam = {k: np.full(s.n_splits, np.nan) for k in keys}
    n_cal = np.zeros(s.n_splits)
    for i in range(s.n_splits):
        cal = split_groups(cases.groups, s.cal_fraction, rng)
        test = ~cal
        n_cal[i] = cal.sum()
        order = np.random.default_rng([s.seed, i])
        for m, a in keys:
            j = crc.select(L[cal], grid, a, m, s.delta, order)
            if j is None:
                continue
            lam[m, a][i] = grid[j]
            risk[m, a][i] = L[test, j].mean()
            cost[m, a][i] = F[test, j].mean()
            above[m, a][i] = (L[test, j] > a).mean()
        if progress and (i % 50 == 0 or i == s.n_splits - 1):
            progress((i + 1) / s.n_splits)
    rows = []
    for m, a in keys:
        ok = ~np.isnan(risk[m, a])
        r = risk[m, a][ok]
        rows.append(dict(method=METHOD_NAMES[m], key=m, alpha=a, splits=s.n_splits,
                         n_cal=int(np.median(n_cal)), n_test=len(cases) - int(np.median(n_cal)),
                         achievable=ok.mean(),
                         threshold_median=np.median(lam[m, a][ok]) if ok.any() else np.nan,
                         test_share_missed=r.mean() if ok.any() else np.nan,
                         test_q05=np.quantile(r, .05) if ok.any() else np.nan,
                         test_q95=np.quantile(r, .95) if ok.any() else np.nan,
                         splits_above_alpha=(r > a).mean() if ok.any() else np.nan,
                         test_false_per_case=np.nanmean(cost[m, a]) if ok.any() else np.nan,
                         test_cases_above_alpha=np.nanmean(above[m, a]) if ok.any() else np.nan))
    return pd.DataFrame(rows)


def curve_table(cases, s: Settings, step=None):
    """Mean share missed, CRC-inflated share, RCPS-HB upper bound (at s.delta), objects missed and false detections
    per case, on a (coarser) grid."""
    grid = grid_for(cases, s.grid_step)
    L, M, F = curves(cases, grid)
    n = len(cases)
    df = pd.DataFrame(dict(threshold=grid, share_missed=L.mean(0), crc_inflated=n / (n + 1) * L.mean(0) + 1 / (n + 1),
                           hb_ucb=crc.hb_ucb(L.mean(0), n, s.delta),
                           missed_per_case=M.mean(0), false_per_case=F.mean(0)))
    if step:
        k = max(1, int(round(step / s.grid_step)))
        df = df.iloc[::k].reset_index(drop=True)
    return df


def per_case(cases, threshold):
    grid = np.array([threshold])
    L, M, F = curves(cases, grid)
    return pd.DataFrame(dict(case=cases.names, group=cases.groups, objects=cases.n_objects, missed=M[:, 0].astype(int),
                             share_missed=L[:, 0], false_detections=F[:, 0].astype(int)))


@dataclass
class Result:
    settings: Settings
    summary: dict
    source: str
    notes: list
    thresholds: pd.DataFrame
    validation: pd.DataFrame = None
    curve: pd.DataFrame = None

    def settings_dict(self):
        return asdict(self.settings)


def run_all(cases, s: Settings, validate_splits=True, progress=None):
    th = calibrate(cases, s)
    va = validate(cases, s, progress) if validate_splits and s.n_splits > 0 else None
    return Result(settings=s, summary=cases.summary(), source=cases.source, notes=list(cases.notes),
                  thresholds=th, validation=va, curve=curve_table(cases, s))
