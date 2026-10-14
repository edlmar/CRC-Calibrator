"""Figures (matplotlib, PNG + SVG) and a self-contained HTML report for a Result."""
import html
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, RULE = "#1E2A36", "#5D6B78", "#D5DCE2"
PROMISE, OBSERVED, INFLATED = "#9DB4C6", "#E4572E", "#1E2A36"
HB = "#4F7396"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": RULE, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#EEF1F4", "grid.linewidth": 0.8,
                     "legend.frameon": False, "svg.fonttype": "none"})


def _pct(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def fig_curve(result, alpha=None):
    """Mean share missed vs threshold, CRC-inflated curve, budget line and chosen threshold."""
    c = result.curve
    th = result.thresholds
    crc_rows = th[(th.key == "crc") & th.achievable]
    if alpha is None and len(crc_rows):
        alpha = 0.10 if (crc_rows.alpha == 0.10).any() else crc_rows.alpha.iloc[len(crc_rows) // 2]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    top = c.threshold[c.share_missed <= max(0.35, (alpha or 0) * 3)].max()
    sel = c.threshold <= top
    ax.plot(c.threshold[sel], 100 * c.share_missed[sel], color=OBSERVED, lw=2.4, label="Mean share of objects missed")
    n = result.summary["cases"]
    ax.plot(c.threshold[sel], 100 * c.crc_inflated[sel], color=INFLATED, lw=1.4, ls=(0, (5, 3)),
            label=f"CRC-inflated: {n}/{n + 1}·R(λ) + 1/{n + 1}")
    if "hb" in result.settings.methods:
        ax.plot(c.threshold[sel], 100 * c.hb_ucb[sel], color=HB, lw=1.3, ls=":",
                label=f"RCPS-HB upper bound (δ = {result.settings.delta:g})")
    if alpha is not None:
        ax.axhline(100 * alpha, color=MUTED, lw=1, ls=":", label=f"Budget α = {_pct(alpha, 0)}")
        row = crc_rows[np.isclose(crc_rows.alpha, alpha)]
        if len(row):
            lam = row.threshold.iloc[0]
            ax.plot([lam, lam], [0, 100 * alpha], color=INK, lw=1.1)
            ax.annotate(f"λ = {lam:.3f}", (lam, 0), xytext=(6, 6), textcoords="offset points", color=INK,
                        fontweight="bold")
    ax.set_xlim(0, top); ax.set_ylim(0, 100 * c.crc_inflated[sel].max() * 1.05)   # HB bound may run off the top
    ax.set_xlabel("Threshold λ"); ax.set_ylabel("Objects missed, mean share per case (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_tradeoff(result):
    """Price: false detections per case against mean share missed, with the CRC thresholds marked."""
    c = result.curve
    th = result.thresholds
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    amax = th.alpha.max() if len(th) else 0.1
    sel = (c.threshold > 0) & (c.share_missed <= max(0.35, 3 * amax))
    ax.plot(100 * c.share_missed[sel], c.false_per_case[sel], color=INK, lw=1.6)
    for _, r in th[(th.key == "crc") & th.achievable].iterrows():
        ax.plot(100 * r.share_missed, r.false_per_case, "o", color=OBSERVED, ms=7, zorder=3,
                label=f"CRC α = {_pct(r.alpha, 0)}: λ = {r.threshold:.3f}, {r.false_per_case:.1f} false / case")
        ax.annotate(f"α = {_pct(r.alpha, 0)}", (100 * r.share_missed, r.false_per_case), xytext=(7, 5),
                    textcoords="offset points", color=INK, fontsize=9)
    if len(th[(th.key == "crc") & th.achievable]):
        ax.text(0.98, 0.96, "\n".join(ax.get_legend_handles_labels()[1]), transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=INK, linespacing=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Objects missed, mean share per case (%)"); ax.set_ylabel("False detections per case (log)")
    fig.tight_layout()
    return fig


def fig_validation(result, method="crc"):
    v = result.validation
    v = v[(v.key == method) & (v.achievable > 0)]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(v))
    ax.bar(x - 0.2, 100 * v.alpha, 0.38, color=PROMISE, label="Promised: mean share missed ≤ α")
    ax.bar(x + 0.2, 100 * v.test_share_missed, 0.38, color=OBSERVED,
           label=f"Observed on test cases (mean of {int(v.splits.iloc[0]) if len(v) else 0:,} splits)")
    ax.errorbar(x + 0.2, 100 * v.test_share_missed,
                yerr=[100 * (v.test_share_missed - v.test_q05), 100 * (v.test_q95 - v.test_share_missed)],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3, label="5–95% of single splits")
    for xi, (a, r) in enumerate(zip(v.alpha, v.test_share_missed)):
        ax.text(xi - 0.2, 100 * a, _pct(a, 0) if 100 * a == round(100 * a) else _pct(a), ha="center", va="bottom",
                color=INK, fontsize=9)
    for xi, (r, q) in enumerate(zip(v.test_share_missed, v.test_q95)):
        ax.text(xi + 0.2, 100 * q, _pct(r), ha="center", va="bottom", color=OBSERVED, fontsize=9, fontweight="bold")
    ax.set_xticks(x, [f"α = {_pct(a, 0) if 100 * a == round(100 * a) else _pct(a)}" for a in v.alpha])
    ax.set_ylabel("Objects missed, mean share (%)")
    ax.set_ylim(0, 100 * max(v.alpha.max(), v.test_q95.max()) * 1.12 if len(v) else 1)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def save(fig, path_stem):
    path_stem = Path(path_stem)
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".svg"))
    buf = io.StringIO(); fig.savefig(buf, format="svg"); plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def _table(df, cols, fmt):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in cols.values())
    body = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{html.escape(fmt.get(c, str)(r[c]))}</td>" for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def num(d):
    return lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


THRESH_COLS = {"method": "Method", "alpha": "Budget α", "threshold": "Threshold λ", "share_missed": "Share missed",
               "missed_per_case": "Missed / case", "false_per_case": "False / case",
               "cases_above_alpha": "Cases above α"}
VALID_COLS = {"method": "Method", "alpha": "Budget α", "achievable": "Threshold found",
              "test_share_missed": "Test share missed (mean)", "test_q05": "5% split", "test_q95": "95% split",
              "splits_above_alpha": "Splits above α", "test_false_per_case": "False / case"}
FMT = {"alpha": _pct, "threshold": num(3), "share_missed": _pct, "missed_per_case": num(1),
       "false_per_case": num(1), "cases_above_alpha": lambda x: _pct(x, 0), "achievable": lambda x: _pct(x, 0),
       "test_share_missed": _pct, "test_q05": _pct, "test_q95": _pct, "splits_above_alpha": lambda x: _pct(x, 0),
       "test_false_per_case": num(1), "method": str}

CSS = f"""body{{font-family:Segoe UI,Arial,sans-serif;color:{INK};max-width:960px;margin:32px auto;padding:0 20px;
line-height:1.5}}h1{{font-size:24px;margin-bottom:4px}}h2{{font-size:17px;margin-top:32px;border-bottom:1px solid {RULE};
padding-bottom:4px}}.muted{{color:{MUTED}}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{text-align:right;padding:6px 8px;border-bottom:1px solid {RULE}}}th:first-child,td:first-child{{text-align:left}}
th{{color:{MUTED};font-weight:600}}.note{{background:#FFF6E5;border-left:3px solid #E0A23B;padding:8px 12px;margin:6px 0}}
svg{{max-width:100%;height:auto}}code{{background:#F2F4F6;padding:1px 4px}}"""


def write(result, out_dir, title="Conformal risk control report", alpha=None):
    """alpha: CRC budget marked on the curve (default 10% or the middle one).
    thresholds.csv, validation.csv, curve.csv, figures/*.png|svg, settings.json, report.html -> out_dir."""
    out = Path(out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)
    result.thresholds.to_csv(out / "thresholds.csv", index=False)
    result.curve.to_csv(out / "curve.csv", index=False)
    if result.validation is not None:
        result.validation.to_csv(out / "validation.csv", index=False)
    (out / "settings.json").write_text(json.dumps(result.settings_dict() | {"source": result.source}, indent=2))
    svgs = {"curve": save(fig_curve(result, alpha), out / "figures" / "curve"),
            "tradeoff": save(fig_tradeoff(result), out / "figures" / "tradeoff")}
    if result.validation is not None and (result.validation.key == "crc").any():
        svgs["validation"] = save(fig_validation(result), out / "figures" / "validation")
    s, st = result.summary, result.settings
    notes = "".join(f'<div class="note">{html.escape(n)}</div>' for n in result.notes)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             f"<style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><div class='muted'>{datetime.now():%d %B %Y, %H:%M} · "
             f"{html.escape(result.source)}</div>",
             "<h2>Data</h2>",
             f"<p>{s['cases']} cases ({s['groups']} groups), {s['objects']:,} annotated objects "
             f"({s['objects_per_case']:.1f} per case, {s['min_objects']}–{s['max_objects']}). "
             f"Loss per case: share of its objects missed. Cost: false detections per case.</p>", notes,
             "<h2>Thresholds from all cases</h2>",
             "<p class='muted'>CRC: mean share missed on a new case ≤ α. RCPS-HB: with probability ≥ 1 − δ "
             f"(δ = {st.delta:g}) over the calibration cases, the true mean share missed ≤ α. Share missed, "
             "missed / case and false / case are measured on the same cases; only the share missed is guaranteed.</p>",
             _table(result.thresholds, THRESH_COLS, FMT),
             f"<p class='muted'>Smallest budget CRC can promise with these {s['cases']} cases: "
             f"{_pct(s['min_alpha'])} (1/(n+1) = {_pct(1 / (s['cases'] + 1))}, raised by the "
             f"{_pct(s['missed_floor'])} of objects per case that were never detected).</p>",
             "<h2>Mean share missed against the threshold</h2>", svgs["curve"],
             "<h2>The price: false detections</h2>", svgs["tradeoff"]]
    if result.validation is not None:
        parts += ["<h2>Check on random calibration / test splits</h2>",
                  f"<p class='muted'>{st.n_splits:,} random splits by group, {_pct(st.cal_fraction, 0)} of groups "
                  "for calibration. CRC bounds the mean of the test share missed over splits; single splits vary.</p>",
                  _table(result.validation, VALID_COLS, FMT), svgs.get("validation", "")]
    parts += ["<h2>Assumptions</h2><ul>",
              "<li>New cases are exchangeable with the calibration cases: same population, lab, stain, scanner, "
              "annotation protocol. After any shift, recalibrate.</li>",
              "<li>Scores must come from a model that never saw the calibration cases (held-out or "
              "cross-validated predictions).</li>",
              "<li>The guarantee is about the average case, not each case; false detections are measured, not "
              "guaranteed.</li></ul>",
              "<h2>References</h2><ul><li>Angelopoulos AN, Bates S, Fisch A, Lei L, Schuster T. Conformal risk "
              "control. ICLR 2024.</li><li>Bates S, Angelopoulos A, Lei L, Malik J, Jordan M. Distribution-free, "
              "risk-controlling prediction sets. J ACM 2021;68(6):43.</li></ul>",
              f"<h2>Settings</h2><p><code>{html.escape(json.dumps(result.settings_dict()))}</code></p></body></html>"]
    (out / "report.html").write_text("".join(parts), encoding="utf-8")
    return out / "report.html"
