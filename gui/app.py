"""CRC Calibrator: desktop GUI, a native window (NiceGUI + pywebview).

Only calls core. Run: pixi run gui, or double-click "Start CRC Calibrator.cmd".
"""
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "crc_calibrator.log"
if sys.stdout is None:   # started without a console (the desktop shortcut runs pythonw): write a log instead
    sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)

import numpy as np  # noqa: E402
import webview  # noqa: E402
from nicegui import app, run, ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import analysis, data, report  # noqa: E402
from core.analysis import METHOD_NAMES  # noqa: E402
from gui import guide  # noqa: E402

HERE = Path(__file__).resolve().parent
EXAMPLE = ROOT / "examples" / "icc_oof.json"
RESULTS = ROOT / "results"
TITLE = "CRC Calibrator"
INK, MUTED, RULE, PROMISE, OBSERVED = "#1E2A36", "#5D6B78", "#D5DCE2", "#9DB4C6", "#E4572E"
HB_COLOR = "#4F7396"   # the promise blue, dark enough for a thin line
TEXT = {"fontFamily": "Segoe UI, system-ui, sans-serif", "color": MUTED}
PCT_TOOLTIP = {":valueFormatter": "v => v == null ? '–' : v.toFixed(1) + '%'"}   # ':' = evaluated as JavaScript


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def alpha_label(a):
    return f"{100 * a:g}%"


class State:
    def __init__(self):
        self.cases = None
        self.result = None
        self.alpha = None          # selected CRC budget
        self.progress = 0.0
        self.busy = False
        self.loading = False       # reading data (spinner on Load, no progress bar)
        self.error = None
        self.last_export = None
        self.tab = "Curve"         # survives refreshes, so picking a budget keeps the open tab
        self.dropped = []          # paths dropped on the window, picked up by the page's timer


S = State()


# ---------------------------------------------------------------- file dialogs

async def pick_file(target, types=("Data files (*.json;*.csv;*.tsv;*.txt)", "All files (*.*)")):
    kind = getattr(getattr(webview, "FileDialog", None), "OPEN", None) or webview.OPEN_DIALOG
    start = str(Path(target.value).parent) if target.value else str(Path.home())
    res = await app.native.main_window.create_file_dialog(kind, directory=start, allow_multiple=False, file_types=types)
    if res:
        target.value = res[0] if isinstance(res, (list, tuple)) else res


async def pick_files():
    kind = getattr(getattr(webview, "FileDialog", None), "OPEN", None) or webview.OPEN_DIALOG
    res = await app.native.main_window.create_file_dialog(
        kind, directory=str(Path.home()), allow_multiple=True,
        file_types=("Data files (*.json;*.csv;*.tsv;*.txt)", "All files (*.*)"))
    return list(res) if res else []


async def pick_folder():
    kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or webview.FOLDER_DIALOG
    res = await app.native.main_window.create_file_dialog(kind, directory=str(RESULTS if RESULTS.exists() else ROOT))
    return (res[0] if isinstance(res, (list, tuple)) else res) if res else None


def resolve(p):
    """Strip quotes Windows adds on 'Copy as path'; relative paths are tried against the program folder too."""
    p = Path(str(p).strip().strip('"'))
    return str(p if p.is_absolute() or p.exists() else ROOT / p)


def remember(el, key):
    """Keep a field's value between sessions (stored in .nicegui/)."""
    app.storage.general.setdefault(key, el.value)
    return el.bind_value(app.storage.general, key)


def open_path(p):
    os.startfile(p)  # noqa: S606  (Windows only, like the pixi workspace)


# ---------------------------------------------------------------- charts

def curve_options(res, alpha):
    c = res.curve
    th = res.thresholds
    row = th[(th.key == "crc") & np.isclose(th.alpha, alpha) & th.achievable] if alpha is not None else th.iloc[:0]
    a = alpha or 0.1
    top = float(c.threshold[c.share_missed <= max(0.35, 3 * a)].max())
    sel = c[c.threshold <= top].iloc[::5]
    n = res.summary["cases"]
    marks = [{"yAxis": round(100 * a, 3), "name": f"α = {alpha_label(a)}",
              "lineStyle": {"type": "dotted", "color": MUTED, "width": 1.2},
              "label": {"formatter": f"budget α = {alpha_label(a)}", "position": "insideEndTop", "color": MUTED}}]
    if len(row):   # from the axis up to the budget, label above its top end
        lam = float(row.threshold.iloc[0])
        marks.append([{"coord": [lam, 0], "lineStyle": {"type": "solid", "color": INK, "width": 1.4},
                       "label": {"formatter": f"λ = {lam:.3f}", "position": "end", "color": INK,
                                 "fontWeight": "bold", "fontSize": 13, "distance": 6}},
                      {"coord": [lam, round(100 * a, 3)]}])
    return {
        "animation": False, "textStyle": TEXT,
        "grid": {"left": 64, "right": 28, "top": 44, "bottom": 52},
        "legend": {"top": 0, "left": 0, "textStyle": {"color": INK}},
        "tooltip": {"trigger": "axis", **PCT_TOOLTIP},
        "xAxis": {"type": "value", "min": 0, "max": round(top, 3), "name": "Threshold λ", "nameLocation": "middle",
                  "nameGap": 30, "axisLine": {"lineStyle": {"color": RULE}}, "splitLine": {"show": False}},
        "yAxis": {"type": "value", "min": 0, "name": "Objects missed, mean share per case", "nameLocation": "middle",
                  # scaled to the CRC curves; the RCPS-HB bound runs off the top where it no longer matters
                  "max": float(5 * np.ceil(20 * sel.crc_inflated.max())),
                  "nameGap": 44, "axisLabel": {"formatter": "{value}%"},
                  "splitLine": {"lineStyle": {"color": "#EEF1F4"}}},
        "series": [
            {"name": "Mean share missed, R(λ)", "type": "line", "symbol": "none", "color": OBSERVED,
             "lineStyle": {"width": 2.6}, "data": [[round(x, 3), round(100 * y, 3)] for x, y in
                                                     zip(sel.threshold, sel.share_missed)],
             "markLine": {"symbol": "none", "silent": True, "data": marks}},
            {"name": f"CRC-inflated: {n}/{n + 1}·R(λ) + 1/{n + 1}", "type": "line", "symbol": "none",
             "color": INK, "lineStyle": {"width": 1.4, "type": "dashed"},
             "data": [[round(x, 3), round(100 * y, 3)] for x, y in zip(sel.threshold, sel.crc_inflated)]},
        ] + hb_series(res, alpha, sel)}


def hb_series(res, alpha, sel):
    """RCPS-HB upper confidence bound: its threshold is the last λ where the bound stays within α."""
    if "hb" not in res.settings.methods:
        return []
    th = res.thresholds
    row = th[(th.key == "hb") & np.isclose(th.alpha, alpha) & th.achievable] if alpha is not None else th.iloc[:0]
    marks = []
    if len(row):
        lam = float(row.threshold.iloc[0])
        marks = [[{"coord": [lam, 0], "lineStyle": {"type": "dashed", "color": HB_COLOR, "width": 1.2},
                   "label": {"formatter": f"RCPS-HB λ = {lam:.3f}", "position": "middle", "color": HB_COLOR,
                             "distance": 6}},
                  {"coord": [lam, round(100 * alpha, 3)]}]]
    return [{"name": f"RCPS-HB upper bound (δ = {res.settings.delta:g})", "type": "line", "symbol": "none",
             "color": HB_COLOR, "lineStyle": {"width": 1.4, "type": "dotted"},
             "data": [[round(x, 3), round(100 * y, 3)] for x, y in zip(sel.threshold, sel.hb_ucb)],
             "markLine": {"symbol": "none", "silent": True, "data": marks}}]


def price_options(res, alpha):
    c = res.curve
    th = res.thresholds
    crc_rows = th[(th.key == "crc") & th.achievable]
    amax = float(th.alpha.max()) if len(th) else 0.1
    sel = c[(c.threshold > 0) & (c.share_missed <= max(0.35, 3 * amax)) & (c.false_per_case > 0)].iloc[::5]
    pts = [{"value": [round(100 * r.share_missed, 2), round(r.false_per_case, 2)],
            "label": {"show": True, "formatter": f"α = {alpha_label(r.alpha)}", "position": "right",
                      "color": INK, "fontWeight": "bold" if alpha is not None and np.isclose(r.alpha, alpha) else "normal"},
            "symbolSize": 13 if alpha is not None and np.isclose(r.alpha, alpha) else 9}
           for _, r in crc_rows.iterrows()]
    return {
        "animation": False, "textStyle": TEXT,
        "grid": {"left": 64, "right": 40, "top": 24, "bottom": 52},
        "tooltip": {"trigger": "item", ":formatter": "p => p.value[0].toFixed(1) + '% missed · ' + "
                                                      "p.value[1].toFixed(1) + ' false / case'"},
        "xAxis": {"type": "value", "min": 0, "name": "Objects missed, mean share per case (%)",
                  "nameLocation": "middle", "nameGap": 30, "axisLabel": {"formatter": "{value}%"},
                  "splitLine": {"show": False}, "axisLine": {"lineStyle": {"color": RULE}}},
        "yAxis": {"type": "log", "name": "False detections per case", "nameLocation": "middle", "nameGap": 44,
                  "splitLine": {"lineStyle": {"color": "#EEF1F4"}}},
        "series": [
            {"type": "line", "showSymbol": False, "color": INK, "lineStyle": {"width": 1.6}, "name": "All thresholds",
             "data": [[round(100 * x, 3), round(y, 3)] for x, y in zip(sel.share_missed, sel.false_per_case)]},
            {"type": "scatter", "color": OBSERVED, "name": "CRC thresholds", "data": pts, "z": 3},
        ]}


def check_options(res):
    v = res.validation
    v = v[(v.key == "crc") & (v.achievable > 0)]
    cats = [f"α = {alpha_label(a)}" for a in v.alpha]
    return {
        "animation": False, "textStyle": TEXT,
        "grid": {"left": 56, "right": 20, "top": 44, "bottom": 36},
        "legend": {"top": 0, "left": 0, "textStyle": {"color": INK}},
        "tooltip": {"trigger": "axis", **PCT_TOOLTIP},
        "xAxis": {"type": "category", "data": cats, "axisLine": {"lineStyle": {"color": RULE}}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "{value}%"}, "splitLine": {"lineStyle": {"color": "#EEF1F4"}}},
        "series": [
            {"name": "Promised: mean share missed ≤ α", "type": "bar", "color": PROMISE, "barGap": "8%",
             "data": [round(100 * a, 1) for a in v.alpha],
             "label": {"show": True, "position": "top", "formatter": "{c}%", "color": INK}},
            {"name": f"Observed on test cases (mean of {int(v.splits.iloc[0]) if len(v) else 0:,} splits)",
             "type": "bar", "color": OBSERVED, "data": [round(100 * r, 1) for r in v.test_share_missed],
             "label": {"show": True, "position": "top", "formatter": "{c}%", "color": OBSERVED, "fontWeight": "bold"}},
        ]}


# ---------------------------------------------------------------- page parts

def settings_from(ui_vals):
    """Settings from the form; ValueError with a readable message when a field is empty or out of range."""
    alphas = set()
    for x in ui_vals["alphas"].value or []:
        try:
            alphas.add(round(float(str(x).replace(",", ".").rstrip("% ")) / 100, 6))
        except ValueError:
            raise ValueError(f"budget {x!r} is not a number; type a percentage such as 7.5") from None
    methods = [m for m, cb in ui_vals["methods"].items() if cb.value]
    if not methods:
        raise ValueError("tick at least one method")
    for k, name in (("delta", "δ"), ("cal", "Calibration %")):
        if ui_vals[k].value is None:
            raise ValueError(f"{name} is empty")
    return analysis.Settings(alphas=sorted(alphas), methods=methods, delta=float(ui_vals["delta"].value),
                             n_splits=int(ui_vals["splits"].value or 0),
                             cal_fraction=float(ui_vals["cal"].value) / 100, seed=int(ui_vals["seed"].value or 0))


def titled(text, tip=None):
    """Panel title with an info icon; the explanation lives in its tooltip, not on the page."""
    with ui.row().classes("title-row w-full") as row:
        ui.label(text).classes("panel-title")
        if tip:
            ui.icon("info_outline", size="16px").classes("info").tooltip(tip)
    return row


def hint(text):
    """One line; the whole text on hover when it does not fit."""
    ui.label(text).classes("hint one-line").tooltip(text)


@ui.refreshable
def data_facts():
    if S.cases is None:
        ui.label("No data loaded.").classes("hint")
        return
    s = S.cases.summary()
    hint(S.cases.source)
    with ui.row().classes("facts"):
        for num, lab in ((s["cases"], "cases"), (f"{s['objects']:,}", "objects"),
                         (f"{s['objects_per_case']:.1f}", "per case"), (pct(s["min_alpha"]), "min α")):
            with ui.column().classes("gap-0"):
                ui.label(str(num)).classes("fact-num")
                ui.label(lab).classes("fact-lab")
    if s["groups"] != s["cases"]:
        ui.label(f"{s['groups']} groups: splits keep each group on one side.").classes("hint")


@ui.refreshable
def results_view(actions=None):
    if S.error:
        ui.label(S.error).classes("error")
    if S.cases is not None:
        for n in S.cases.notes:
            ui.label(n).classes("note")
    res = S.result
    if res is None:
        with ui.column().classes("empty"):
            ui.label("Thresholds with a guarantee").classes("empty-title")
            ui.label("For every miss budget α: the threshold λ, the promise that comes with it, and its price in "
                     "false detections.").classes("hint")
            with ui.row().classes("empty-steps"):
                for n, title, text, (label, icon, key) in (
                        (1, "Prepare the data", "Scores of your detector on labelled cases it has never seen: "
                         "COCO files or a simple CSV.", ("Data guide", "menu_book", "guide")),
                        (2, "Drop the files here", "Or browse. Each file is recognised by its content; for COCO, "
                         "drop annotations and detections together.", ("Browse…", "folder_open", "browse")),
                        (3, "Read the thresholds", "Calibration runs on its own after loading. Pick a budget to "
                         "see its curve, price and cases.", ("Try the example", "science", "example"))):
                    with ui.column().classes("empty-step"):
                        ui.label(str(n)).classes("guide-num")
                        ui.label(title).classes("guide-step-title")
                        ui.label(text).classes("hint")
                        if actions:
                            ui.button(label, icon=icon, on_click=actions[key]) \
                                .props("flat dense no-caps" if key != "browse" else "unelevated dense no-caps") \
                                .classes("px-3")
        return
    th = res.thresholds
    crc_rows = th[th.key == "crc"]
    st = res.settings
    with ui.row().classes("w-full items-baseline justify-between no-wrap"):
        ui.label("CRC: mean share of objects missed on a new case ≤ α").classes("panel-title")
        ui.label(f"fitted on all {res.summary['cases']} cases · click a budget").classes("hint")
    with ui.row().classes("budgets"):
        for _, r in crc_rows.iterrows():
            sel = S.alpha is not None and np.isclose(r.alpha, S.alpha)
            cls = "budget" + (" selected" if sel else "") + ("" if r.achievable else " unreachable")
            card = ui.column().classes(cls)
            if r.achievable:   # a button for the keyboard and screen readers too
                card.props(f'tabindex=0 role=button aria-pressed={str(sel).lower()} '
                           f'aria-label="Show budget {alpha_label(r.alpha)}"')
                for ev in ("click", "keydown.enter", "keydown.space.prevent"):
                    card.on(ev, lambda _, a=r.alpha: select_alpha(a))
            with card:
                with ui.row().classes("budget-head"):
                    ui.label(f"α = {alpha_label(r.alpha)}").classes("budget-alpha")
                    if r.achievable:
                        ui.label(f"λ = {r.threshold:.3f}").classes("budget-lam")
                    else:
                        ui.label("not reachable").classes("budget-lam small")
                if not r.achievable:
                    ui.label(f"needs α ≥ {pct(res.summary['min_alpha'])}").classes("budget-row")
                    continue
                ui.html(f"<b>{r.missed_per_case:.1f}</b> missed · <b>{r.false_per_case:.1f}</b> false / case"
                        ).classes("budget-row")
                ui.html(f"<b>{pct(r.cases_above_alpha, 0)}</b> of cases above α"
                        + (" · keeps all" if r.threshold == 0 else "")).classes("budget-row")
    others = th[(th.key != "crc") & th.achievable]
    if len(others):
        txt = "; ".join(f"{r.method} α = {alpha_label(r.alpha)}: λ = {r.threshold:.3f}" for _, r in others.iterrows())
        hint(f"With probability ≥ {1 - st.delta:.0%} (δ = {st.delta:g}): {txt}.")
    with ui.tabs(value=S.tab, on_change=lambda e: setattr(S, "tab", e.value)) \
            .props("align=left dense no-caps active-color=dark indicator-color=dark").classes("w-full") as tabs:
        t_curve = ui.tab("Curve")
        t_price = ui.tab("Price")
        t_check = ui.tab("Check")
        t_cases = ui.tab("Cases")
        t_table = ui.tab("All thresholds")
    with ui.tab_panels(tabs, value=S.tab, animated=False).classes("w-full panel results-panels"):
        with ui.tab_panel(t_curve):
            hint("Mean share of objects missed as the threshold rises. CRC takes the highest λ where the inflated "
                 "curve stays within the budget; RCPS-HB, the highest λ where its upper bound does.")
            ui.echart(curve_options(res, S.alpha)).classes("w-full chart-fill")
        with ui.tab_panel(t_price):
            hint("Fewer misses cost more false detections. Only the misses are guaranteed; false detections are "
                 "measured on these cases.")
            ui.echart(price_options(res, S.alpha)).classes("w-full chart-fill")
        with ui.tab_panel(t_check):
            if res.validation is None:
                ui.label("The check was switched off (0 splits).").classes("hint")
            else:
                hint(f"{st.n_splits:,} random splits: calibrate on {pct(st.cal_fraction, 0)} of the groups, test on "
                     "the rest. CRC bounds the mean over splits; single splits vary.")
                ui.echart(check_options(res)).classes("w-full chart-fill")
                cols = [("method", "Method", str), ("alpha", "α", alpha_label),
                        ("test_share_missed", "Test share missed", pct), ("test_q05", "5% of splits", pct),
                        ("test_q95", "95% of splits", pct), ("splits_above_alpha", "Splits above α", lambda x: pct(x, 0)),
                        ("test_false_per_case", "False / case", lambda x: f"{x:.1f}" if np.isfinite(x) else "–"),
                        ("achievable", "Threshold found", lambda x: pct(x, 0))]
                table(res.validation, cols).classes("check-table")
        with ui.tab_panel(t_cases):
            if S.alpha is None:
                ui.label("No reachable CRC budget.").classes("hint")
            else:
                lam = float(crc_rows[np.isclose(crc_rows.alpha, S.alpha)].threshold.iloc[0])
                pc = analysis.per_case(S.cases, lam).sort_values("share_missed", ascending=False)
                k = int((pc.share_missed > S.alpha).sum())
                hint(f"At λ = {lam:.3f}: {k} of {len(pc)} cases ({pct(k / len(pc), 0)}) miss more than "
                     f"{alpha_label(S.alpha)} of their objects; the mean is {pct(pc.share_missed.mean())}. "
                     "The guarantee is about the mean, not each case.")
                pc = pc.assign(above=np.where(pc.share_missed > S.alpha, "above α", ""))
                grouped = (pc.group.astype(str) != pc.case.astype(str)).any()
                t = table(pc, [("case", "Case", str)] + ([("group", "Group", str)] if grouped else []) +
                          [("objects", "Objects", str), ("missed", "Missed", str), ("share_missed", "Share missed", pct),
                           ("false_detections", "False", str), ("above", "", str)])
                t.add_slot("body-cell-above", '<q-td :props="props"><q-badge v-if="props.value" '
                                              'class="above-badge" :label="props.value" /></q-td>')
        with ui.tab_panel(t_table):
            hint("Measured on the same cases the thresholds were fitted on; only the guarantee on the mean share "
                 "missed carries over to new cases.")
            cols = [("method", "Method", str), ("alpha", "α", alpha_label),
                    ("threshold", "λ", lambda x: f"{x:.3f}" if np.isfinite(x) else "not reachable"),
                    ("share_missed", "Share missed", pct), ("missed_per_case", "Missed / case",
                                                            lambda x: f"{x:.1f}" if np.isfinite(x) else "–"),
                    ("false_per_case", "False / case", lambda x: f"{x:.1f}" if np.isfinite(x) else "–"),
                    ("cases_above_alpha", "Cases above α", lambda x: pct(x, 0))]
            table(th, cols)


def _raw(v):
    """Sort key sent along with the shown text: numbers as numbers (NaN last), the rest as text."""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, float, np.number)):
        return float(v) if np.isfinite(v) else None
    return str(v)


# empty last; text in natural order (slide 2 before slide 10)
SORT_JS = ("(a, b, ra, rb) => {{ const x = ra['{0}__raw'], y = rb['{0}__raw']; if (x === y) return 0; "
           "if (x === null) return 1; if (y === null) return -1; return typeof x === 'number' ? x - y : "
           "String(x).localeCompare(String(y), undefined, {{numeric: true}}); }}")


def table(df, cols):
    """cols: (field, header, formatter). Cells show the formatted text; clicking a header sorts by the value.
    All rows, in a box that fills the free height and scrolls under a fixed header."""
    rows = [{c: f(r[c]) for c, _, f in cols} | {f"{c}__raw": _raw(r[c]) for c, _, _ in cols} | {"_id": i}
            for i, (_, r) in enumerate(df.iterrows())]
    columns = [{"name": c, "label": lab, "field": c, "align": "left" if k == 0 else "right", "sortable": True,
                ":sort": SORT_JS.format(c)} for k, (c, lab, _) in enumerate(cols)]
    return ui.table(rows=rows, columns=columns, row_key="_id", pagination=0) \
        .props("flat dense binary-state-sort hide-bottom").classes("w-full fill-table")


def select_alpha(a):
    S.alpha = a
    results_view.refresh()


def default_alpha(res):
    crc_rows = res.thresholds[(res.thresholds.key == "crc") & res.thresholds.achievable]
    if not len(crc_rows):
        return None
    if np.isclose(crc_rows.alpha, 0.10).any():
        return 0.10
    return float(crc_rows.alpha.iloc[len(crc_rows) // 2])


# ---------------------------------------------------------------- page

GUARANTEE_TIP = ("CRC: mean share missed on a new case ≤ α. RCPS-HB / RCPS-WSR: the same with probability ≥ 1 − δ "
                 "over the calibration cases; needs more cases. Splits: random calibration/test splits for the check "
                 "(0 = skip). Cal %: share of groups used for calibration in each split.")
REFS = ("Angelopoulos et al., Conformal risk control, ICLR 2024 · Bates et al., Distribution-free, "
        "risk-controlling prediction sets, J ACM 2021")


def root():
    ui.add_css((HERE / "theme.css").read_text(encoding="utf-8"))
    ui.colors(primary=INK, secondary=MUTED, accent=OBSERVED)
    ui.page_title(TITLE)
    with ui.row().classes("topbar"):
        ui.label(TITLE).classes("app-name")
        ui.label("Distribution-free miss guarantees for detectors in medical imaging").classes("app-sub")
        ui.icon("info_outline", size="16px").classes("info").tooltip(REFS)
        ui.space()
        exp_label = ui.label("").classes("hint mono one-line").style("max-width:420px")
        open_btn = ui.button("Open last", icon="open_in_new",
                             on_click=lambda: S.last_export and open_path(S.last_export)).props("flat dense")
        exp_btn = ui.button("Export report", icon="download", on_click=lambda: export()) \
            .props("unelevated dense").classes("px-3") \
            .tooltip("HTML report, CSV tables, figures (PNG, SVG) and the matched scores")

    with ui.row().classes("layout"):
        with ui.column().classes("side"):
            # ---- data
            with ui.column().classes("panel"):
                with titled("1  Data"):
                    ui.space()
                    ui.button("Guide", icon="menu_book", on_click=lambda: guide_dlg.open()) \
                        .props("flat dense no-caps").classes("px-1") \
                        .tooltip("What the data look like and how to prepare them")
                    ui.button("Example", on_click=lambda: load_example()).props("flat dense").classes("px-1")
                    load_btn = ui.button("Load", on_click=lambda: load()).props("unelevated dense").classes("px-3")
                with ui.row().classes("drop-row"):
                    ui.icon("upload_file", size="20px")
                    ui.label("Drop data files here, or").classes("one-line")
                    ui.button("Browse…", on_click=lambda: browse()).props("flat dense no-caps").classes("px-1")
                mode = remember(ui.toggle({"scores": "Scores file", "coco": "COCO detection"}, value="scores"), "mode") \
                    .props("no-caps unelevated toggle-color=dark spread dense").classes("w-full mode")
                with ui.column().classes("w-full gap-2").bind_visibility_from(mode, "value", value="scores"):
                    with ui.row().classes("w-full no-wrap items-center gap-1"):
                        sc_path = remember(ui.input("Scores (.json or .csv)"), "sc_path") \
                            .on("keydown.enter", lambda: load()).props("dense outlined clearable").classes("flex-grow")
                        ui.button(icon="folder_open", on_click=lambda: pick_file(sc_path)).props("flat dense")
                with ui.column().classes("w-full gap-2").bind_visibility_from(mode, "value", value="coco"):
                    with ui.row().classes("w-full no-wrap items-center gap-1"):
                        gt_path = remember(ui.input("Annotations (COCO .json)"), "gt_path") \
                            .on("keydown.enter", lambda: load()).props("dense outlined clearable").classes("flex-grow")
                        ui.button(icon="folder_open", on_click=lambda: pick_file(gt_path)).props("flat dense")
                    with ui.row().classes("w-full no-wrap items-center gap-1"):
                        pr_path = remember(ui.input("Detections (COCO results .json)"), "pr_path") \
                            .on("keydown.enter", lambda: load()).props("dense outlined clearable").classes("flex-grow")
                        ui.button(icon="folder_open", on_click=lambda: pick_file(pr_path)).props("flat dense")
                    with ui.row().classes("w-full no-wrap gap-2 nums"):
                        case_rule = remember(ui.select({"image": "Each image", "regex": "Name pattern",
                                                        "mapping": "Case table"},
                                                       value="regex", label="Case ="), "case_rule") \
                            .props("dense outlined")
                        match_rule = remember(ui.select({"centre": "Centre", "iou": "IoU"},
                                                        value="centre", label="Match by"), "match_rule") \
                            .props("dense outlined")
                        iou = ui.number("IoU ≥", value=0.5, min=0.05, max=0.95, step=0.05, format="%.2f") \
                            .props("dense outlined").style("max-width:70px") \
                            .bind_visibility_from(match_rule, "value", value="iou")
                    pattern = remember(ui.input("Pattern (first group = case)", value=r"^([^_]+)_"), "pattern") \
                        .props("dense outlined").classes("w-full mono") \
                        .bind_visibility_from(case_rule, "value", value="regex")
                    with ui.row().classes("w-full no-wrap items-center gap-1") \
                            .bind_visibility_from(case_rule, "value", value="mapping"):
                        map_path = remember(ui.input("Case table (.csv)"), "map_path") \
                            .props("dense outlined clearable").classes("flex-grow")
                        ui.button(icon="folder_open", on_click=lambda: pick_file(map_path)).props("flat dense")
                data_facts()

            # ---- settings
            with ui.column().classes("panel"):
                titled("2  Guarantee", GUARANTEE_TIP)
                alphas = ui.select([1, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30], value=[5, 10, 15, 20], multiple=True,
                                   label="Miss budgets α (%)", new_value_mode="add-unique") \
                    .props("dense outlined use-chips").classes("w-full")
                with ui.row().classes("w-full gap-0 no-wrap justify-between"):
                    methods = {"crc": ui.checkbox("CRC", value=True).props("dense"),
                               "hb": ui.checkbox("RCPS-HB", value=True).props("dense"),
                               "wsr": ui.checkbox("RCPS-WSR", value=False).props("dense")}
                with ui.row().classes("w-full no-wrap gap-2 nums"):
                    delta = ui.number("δ", value=0.10, min=0.01, max=0.5, step=0.01, format="%.2f") \
                        .props("dense outlined")
                    splits = ui.number("Splits", value=2000, min=0, max=20000, step=500, format="%d") \
                        .props("dense outlined")
                    cal = ui.number("Cal %", value=75, min=10, max=90, step=5, format="%d").props("dense outlined")
                    seed = ui.number("Seed", value=0, min=0, step=1, format="%d").props("dense outlined")
                ui_vals = dict(alphas=alphas, methods=methods, delta=delta, splits=splits, cal=cal, seed=seed)
                run_btn = ui.button("Calibrate", on_click=lambda: calibrate()).props("unelevated").classes("w-full")
                settings_msg = ui.label("").classes("settings-msg")
                bar = ui.linear_progress(value=0, show_value=False).props("color=dark").classes("w-full")
                bar.set_visibility(False)

        with ui.column().classes("main"):
            results_view(dict(example=lambda: load_example(), guide=lambda: guide_dlg.open(),
                              browse=lambda: browse()))

    guide_dlg = guide.build(lambda: save_templates(), lambda: load_example())

    def sync_buttons():
        try:
            s, msg = settings_from(ui_vals), ""
            if S.result is not None and s != S.result.settings and not S.busy:
                msg = "Settings changed: press Calibrate to update the results."
        except ValueError as e:
            s, msg = None, f"Cannot calibrate: {e}."
        settings_msg.set_text(msg)   # hidden by CSS while empty
        cls = "settings-msg" + (" invalid" if s is None else "")
        if getattr(settings_msg, "_cls", None) != cls:   # classes() always resends; only on change
            settings_msg.classes(replace=cls); settings_msg._cls = cls
        run_btn.set_enabled(S.cases is not None and not S.busy and s is not None)
        exp_btn.set_enabled(S.result is not None and not S.busy)
        open_btn.set_enabled(S.last_export is not None)
        load_btn.set_enabled(not S.busy)
        bar.set_visibility(S.busy and not S.loading)
        bar.set_value(S.progress)

    ui.timer(0.2, sync_buttons)

    async def take_dropped():     # own timer: loading may take a while, sync_buttons must keep running
        if S.dropped and not S.busy:
            files, S.dropped = S.dropped, []
            await accept(files)

    ui.timer(0.3, take_dropped)

    async def accept(paths):
        """Files from Browse or dropped on the window: recognise each by its content, fill in the form, load."""
        found, unknown = {}, []
        for p in paths:
            kind = await run.io_bound(data.sniff, p)
            if kind is None:
                unknown.append(Path(p).name)
            else:
                found[kind] = p
        if unknown:
            ui.notify(f"Not recognised: {', '.join(unknown)}. The Guide shows what the files should look like.",
                      type="warning", multi_line=True)
        if "scores" in found:
            mode.value, sc_path.value = "scores", found["scores"]
            await load()
            return
        if not found:
            return
        mode.value = "coco"
        for kind, field in (("coco_gt", gt_path), ("coco_pred", pr_path), ("mapping", map_path)):
            if kind in found:
                field.value = found[kind]
        if "mapping" in found:
            case_rule.value = "mapping"
        missing = [name for name, field in (("annotation", gt_path), ("detection", pr_path)) if not field.value]
        if missing:
            ui.notify(f"Now add the COCO {missing[0]} file: drop it or press Browse.", type="info")
            return
        await load()

    async def browse():
        await accept(await pick_files())

    async def save_templates():
        folder = await pick_folder()
        if folder:
            out = guide.write_templates(folder)
            ui.notify(f"Templates written to {out}", type="positive")
            open_path(out)

    async def read(fn, *args):
        if S.busy:
            return
        S.error, S.busy, S.loading = None, True, True
        load_btn.props("loading")
        try:
            S.cases = await run.io_bound(fn, *args)
            S.result = None; S.alpha = None
        except Exception as e:  # shown to the user, not swallowed
            S.cases, S.result = None, None
            S.error = f"Could not read the data: {e}"
        finally:
            S.busy = S.loading = False
            load_btn.props(remove="loading")
        data_facts.refresh(); results_view.refresh()
        if S.cases is not None:
            await calibrate()

    async def load():
        if mode.value == "scores":
            if not sc_path.value:
                ui.notify("Choose a scores file first."); return
            kind = await run.io_bound(data.sniff, resolve(sc_path.value))
            if kind in ("coco_gt", "coco_pred", "mapping"):      # a COCO file in the scores field: move it
                await accept([resolve(sc_path.value)]); return
            await read(data.read_any, resolve(sc_path.value))
        else:
            if not gt_path.value or not pr_path.value:
                ui.notify("Choose the annotation and detection files."); return
            if case_rule.value == "mapping" and not map_path.value:
                ui.notify("Choose the case table."); return
            await read(data.read_coco, resolve(gt_path.value), resolve(pr_path.value),
                       case_rule.value, pattern.value, resolve(map_path.value) if map_path.value else None,
                       match_rule.value, float(iou.value or 0.5))

    async def load_example():
        mode.value = "scores"
        sc_path.value = str(EXAMPLE.relative_to(ROOT))
        await read(data.read_any, str(EXAMPLE))

    async def calibrate():
        if S.cases is None or S.busy:
            return
        try:
            s = settings_from(ui_vals)
        except ValueError as e:
            ui.notify(f"Cannot calibrate: {e}.", type="warning"); return
        S.busy, S.progress, S.error = True, 0.0, None
        try:
            S.result = await run.io_bound(analysis.run_all, S.cases, s, s.n_splits > 0,
                                          lambda p: setattr(S, "progress", p))
            S.alpha = S.alpha if S.alpha in s.alphas else default_alpha(S.result)
        except Exception as e:
            S.error = f"Calibration failed: {e}"
        finally:
            S.busy = False
        results_view.refresh()

    async def export():
        if S.result is None:
            return
        stamp = f"crc_{datetime.now():%Y%m%d-%H%M%S}"
        folder = await pick_folder()
        if not folder:      # dialog cancelled
            return
        out = Path(folder) / stamp

        def write():
            page = report.write(S.result, out, alpha=S.alpha)
            data.write_scores_csv(S.cases, out / "scores.csv")
            return page
        exp_btn.props("loading")
        try:
            page = await run.io_bound(write)
        except Exception as e:
            ui.notify(f"Export failed: {e}", type="negative"); return
        finally:
            exp_btn.props(remove="loading")
        S.last_export = page
        exp_label.set_text(str(out))
        ui.notify("Report written.", type="positive")
        open_path(page)


def sharpen_window_icon():
    """NiceGUI loads the .ico at its first (16 px) size for both icon slots, and Windows blows that up. Reload it at
    the sizes this window draws for its DPI: title bar (SM_CXSMICON) and Alt+Tab (SM_CXICON)."""
    import ctypes
    from ctypes import wintypes
    u32 = ctypes.windll.user32
    u32.FindWindowW.restype = wintypes.HWND
    u32.LoadImageW.restype = wintypes.HANDLE
    u32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int,
                               wintypes.UINT]
    u32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    hwnd = u32.FindWindowW(None, TITLE)
    if not hwnd:
        return
    dpi = u32.GetDpiForWindow(hwnd)
    for which, metric in ((0, 49), (1, 11)):                       # ICON_SMALL, ICON_BIG
        size = u32.GetSystemMetricsForDpi(metric, dpi)
        icon = u32.LoadImageW(None, str(HERE / "icon.ico"), 1, size, size, 0x10)   # IMAGE_ICON, LR_LOADFROMFILE
        if icon:
            u32.SendMessageW(hwnd, 0x80, which, icon)                # WM_SETICON


def main():
    # .ico: NiceGUI then also puts it on the title bar and the taskbar of the native window
    opts = dict(title=TITLE, reload=False, favicon=str(HERE / "icon.ico"), show_welcome_message=False)
    # the page connects once the window is up; a moment later NiceGUI has set its icon, so ours comes last
    app.on_connect(lambda: threading.Timer(1.0, sharpen_window_icon).start())
    app.native.on("drop", lambda e: S.dropped.extend(f for f in e.args.get("files", []) if f))
    app.native.window_args["min_size"] = (1200, 740)   # the layout fits without scrolling from here up
    ui.run(root, native=True, window_size=(1440, 920), **opts)


if __name__ in {"__main__", "__mp_main__"}:
    try:
        main()
    except Exception as e:   # without a console nobody would see it: say so in a Windows message box
        import ctypes
        import traceback
        traceback.print_exc()
        ctypes.windll.user32.MessageBoxW(None, f"CRC Calibrator could not start:\n\n{e}\n\nDetails are in {LOG}",
                                         TITLE, 0x10)   # MB_ICONERROR
        raise
