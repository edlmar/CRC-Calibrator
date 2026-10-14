"""Command line: python -m core.cli <scores.json|csv> [options]   or   python -m core.cli coco <gt> <pred> [options]"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import analysis, data, report


def add_common(p):
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--methods", nargs="+", default=["crc", "hb"], choices=["crc", "hb", "wsr"])
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--splits", type=int, default=2000, help="random calibration/test splits (0 = skip the check)")
    p.add_argument("--cal-fraction", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output folder (default: results/<timestamp>)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="crc_calibrator", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("scores", help="per-object scores (JSON records or CSV case,kind,score)")
    a.add_argument("path"); add_common(a)
    c = sub.add_parser("coco", help="COCO annotations + COCO detections")
    c.add_argument("gt"); c.add_argument("pred")
    c.add_argument("--case", choices=["image", "regex", "mapping"], default="image")
    c.add_argument("--pattern", default=r"^([^_]+)_")
    c.add_argument("--mapping", default=None)
    c.add_argument("--match", choices=["centre", "iou"], default="centre")
    c.add_argument("--iou", type=float, default=0.5)
    c.add_argument("--save-scores", default=None, help="also write the matched per-object scores as CSV")
    add_common(c)
    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help(); return 1
    try:
        s = analysis.Settings(alphas=args.alphas, methods=args.methods, delta=args.delta, n_splits=args.splits,
                              cal_fraction=args.cal_fraction, seed=args.seed)
    except ValueError as e:
        ap.error(str(e))
    if args.cmd == "coco":
        cases = data.read_coco(args.gt, args.pred, args.case, args.pattern, args.mapping, args.match, args.iou)
        if args.save_scores:
            data.write_scores_csv(cases, args.save_scores)
    else:
        cases = data.read_any(args.path)
    print(cases.source)
    for n in cases.notes:
        print("note:", n)
    res = analysis.run_all(cases, s, validate_splits=args.splits > 0)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(res.thresholds.drop(columns=["key", "promise"]).round(4).to_string(index=False))
        if res.validation is not None:
            print(res.validation.drop(columns=["key"]).round(4).to_string(index=False))
    out = Path(args.out or Path("results") / f"{datetime.now():%Y%m%d-%H%M%S}")
    print("report ->", report.write(res, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
