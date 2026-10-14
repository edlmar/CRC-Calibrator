"""Input data -> one Cases object: per case, the score at which each annotated object was found and the scores of
detections that matched nothing.

A case is the unit the guarantee is about (a slide, a patient, a study). Everything downstream needs only:
    objects[i]  score of the detection matched to each annotated object of case i (0 = never detected)
    false[i]    scores of case i's detections that matched no object
Matching is greedy by score, so one pass at a low score floor gives the matching at every higher threshold
(raising the threshold removes the lowest-scored detections first, and no earlier match changes).

Readers:
    read_scores_json   list of records {case|slide, objects|gt_best, false|fp, [group]}; records of one case
                       (tiles of a slide) are merged. crc/oof.json from the ICC project reads as is.
    read_scores_csv    long table: case, kind (object|false), score, [group]
    read_coco          COCO ground truth + COCO detection results; matching done here
"""
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CASE_KEYS = ("case", "slide", "patient", "study", "image", "file")
OBJECT_WORDS = {"object", "objects", "gt", "lesion", "cell", "found", "target", "positive"}
FALSE_WORDS = {"false", "fp", "false_positive", "false detection", "false_detection", "extra"}


@dataclass
class Cases:
    names: list                       # case ids, str
    objects: list                     # per case: np.ndarray of scores (0 = never detected)
    false: list                       # per case: np.ndarray of false-detection scores
    groups: np.ndarray = None         # per case group id (e.g. patient); splits keep a group together
    source: str = ""                  # human-readable description of where the data came from
    notes: list = field(default_factory=list)   # warnings found while reading

    def __post_init__(self):
        self.objects = [np.asarray(o, float) for o in self.objects]
        self.false = [np.asarray(f, float) for f in self.false]
        if self.groups is None:
            self.groups = np.asarray(self.names, dtype=object)
        self.groups = np.asarray(self.groups, dtype=object)

    def __len__(self):
        return len(self.names)

    @property
    def n_objects(self):
        return np.array([len(o) for o in self.objects])

    @property
    def max_score(self):
        s = [x.max() for x in self.objects + self.false if len(x)]
        return float(max(s)) if s else 1.0

    def summary(self):
        k = self.n_objects
        never = np.array([(o <= 0).sum() for o in self.objects])
        floor = float(np.mean(np.divide(never, k, out=np.zeros(len(k)), where=k > 0))) if len(k) else 0.0
        n = len(self)
        return dict(cases=n, groups=len(np.unique(self.groups)), objects=int(k.sum()),
                    never_detected=int(never.sum()),
                    missed_floor=floor,                      # mean share missed at every threshold
                    min_alpha=n / (n + 1) * floor + 1 / (n + 1) if n else 1.0,   # smallest budget CRC can meet
                    objects_per_case=float(k.mean()) if len(k) else 0.0, min_objects=int(k.min()) if len(k) else 0,
                    max_objects=int(k.max()) if len(k) else 0, cases_without_objects=int((k == 0).sum()),
                    false_per_case_at_0=float(np.mean([len(f) for f in self.false])) if len(k) else 0.0)


def _check(cases):
    """Warnings a user should see before trusting the numbers."""
    if not len(cases):
        raise ValueError(f"no cases found in {cases.source or 'the input'}")
    notes = list(cases.notes)
    s = cases.summary()
    if s["cases_without_objects"]:
        notes.append(f"{s['cases_without_objects']} case(s) have no annotated objects: their loss is 0 at every "
                     "threshold, which makes the mean easier to meet.")
    if s["never_detected"]:
        notes.append(f"{s['never_detected']} annotated object(s) were never detected (score 0). They count as missed "
                     f"at every threshold: {s['missed_floor']:.1%} of objects per case are missed whatever λ is, so "
                     f"the smallest budget CRC can promise is {s['min_alpha']:.1%}.")
    if len(cases) < 20:
        notes.append(f"Only {len(cases)} cases: the smallest budget CRC can promise is at least 1/(n+1) = "
                     f"{1 / (len(cases) + 1):.1%}.")
    lows = [x.min() for x in cases.false if len(x)]
    if lows and min(lows) > 0.05:
        notes.append(f"The lowest false-detection score is {min(lows):.2f}: detections below it were probably "
                     "dropped before export, so thresholds under it are not reliable. Export with a low score "
                     "floor (e.g. 0.001).")
    if cases.max_score > 1:
        notes.append(f"Scores go up to {cases.max_score:.3g}, so they are not probabilities; the threshold grid "
                     "is stretched to cover them.")
    cases.notes = notes
    return cases


def _merge(rows):
    """rows: iterable of (case, group, objects, false) -> Cases, cases in first-seen order."""
    order, obj, fal, grp = [], {}, {}, {}
    for case, group, o, f in rows:
        case = str(case)
        if case not in obj:
            order.append(case); obj[case], fal[case] = [], []
            grp[case] = case if group in (None, "") else str(group)
        obj[case] += list(o); fal[case] += list(f)
    return order, [obj[c] for c in order], [fal[c] for c in order], [grp[c] for c in order]


def _first(rec, keys, default=None):
    for k in keys:
        if k in rec:
            return rec[k]
    return default


def read_scores_json(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = _first(data, ("cases", "records", "data"), [])
    rows = []
    for i, r in enumerate(data):
        case = _first(r, CASE_KEYS)
        if case is None:
            raise ValueError(f"record {i}: no case id (expected one of case, slide, patient, study, image)")
        o = _first(r, ("objects", "gt_best", "found", "object_scores"), [])
        f = _first(r, ("false", "fp", "false_scores", "false_positives"), [])
        rows.append((case, _first(r, ("group", "patient_id")), o, f))
    names, o, f, g = _merge(rows)
    n_rec = len(data)
    src = f"{path.name}: {n_rec} records" + (f" merged into {len(names)} cases" if n_rec != len(names) else "")
    return _check(Cases(names, o, f, g, source=src))


def read_scores_csv(path):
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096); fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(fh, dialect=dialect)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}
        need = {"case", "kind", "score"}
        if not need <= cols.keys():
            raise ValueError(f"{path.name}: needs columns case, kind, score (found {', '.join(reader.fieldnames)})")
        rows = []
        for n, r in enumerate(reader, start=2):
            kind = r[cols["kind"]].strip().lower()
            score = float(r[cols["score"]]) if r[cols["score"]].strip() else 0.0
            if kind in OBJECT_WORDS:
                o, f = [score], []
            elif kind in FALSE_WORDS:
                o, f = [], [score]
            elif kind in ("", "none", "empty"):     # a case with nothing in it still counts as a case
                o, f = [], []
            else:
                raise ValueError(f"{path.name}, line {n}: kind must be 'object' or 'false', got {kind!r}")
            rows.append((r[cols["case"]], r.get(cols.get("group", ""), None) if "group" in cols else None, o, f))
    names, o, f, g = _merge(rows)
    return _check(Cases(names, o, f, g, source=f"{path.name}: {len(names)} cases"))


def write_scores_csv(cases, path):
    """The normalised long table (round-trips through read_scores_csv)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "group", "kind", "score"])
        for name, grp, o, f in zip(cases.names, cases.groups, cases.objects, cases.false):
            if not len(o) and not len(f):
                w.writerow([name, grp, "none", ""])
            w.writerows([name, grp, "object", f"{s:.6g}"] for s in o)
            w.writerows([name, grp, "false", f"{s:.6g}"] for s in f)


# ---------------------------------------------------------------- COCO detection

def iou_matrix(a, b):
    """IoU between xywh boxes a (n, 4) and b (m, 4)."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match(gt_xywh, det_xywh, det_score, rule="centre", iou_thr=0.5):
    """Greedy one-to-one matching by descending score.
    rule 'centre': a detection hits an object when its centre lies within half the object's larger side of the
    object's centre (nearest free object wins). rule 'iou': IoU >= iou_thr (highest free IoU wins).
    Returns (objects, false): per object the score of its match (0 if none), and unmatched detection scores."""
    g = np.asarray(gt_xywh, float).reshape(-1, 4)
    d = np.asarray(det_xywh, float).reshape(-1, 4)
    s = np.asarray(det_score, float).reshape(-1)
    best, used, false = np.zeros(len(g)), np.zeros(len(g), bool), []
    if rule == "iou" and len(g) and len(d):
        iou = iou_matrix(d, g)
    gc = g[:, :2] + g[:, 2:] / 2
    reach = np.maximum(g[:, 2], g[:, 3]) / 2
    for k in np.argsort(-s, kind="stable"):
        if len(g) == 0:
            false.append(s[k]); continue
        if rule == "iou":
            ok = ~used & (iou[k] >= iou_thr)
            key = -iou[k]
        else:
            dc = d[k, :2] + d[k, 2:] / 2
            dist = np.hypot(gc[:, 0] - dc[0], gc[:, 1] - dc[1])
            ok = ~used & (dist <= reach)
            key = dist
        if ok.any():
            j = np.flatnonzero(ok)[np.argmin(key[ok])]
            used[j] = True; best[j] = s[k]
        else:
            false.append(s[k])
    return best, np.asarray(false, float)


def case_of(file_name, rule="image", pattern=r"^([^_]+)_", mapping=None):
    """Case id for an image. rule: 'image' (each image is a case), 'regex' (first group of pattern, else whole
    name), 'mapping' (dict file_name -> case; also tried on the stem)."""
    if rule == "mapping":
        stem = Path(file_name).stem
        if file_name in mapping:
            return mapping[file_name]
        if stem in mapping:
            return mapping[stem]
        raise KeyError(f"{file_name} is not in the case table")
    if rule == "regex":
        m = re.search(pattern, Path(file_name).name)
        return m.group(1) if m and m.groups() else (m.group(0) if m else Path(file_name).stem)
    return Path(file_name).stem


def read_mapping(path):
    """CSV with columns image (or file_name) and case, optional group -> (case map, group map)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        dialect = csv.Sniffer().sniff(fh.read(4096), delimiters=",;\t"); fh.seek(0)
        reader = csv.DictReader(fh, dialect=dialect)
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        img = cols.get("image") or cols.get("file_name") or cols.get("file")
        if not img or "case" not in cols:
            raise ValueError(f"{Path(path).name}: needs columns image (or file_name) and case")
        cmap, gmap = {}, {}
        for r in reader:
            cmap[r[img]] = r[cols["case"]]
            if "group" in cols:
                gmap[r[cols["case"]]] = r[cols["group"]]
    return cmap, gmap


def read_coco(gt_path, pred_path, case_rule="image", pattern=r"^([^_]+)_", mapping_csv=None, rule="centre",
              iou_thr=0.5, categories=None):
    """COCO annotations + detections -> Cases. pred_path: a COCO results list [{image_id, bbox, score, ...}] or a
    COCO dataset whose annotations carry 'score'. Predictions may also refer to images by file_name.
    categories: optional set of category ids to keep (default: all, class-agnostic)."""
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    if isinstance(pred, dict):
        pred = pred.get("annotations", [])
    images = {im["id"]: im for im in gt["images"]}
    by_name = {Path(im["file_name"]).name: im["id"] for im in gt["images"]}
    keep = (lambda c: True) if not categories else (lambda c: c in categories)
    g_boxes = {i: [] for i in images}
    for a in gt.get("annotations", []):
        if keep(a.get("category_id")) and not a.get("iscrowd", 0):
            g_boxes[a["image_id"]].append(a["bbox"])
    d_boxes, d_scores, unknown = {i: [] for i in images}, {i: [] for i in images}, 0
    for p in pred:
        iid = p.get("image_id")
        if iid not in images and "file_name" in p:
            iid = by_name.get(Path(p["file_name"]).name)
        if iid not in images:
            unknown += 1; continue
        if keep(p.get("category_id")):
            if "score" not in p:
                raise ValueError("a prediction has no 'score'")
            d_boxes[iid].append(p["bbox"]); d_scores[iid].append(p["score"])
    cmap, gmap = read_mapping(mapping_csv) if case_rule == "mapping" else (None, {})
    rows = []
    for iid, im in images.items():
        o, f = match(g_boxes[iid], d_boxes[iid], d_scores[iid], rule, iou_thr)
        case = case_of(im["file_name"], case_rule, pattern, cmap)
        rows.append((case, gmap.get(case), o, f))
    names, o, f, g = _merge(rows)
    how = "centre within half box" if rule == "centre" else f"IoU >= {iou_thr:g}"
    cases = Cases(names, o, f, g, source=f"{Path(gt_path).name} + {Path(pred_path).name}: {len(images)} images "
                                           f"in {len(names)} cases, matched by {how}")
    if unknown:
        cases.notes.append(f"{unknown} prediction(s) refer to images that are not in the annotation file; ignored.")
    return _check(cases)


def sniff(path):
    """What a file holds, judged by its content: 'scores', 'coco_gt', 'coco_pred', 'mapping', or None.
    Lets the GUI take whatever the user drops and put it in the right place."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            cols = {c.strip().lower() for c in re.split(r"[,;\t]", fh.readline())}
        if {"case", "kind", "score"} <= cols:
            return "scores"
        if "case" in cols and cols & {"image", "file_name", "file"}:
            return "mapping"
        return None
    if ext != ".json":
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(d, dict) and "images" in d and "annotations" in d:
        anns = d["annotations"]
        return "coco_pred" if anns and all("score" in a for a in anns[:50]) else "coco_gt"
    if isinstance(d, dict):
        d = _first(d, ("cases", "records", "data"))
    if isinstance(d, list) and d and isinstance(d[0], dict):
        if "bbox" in d[0] and "score" in d[0]:
            return "coco_pred"
        if _first(d[0], CASE_KEYS) is not None:
            return "scores"
    return None


def read_any(path):
    """Score files by extension (.json -> records, .csv/.tsv/.txt -> long table)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return read_scores_json(p)
    return read_scores_csv(p)
