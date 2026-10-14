"""The data guide: what the input looks like, how to prepare it, and template files to start from."""
import html
from pathlib import Path

from nicegui import ui

SCORES_CSV = """case,kind,score,group
slide_01,object,0.91,patient_A
slide_01,object,0.47,patient_A
slide_01,object,0,patient_A
slide_01,false,0.12,patient_A
slide_01,false,0.03,patient_A
slide_02,object,0.88,patient_A
slide_02,object,0.64,patient_A
slide_02,false,0.40,patient_A
slide_03,object,0.95,patient_B
slide_03,object,0.02,patient_B
slide_03,false,0.31,patient_B
"""

SCORES_JSON = """[
  {"case": "slide_01", "group": "patient_A", "objects": [0.91, 0.47, 0], "false": [0.12, 0.03]},
  {"case": "slide_02", "group": "patient_A", "objects": [0.88, 0.64], "false": [0.40]},
  {"case": "slide_03", "group": "patient_B", "objects": [0.95, 0.02], "false": [0.31]}
]
"""

CASE_TABLE_CSV = """image,case,group
12_tile1.png,12,patient_A
12_tile2.png,12,patient_A
13_tile1.png,13,patient_B
"""

DETECTIONS_JSON = """[
  {"image_id": 1, "bbox": [104.0, 220.5, 18.0, 17.5], "score": 0.93, "category_id": 1},
  {"image_id": 1, "bbox": [300.2, 48.0, 21.0, 19.0], "score": 0.004, "category_id": 1}
]
"""

TEMPLATES = {"scores_template.csv": SCORES_CSV, "scores_template.json": SCORES_JSON,
             "case_table_template.csv": CASE_TABLE_CSV, "coco_detections_example.json": DETECTIONS_JSON}

SNIPPET_SCORES = '''import csv

# One entry per case, from your own matching of detections to annotated objects:
#   objects: per annotated object, the score of the detection that found it (0 = never found)
#   false:   scores of detections that matched no object
results = [
    ("slide_01", "patient_A", [0.91, 0.47, 0.0], [0.12, 0.03]),
]

with open("scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "kind", "score", "group"])
    for case, group, objects, false in results:
        w.writerows([case, "object", s, group] for s in objects)
        w.writerows([case, "false", s, group] for s in false)
'''

SNIPPET_COCO = '''import json

# predictions: per image of the annotation file, its boxes (x, y, width, height in pixels) and scores.
# Run the model with a LOW score threshold (e.g. 0.001) so that weak detections are kept.
detections = []
for image_id, boxes, scores in predictions:
    detections += [{"image_id": image_id, "bbox": [float(v) for v in box], "score": float(s), "category_id": 1}
                   for box, s in zip(boxes, scores)]

with open("detections.json", "w") as f:
    json.dump(detections, f)
# image_id must be the id of the image in the annotation file; "file_name" instead of "image_id" works too.
'''

CHECKLIST = [
    ("Held-out predictions", "The model never saw these cases in training: use a test set or out-of-fold "
                             "predictions from cross-validation. Otherwise the guarantee does not hold."),
    ("Every annotated object", "Include all annotated objects of each case, also those the detector never found "
                               "(score 0). Missing them hides misses."),
    ("Low score floor", "Export detections down to a score of about 0.001. Thresholds below the lowest exported "
                        "score cannot be evaluated."),
    ("Same setting", "Cases from the same population, lab, stain, scanner and annotation protocol as where the "
                     "model will be used. After any change, calibrate again."),
    ("Enough cases", "At least about 20 cases. With n cases the smallest budget CRC can promise is 1/(n+1): "
                     "40 cases → 2.4%, 20 cases → 4.8%. RCPS needs more."),
    ("Groups", "If several cases come from one patient, give them the same group, so that the check never "
               "splits a patient between calibration and test."),
]


def write_templates(folder):
    folder = Path(folder) / "crc_templates"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in TEMPLATES.items():
        (folder / name).write_text(text, encoding="utf-8")
    (folder / "make_scores_csv.py").write_text(SNIPPET_SCORES, encoding="utf-8")
    (folder / "make_coco_detections.py").write_text(SNIPPET_COCO, encoding="utf-8")
    return folder


def _p(text):
    ui.label(text).classes("guide-p")


def build(on_save_templates, on_example):
    """The guide dialog; returns it (call .open())."""
    with ui.dialog() as dlg, ui.card().classes("guide"):
        with ui.row().classes("w-full items-center no-wrap"):
            ui.label("Preparing the data").classes("guide-title")
            ui.space()
            ui.button("Save templates…", icon="file_download", on_click=on_save_templates) \
                .props("flat dense no-caps").tooltip("Writes example files and two small Python scripts to a folder")
            ui.button(icon="close", on_click=dlg.close).props("flat round dense")
        with ui.tabs().props("dense no-caps align=left active-color=dark indicator-color=dark").classes("w-full") \
                as tabs:
            t_start = ui.tab("Start here")
            t_scores = ui.tab("Scores file")
            t_coco = ui.tab("COCO detection")
            t_check = ui.tab("Checklist")
        with ui.tab_panels(tabs, value=t_start, animated=False).classes("guide-body"):
            with ui.tab_panel(t_start):
                _p("The program needs, for labelled cases the model has never seen, how confidently the detector "
                   "found each annotated object and how confident its false detections were. No images, no model.")
                _p("A case is the unit of the guarantee: a slide, a patient, a study. Tiles of one slide are one case.")
                with ui.column().classes("guide-steps"):
                    for n, (title, text) in enumerate((
                            ("Pick the format you already have",
                             "COCO files from your detector (annotations + detections) → the program matches them. "
                             "Your own list of found and missed objects per case → a scores file."),
                            ("Check the data", "Held-out predictions, every annotated object, a low score floor: "
                                               "see the Checklist tab."),
                            ("Drop the files on the window",
                             "Or press Browse. The program recognises each file by its content, fills in the form "
                             "and loads it. For COCO, drop both files together.")), start=1):
                        with ui.row().classes("no-wrap items-start gap-3"):
                            ui.label(str(n)).classes("guide-num")
                            with ui.column().classes("gap-0"):
                                ui.label(title).classes("guide-step-title")
                                ui.label(text).classes("guide-p")
                with ui.row().classes("gap-2"):
                    ui.button("Load the example", icon="science", on_click=lambda: (dlg.close(), on_example())) \
                        .props("unelevated dense no-caps").classes("px-3")
                    ui.button("Save templates…", icon="file_download", on_click=on_save_templates) \
                        .props("flat dense no-caps")
            with ui.tab_panel(t_scores):
                _p("Use this when you have matched detections to annotated objects yourself. One row per annotated "
                   "object and per false detection. CSV with comma, semicolon or tab.")
                ui.code(SCORES_CSV, language="text").classes("w-full guide-code")
                ui.html("<table class='guide-table'>" + "".join(
                    f"<tr><td class='mono'>{col}</td><td>{html.escape(meaning)}</td></tr>" for col, meaning in (
                        ("case", "case id: slide, patient, study"),
                        ("kind", "object: an annotated object · false: a detection that matched no object"),
                        ("score", "object: score of the detection that found it, 0 if none found · false: its score"),
                        ("group", "optional: e.g. patient id; keeps related cases together"))) + "</table>")
                _p("The same as JSON, one record per case (records of one case, e.g. its tiles, are merged):")
                ui.code(SCORES_JSON, language="json").classes("w-full guide-code")
                _p("Writing the CSV from your own matching:")
                ui.code(SNIPPET_SCORES, language="python").classes("w-full guide-code")
            with ui.tab_panel(t_coco):
                _p("Use this when your detector writes COCO. Two files: the annotation file (COCO dataset with "
                   "images and annotations) and the detections in COCO results format. The program matches them.")
                ui.code(DETECTIONS_JSON, language="json").classes("w-full guide-code")
                _p("Cases: each image is a case, or the case comes from the file name (the first group of a pattern, "
                   "e.g. ^(\\d+)_ turns 12_tile3.png into case 12), or from a case table:")
                ui.code(CASE_TABLE_CSV, language="text").classes("w-full guide-code")
                _p("Matching: Centre (the detection's centre lies within half the object's larger side; good for "
                   "cells and small objects) or IoU (box overlap ≥ a threshold). Category ids are ignored: all "
                   "boxes count as one class.")
                _p("Writing detections from your own model:")
                ui.code(SNIPPET_COCO, language="python").classes("w-full guide-code")
            with ui.tab_panel(t_check):
                for title, text in CHECKLIST:
                    with ui.row().classes("no-wrap items-start gap-3"):
                        ui.icon("check_circle", size="20px").classes("guide-ok")
                        with ui.column().classes("gap-0"):
                            ui.label(title).classes("guide-step-title")
                            ui.label(text).classes("guide-p")
    return dlg
