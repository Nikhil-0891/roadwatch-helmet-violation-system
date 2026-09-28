# RoadWatch — Helmet Violation & Plate Reader

Flask dashboard for image/video uploads, helmet candidate detection, plate OCR, violation snapshots, and SQLite history.

## Setup and run (Windows / Python 3.11)

```powershell
cd helmet-violation-system
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Tesseract must be installed separately. The configured default is `C:\Program Files\Tesseract-OCR\tesseract.exe`; change `TESSERACT_CMD` in `config.py` if needed.

RoadWatch keeps uploads, detections, snapshots, and annotated outputs only for the current server run. Starting the app clears data left by an earlier run; stopping it normally (Ctrl+C) clears the current run's generated data and database records. The database file and schema remain, but its session records are cleared. A forced process kill may prevent shutdown cleanup; any leftovers are cleared the next time the app starts.

Run local checks with `python test_system.py`.

## What model is included

`yolo-coco/yolov3-obj.cfg`, `yolov3-obj_2400.weights`, and `cocohelmet.names` are a Darknet YOLOv3 model with **one class: `Helmet`**. This is confirmed by the names file and each YOLO block in the config (`classes=1`, output filters=18). It cannot inherently classify a bare head.

With this model, RoadWatch uses OpenCV HOG person boxes and checks whether a detected Helmet box overlaps the estimated top 30% of each person box. HOG detects generic pedestrians, not motorcycle riders. A matching helmet box is positive helmet evidence; no overlap becomes **unknown**, not a violation. HOG's SVM margin is retained only as a diagnostic score and never shown as probability. This fallback cannot identify no-helmet violations reliably.

For true two-class detection, train/fine-tune a detector on a labeled dataset containing both `Helmet` and `No-Helmet` rider/head boxes, including representative camera angles, distances, occlusion, lighting, and local road conditions. The current inference integration expects a Darknet YOLO config, matching weights, and names file; names must include a recognized no-helmet label (see `NO_HELMET_LABELS` in `config.py`). Place the matching files under `yolo-coco/` and update `HELMET_CFG`, `HELMET_WEIGHTS`, and `HELMET_NAMES`. The bundled one-class config/weights cannot be converted into a valid two-class model by changing the names file; the model must be trained/fine-tuned with two classes and a matching output head. A modern YOLOv8/YOLO11 checkpoint is not directly loadable by this Darknet inference path; it must be exported and integrated with a matching inference backend before use.

Training needs a dataset in the detector's supported annotation format with class IDs 0=`Helmet`, 1=`No-Helmet` and a train/validation split. Audit labels carefully: the target regions and class definitions must be consistent. GPU training is recommended. No dataset or two-class checkpoint is included.

## Plates, OCR, and records

Plate localization currently tries OpenCV's Russian plate Haar cascade, then an edge/contour geometry fallback. Candidate crops are enlarged, contrast enhanced, denoised, thresholded, and morphologically cleaned before Tesseract OCR. OCR output is only returned as a vehicle number when it matches a configured Indian plate template; position-aware `O/0`, `I/1`, `S/5`, `B/8` style corrections are applied only where the template requires that character type. Otherwise the displayed value is `Unreadable`, with OCR confidence/status retained. Classical plate localization and OCR remain sensitive to small, blurred, angled, or occluded plates.

Person-track snapshots are saved as padded, boxed crops in `data/snapshots/tracks/` and shown in the track table; confirmed violation snapshots are also written to `data/snapshots/`, with localized plate crops under `data/snapshots/plates/`. These are temporary for the current server run. The plate reader records whether it found candidates, validated their geometry, attempted OCR, failed OCR, failed format validation, or extracted a valid plate. OCR runs only in the person-centered rider/nearby-vehicle evidence ROI, never over the full video frame. This project does not include a motorcycle/vehicle detector, so that ROI is approximate.

Each upload gets a processing session ID. `detections` stores one row per associated person track in that session; `violation_events` stores confirmed violation events separately; `processing_sessions` retains upload status and timing. Dashboard statistics are scoped to the latest upload or an explicit session ID, while prior events remain in SQLite. Additive schema migrations preserve prior data. A development reset is available with `python reset_data.py --yes`; it clears events, track rows, sessions, snapshots, and generated outputs while preserving uploads, weights, config, code, and the database schema.

Video uses configurable frame sampling, one-to-one IoU/center-distance association, track expiry, and temporal confirmation. A true two-class `No-Helmet` result must persist for `HELMET_CONFIRMATION_FRAMES` sampled observations before one event is created for that track; event cooldown is configurable. Single-class fallback absences stay unknown. Association and confirmation defaults are in `config.py`. With `DEBUG_DETECTIONS = True`, per-frame person/head/helmet boxes, overlap, scores, final state, and track ID are appended to `data/outputs/detection_debug.jsonl`.

## Project layout

- `app.py`: Flask page, upload, and JSON endpoints
- `pipeline.py`: image/video flow, annotation, snapshots, tracking
- `detection/helmet_detector.py`: Darknet inference and fallback classification
- `detection/plate_reader.py`: plate candidates, OCR, normalization/validation
- `database.py`: SQLite persistence and dashboard statistics
- `config.py`: paths and thresholds
- `templates/`, `static/`: dashboard
- `test_system.py`: local smoke checks for model loading/inference, validation, snapshots, API, and video deduplication

## Known limitations

- The included helmet model is one-class and the HOG overlap fallback is not a reliable Helmet/No-Helmet classifier. A properly labeled and trained two-class model is required for the target classification.
- Haar/contour localization is a baseline. Production plate reading generally needs a plate detector trained for the deployment camera/domain and image quality.
- The current person detector is generic HOG, not a motorcycle/rider detector; dashboard counts are person tracks until a rider/vehicle model is added.
- IoU/center-distance tracking is lightweight and can lose identities during long occlusions, abrupt motion, or crossing people; temporal confirmation and cooldown limit duplicate events while a track remains associated.
- The Flask development server is intended for local use, not public deployment.
