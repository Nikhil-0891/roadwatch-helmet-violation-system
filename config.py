"""
Central configuration for the Helmet Violation & Plate Reader system.
Keeping every tunable value in one file means the detection code never
hardcodes a path or a threshold - change behaviour here, not in the logic.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "snapshots")   # cropped violation images
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")       # full annotated image/video
PLATE_SNAPSHOT_DIR = os.path.join(SNAPSHOT_DIR, "plates")
DETECTION_DEBUG_LOG = os.path.join(OUTPUT_DIR, "detection_debug.jsonl")
DB_PATH = os.path.join(BASE_DIR, "data", "violations.db")

for _d in (UPLOAD_DIR, SNAPSHOT_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Helmet detector (your existing custom-trained YOLOv3-Darknet model)
# ---------------------------------------------------------------------------
YOLO_DIR = os.path.join(BASE_DIR, "yolo-coco")
HELMET_CFG = os.path.join(YOLO_DIR, "yolov3-obj.cfg")
HELMET_NAMES = os.path.join(YOLO_DIR, "cocohelmet.names")
# Bundled Darknet weights (single-class Helmet detector).
HELMET_WEIGHTS = os.path.join(YOLO_DIR, "yolov3-obj_2400.weights")

HELMET_CONFIDENCE_THRESHOLD = 0.5
HELMET_NMS_THRESHOLD = 0.3
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# A genuine two-class Darknet model needs matching config, weights, and names.
NO_HELMET_LABELS = {"no-helmet", "no_helmet", "without helmet", "without-helmet", "nohelmet"}

# ---------------------------------------------------------------------------
# Person / rider detector (OpenCV's built-in HOG detector - ships with
# opencv-python, needs no extra download). It detects generic pedestrians,
# not motorcyclists, and cannot by itself establish No-Helmet status.
# ---------------------------------------------------------------------------
PERSON_CONFIDENCE_THRESHOLD = 0.0  # HOG scores are margins, not probabilities.
HEAD_REGION_FRACTION = 0.30        # top 30% of a person box is the helmet-match region

# ---------------------------------------------------------------------------
# Number-plate detector + OCR
# ---------------------------------------------------------------------------
PLATE_MIN_ASPECT = 2.0     # width / height of a plausible plate region
PLATE_MAX_ASPECT = 6.0
PLATE_MIN_AREA = 300       # pixels, on the *cropped rider/vehicle* image, not the full frame

# Indian plate layouts, e.g. "AP09XY1234", "TN30S0398", "KA05MJ1234".
# L = letter, D = digit. Add more templates here for other countries.
PLATE_TEMPLATES = [
    "LLDDLLDDDD",  # AP 09 XY 1234  (current common format)
    "LLDDLLLDDDD", # state, district, three-letter series, four-digit number
    "LLDDLDDDD",   # TN 30 S 0398   (older / 2-wheeler style)
    "LLDDLLDDD",   # 3-digit tail variant
]

# OCR misreads that are format-aware-correctable (index-position dependent,
# applied only where the template expects the *other* kind of character).
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0"}
DIGIT_TO_LETTER = {v: k for k, v in {"O": "0", "I": "1", "S": "5", "B": "8"}.items()}

# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
VIDEO_FRAME_SKIP = 5  # run detection on every Nth frame (speed vs. thoroughness)
TRACKING_IOU_THRESHOLD = 0.25
TRACKING_CENTER_DISTANCE_FACTOR = 1.5
TRACKING_MAX_MISSED_FRAMES = 12
VIOLATION_COOLDOWN_SECONDS = 30
HELMET_CONFIRMATION_FRAMES = 3
# Conservative single-image gate for explicit class-1 model output; calibrate
# against held-out local video before using this as an enforcement threshold.
NO_HELMET_HIGH_CONFIDENCE_THRESHOLD = 0.90
DEBUG_DETECTIONS = True

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
