"""
Helmet / No-Helmet classification.

The bundled YOLO model has one class: Helmet. HOG supplies generic pedestrian
boxes, not motorcycle-rider detections. A helmet box overlapping a person's
estimated head region is positive helmet evidence. No overlap is unknown,
because a missed helmet detection is not proof of an uncovered head. A
matching two-class model can produce direct Helmet/No-Helmet detections;
video violations then require repeated explicit No-Helmet outputs.
"""
import cv2
import numpy as np
import math

import config


class HelmetDetector:
    """Thin wrapper around the existing Darknet YOLOv3 model."""

    def __init__(self, cfg_path=config.HELMET_CFG, weights_path=config.HELMET_WEIGHTS,
                 names_path=config.HELMET_NAMES):
        self.available = False
        self.error = None
        self.labels = []
        self.direct_two_class = False
        self.net = None
        self.output_layers = None

        try:
            with open(names_path) as f:
                self.labels = [line.strip() for line in f if line.strip()]
        except OSError as exc:
            self.error = f"Could not read class names file '{names_path}': {exc}"
            return

        self.direct_two_class = any(lbl.lower() in config.NO_HELMET_LABELS for lbl in self.labels)

        import os
        if not os.path.isfile(weights_path):
            self.error = (
                f"Weights file not found at '{weights_path}'. "
                "Detection will report riders as 'unknown' until you add it - see README."
            )
            return

        try:
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            layer_names = self.net.getLayerNames()
            try:
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
            except AttributeError:
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            self.available = True
        except cv2.error as exc:
            self.error = f"OpenCV could not load the model ({cfg_path} / {weights_path}): {exc}"

    def detect(self, image):
        """Return a list of dicts: {box:(x,y,w,h), label, confidence}. [] if unavailable."""
        if not self.available:
            return []

        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    continue
                if confidence > config.HELMET_CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (cx, cy, w, h) = box.astype("int")
                    x, y = int(cx - w / 2), int(cy - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, config.HELMET_CONFIDENCE_THRESHOLD, config.HELMET_NMS_THRESHOLD)
        results = []
        for i in np.array(idxs).flatten() if len(boxes) else []:
            label = self.labels[class_ids[i]] if class_ids[i] < len(self.labels) else "object"
            results.append({"box": tuple(boxes[i]), "label": label, "confidence": confidences[i]})
        return results


class PersonDetector:
    """OpenCV's built-in HOG + SVM pedestrian detector. Ships with opencv-python - no download."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image):
        rects, weights = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
        candidates = []
        for (box, weight) in zip(rects, weights):
            # OpenCV HOG returns an unbounded SVM margin, not a probability.
            # Keep it out of the confidence field; it must never be shown as %.
            candidates.append({"box": tuple(int(v) for v in box), "confidence": None,
                               "detector_score": float(weight) if math.isfinite(float(weight)) else None})
        # HOG may return nested/overlapping windows for the same pedestrian.
        candidates.sort(key=lambda item: item["detector_score"] if item["detector_score"] is not None else -math.inf,
                        reverse=True)
        results = []
        for candidate in candidates:
            if any(_box_iou(candidate["box"], kept["box"]) >= 0.55 for kept in results):
                continue
            results.append(candidate)
        return results


def _iou_overlap_fraction(head_box, helmet_box):
    """Fraction of the *helmet* box that falls inside the head region - robust to the
    head-region estimate being loose, since we only need the helmet to be roughly there."""
    hx, hy, hw, hh = head_box
    ex, ey, ew, eh = helmet_box
    ix1, iy1 = max(hx, ex), max(hy, ey)
    ix2, iy2 = min(hx + hw, ex + ew), min(hy + hh, ey + eh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    helmet_area = ew * eh
    return intersection / helmet_area if helmet_area > 0 else 0.0


def _box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def classify_riders(image, helmet_detector: HelmetDetector, person_detector: PersonDetector):
    """
    Returns a list of dicts, one per detected rider:
      {box, head_box, helmet_box, overlap, status, confidence, detector_score}
    """
    helmet_detections = helmet_detector.detect(image)

    # Direct 2-class mode: the model already says Helmet vs No-Helmet per box,
    # no person detector / overlap heuristic needed.
    if helmet_detector.direct_two_class:
        riders = []
        for det in helmet_detections:
            status = "no_helmet" if det["label"].lower() in config.NO_HELMET_LABELS else "helmet"
            riders.append({"box": det["box"], "head_box": det["box"], "helmet_box": det["box"] if status == "helmet" else None,
                            "overlap": 1.0 if status == "helmet" else 0.0, "status": status,
                            "confidence": det["confidence"] if status == "helmet" else None,
                            "detector_score": None,
                            "no_helmet_confidence": det["confidence"] if status == "no_helmet" else None})
        return riders

    people = person_detector.detect(image)
    riders = []
    for person in people:
        (x, y, w, h) = person["box"]
        head_box = (x, y, w, max(1, int(h * config.HEAD_REGION_FRACTION)))

        if not helmet_detector.available:
            riders.append({"box": person["box"], "head_box": head_box, "helmet_box": None,
                            "overlap": 0.0, "status": "unknown", "confidence": None,
                            "detector_score": person.get("detector_score"), "no_helmet_confidence": None})
            continue

        best_overlap, best_conf = 0.0, 0.0
        for det in helmet_detections:
            overlap = _iou_overlap_fraction(head_box, det["box"])
            if overlap > best_overlap:
                best_overlap, best_conf = overlap, det["confidence"]

        if best_overlap >= 0.2:
            matched = next((d for d in helmet_detections if d["confidence"] == best_conf and
                            _iou_overlap_fraction(head_box, d["box"]) == best_overlap), None)
            riders.append({"box": person["box"], "head_box": head_box,
                            "helmet_box": matched["box"] if matched else None, "overlap": best_overlap,
                            "status": "helmet", "confidence": best_conf,
                            "detector_score": person.get("detector_score"), "no_helmet_confidence": None})
        else:
            # The single-class model cannot confirm an uncovered head. Lack of
            # a helmet box is weak negative evidence, so preserve uncertainty.
            riders.append({"box": person["box"], "head_box": head_box, "helmet_box": None,
                            "overlap": best_overlap, "status": "unknown", "confidence": None,
                            "detector_score": person.get("detector_score"), "no_helmet_confidence": None})
    return riders
