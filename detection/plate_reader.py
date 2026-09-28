"""
Number-plate localization + OCR.

Two localization strategies are tried in order (fastest/most-precise first):
  1. Haar cascade  - ships inside opencv-python, zero extra downloads, good
     precision on roughly frontal/rear plates.
  2. Contour search - classic edge-detection + rectangular-contour approach.
     Lower precision but higher recall, used as a fallback when the cascade
     finds nothing (angled plates, unusual lighting, etc).

Once a candidate region is found, pytesseract reads it, and
`clean_plate_text` turns the raw OCR string into a validated vehicle number
using position-aware letter/digit correction against known plate templates
(see config.PLATE_TEMPLATES). Unvalidated text remains diagnostic OCR data
and is never returned as a vehicle number.

Swap in a trained plate-detector model later (see README) by replacing
`locate_plate_candidates` - `read_best_plate` doesn't care how the box was
found.
"""
import os
import re
import cv2
import numpy as np
import pytesseract

import config

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def locate_plate_candidates(bgr_image):
    """Return a list of (x, y, w, h) boxes that might contain a plate, best guess first."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    boxes = _haar_candidates(gray)
    if not boxes:
        boxes = _contour_candidates(gray)

    # Clip to image bounds and drop degenerate boxes.
    clean = []
    for (x, y, w, h) in boxes:
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_img - x), min(h, h_img - y)
        if w > 0 and h > 0:
            clean.append((x, y, w, h))
    return clean


def _haar_candidates(gray):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    detections = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 13))
    return [tuple(map(int, box)) for box in detections]


def _contour_candidates(gray):
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        area = w * h
        if config.PLATE_MIN_ASPECT <= aspect <= config.PLATE_MAX_ASPECT and area >= config.PLATE_MIN_AREA:
            candidates.append((x, y, w, h))
    return candidates


def _preprocess_for_ocr(crop_gray):
    """Upscale + denoise + binarize a small plate crop so tesseract has a fighting chance."""
    scale = max(1, 200 // max(crop_gray.shape[0], 1))
    resized = cv2.resize(crop_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    denoised = cv2.bilateralFilter(enhanced, 7, 40, 40)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def _ocr_text(processed_crop):
    tess_config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        data = pytesseract.image_to_data(processed_crop, config=tess_config,
                                         output_type=pytesseract.Output.DICT)
        pieces, scores = [], []
        for word, score in zip(data.get("text", []), data.get("conf", [])):
            cleaned = re.sub(r"[^A-Z0-9]", "", word.upper())
            try:
                value = float(score)
                if cleaned and value >= 0:
                    pieces.append(cleaned)
                    scores.append(value / 100.0)
            except (ValueError, TypeError):
                continue
        return "".join(pieces), (sum(scores) / len(scores) if scores else 0.0), None
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError, OSError) as exc:
        return "", 0.0, str(exc)


def clean_plate_text(raw_text):
    """
    Try to match raw_text against a known plate template, applying
    position-aware O/0, I/1, S/5, B/8-style correction only where the
    template says that slot should be the other kind of character.
    Returns (clean_text, matched_template: bool).
    """
    for template in config.PLATE_TEMPLATES:
        if len(raw_text) != len(template):
            continue
        candidate = list(raw_text)
        ok = True
        for i, slot in enumerate(template):
            ch = candidate[i]
            if slot == "L" and ch.isdigit():
                ch = config.DIGIT_TO_LETTER.get(ch, ch)
            elif slot == "D" and ch.isalpha():
                ch = config.LETTER_TO_DIGIT.get(ch, ch)
            if slot == "L" and not ch.isalpha():
                ok = False
                break
            if slot == "D" and not ch.isdigit():
                ok = False
                break
            candidate[i] = ch
        if ok:
            return "".join(candidate), True

    # No template matched exactly - still hand back the best-effort string.
    return None, False


def read_best_plate(bgr_image):
    """
    Run the full localize -> OCR -> clean pipeline on an image (typically a
    crop around a single flagged rider/vehicle, not a whole street scene).
    Returns a dict; plate_raw is None if nothing readable was found at all.
    """
    if bgr_image is None or not getattr(bgr_image, "size", 0):
        return {"plate_raw": None, "plate_clean": None, "matched_template": False,
                "bbox": None, "plate_confidence": 0.0, "plate_status": "candidate_invalid",
                "plate_candidate_count": 0, "ocr_attempted": False,
                "plate_debug": {"candidate_count": 0, "ocr_attempted": False}}
    candidates = locate_plate_candidates(bgr_image)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    best = {"plate_raw": None, "plate_clean": None, "matched_template": False,
            "bbox": None, "plate_confidence": 0.0,
            "plate_status": "no_plate_candidate" if not candidates else "candidate_invalid",
            "plate_candidate_count": len(candidates), "ocr_attempted": False,
            "plate_debug": {"candidate_count": len(candidates), "ocr_attempted": False,
                            "candidate_results": []}}
    best_len = 0

    for (x, y, w, h) in candidates:
        crop = gray[y:y + h, x:x + w]
        aspect = w / max(float(h), 1.0)
        if (crop.size == 0 or w*h < config.PLATE_MIN_AREA or
                not config.PLATE_MIN_ASPECT <= aspect <= config.PLATE_MAX_ASPECT):
            best["plate_debug"]["candidate_results"].append({"bbox": (x,y,w,h), "stage": "candidate_invalid"})
            continue
        processed = _preprocess_for_ocr(crop)
        raw, ocr_confidence, error = _ocr_text(processed)
        best["ocr_attempted"] = True
        best["plate_debug"]["ocr_attempted"] = True
        stage = "ocr_failed" if error or len(raw) < 4 else "ocr_succeeded"
        best["plate_debug"]["candidate_results"].append({"bbox": (x,y,w,h), "stage": stage,
            "raw_text": raw, "ocr_confidence": ocr_confidence, "error": error})
        if error:
            best["plate_status"] = "ocr_unavailable"
            best["plate_debug"]["error"] = error
            continue
        if len(raw) < 4:
            best["plate_status"] = "ocr_failed"
            continue
        clean, matched = clean_plate_text(raw)
        if not matched:
            best["plate_status"] = "validation_failed"
            if len(raw) > best_len:
                best["plate_raw"] = raw
                best["bbox"] = (x,y,w,h)
                best["plate_debug"]["unvalidated_text"] = raw
                best_len = len(raw)
            continue

        # Prefer a template match; among non-matches, prefer the longer read.
        is_better = (matched and not best["matched_template"]) or (
            matched == best["matched_template"] and len(raw) > best_len
        )
        if is_better:
            best = {"plate_raw": raw, "plate_clean": clean, "matched_template": True,
                    "bbox": (x, y, w, h), "plate_confidence": min(1.0, max(0.0, ocr_confidence)),
                    "plate_status": "validated", "crop": bgr_image[y:y+h, x:x+w].copy(),
                    "plate_candidate_count": len(candidates), "ocr_attempted": True,
                    "plate_debug": best["plate_debug"]}
            best_len = len(raw)

    return best
