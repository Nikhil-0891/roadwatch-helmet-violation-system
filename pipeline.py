"""Image/video inference, temporal rider association, and violation events."""
import json
import os
import posixpath
import uuid

import cv2

import config
import database
from detection.helmet_detector import HelmetDetector, PersonDetector, classify_riders
from detection.plate_reader import read_best_plate

_helmet_detector = HelmetDetector()
_person_detector = PersonDetector()

STATUS_COLOR = {"helmet": (70, 200, 120), "no_helmet": (50, 50, 230), "unknown": (150, 150, 150)}
STATUS_LABEL = {"helmet": "HELMET", "no_helmet": "NO HELMET", "unknown": "UNKNOWN"}


def detector_status():
    return {"helmet_model_loaded": _helmet_detector.available,
            "direct_two_class": _helmet_detector.direct_two_class,
            "class_names": _helmet_detector.labels,
            "error": _helmet_detector.error}


def _evidence_box(box, image_shape):
    """Person-centered evidence ROI extended down to include nearby vehicle area."""
    x, y, w, h = box
    height, width = image_shape[:2]
    left = max(0, int(x - 0.55 * w))
    right = min(width, int(x + 1.55 * w))
    top = max(0, int(y - 0.05 * h))
    bottom = min(height, int(y + 2.35 * h))
    return left, top, max(1, right-left), max(1, bottom-top)


def _annotate(image, riders):
    out = image.copy()
    for rider in riders:
        x, y, w, h = rider["box"]
        color = STATUS_COLOR.get(rider["status"], STATUS_COLOR["unknown"])
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        conf = rider.get("confidence")
        suffix = f" {conf:.2f}" if conf is not None else ""
        track = f" ID {rider['track_id']}" if rider.get("track_id") is not None else ""
        cv2.putText(out, f"{STATUS_LABEL[rider['status']]}{suffix}{track}",
                    (x, max(16, y-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def _debug_record(session_id, source_filename, frame_number, rider):
    if not config.DEBUG_DETECTIONS:
        return
    os.makedirs(os.path.dirname(config.DETECTION_DEBUG_LOG), exist_ok=True)
    record = {"session_id": session_id, "source": source_filename, "frame": frame_number,
              "tracking_id": rider.get("track_id"), "person_bbox": rider.get("box"),
              "head_bbox": rider.get("head_box"), "helmet_bbox": rider.get("helmet_box"),
              "helmet_overlap_fraction": rider.get("overlap"),
              "helmet_model_confidence": rider.get("confidence"),
              "hog_detector_score_margin": rider.get("detector_score"),
              "model_status": rider.get("model_status", rider.get("status")),
              "status": rider.get("status")}
    with open(config.DETECTION_DEBUG_LOG, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _plate_result(roi):
    try:
        return read_best_plate(roi)
    except Exception as exc:
        return {"plate_raw": None, "plate_clean": None, "matched_template": False,
                "plate_confidence": 0.0, "plate_status": "ocr_failed",
                "plate_debug": {"error": str(exc), "ocr_attempted": False}}


def _handle_violation(image, rider, source_filename, media_type, session_id, frame_number=None):
    x, y, w, h = _evidence_box(rider["box"], image.shape)
    evidence = image[y:y+h, x:x+w]
    plate = _plate_result(evidence) if evidence.size else {
        "plate_status": "candidate_invalid", "plate_debug": {"error": "empty evidence ROI"}}
    plate.setdefault("plate_raw", None)
    plate.setdefault("plate_clean", None)
    plate.setdefault("matched_template", False)
    plate.setdefault("plate_confidence", 0.0)
    plate.setdefault("plate_status", "unreadable")
    plate.setdefault("plate_candidate_count", 0)
    plate.setdefault("ocr_attempted", False)

    token = f"{session_id}_{rider['track_id']}_{frame_number if frame_number is not None else 'image'}".replace(":", "_")
    snapshot_name = f"{token}.jpg"
    full_snapshot = os.path.join(config.SNAPSHOT_DIR, snapshot_name)
    snapshot_rel = posixpath.join("snapshots", snapshot_name) if evidence.size and cv2.imwrite(full_snapshot, evidence) else None

    plate_snapshot_rel = None
    plate_crop = plate.get("crop")
    if plate_crop is not None and getattr(plate_crop, "size", 0):
        os.makedirs(config.PLATE_SNAPSHOT_DIR, exist_ok=True)
        plate_name = f"{token}_plate.jpg"
        if cv2.imwrite(os.path.join(config.PLATE_SNAPSHOT_DIR, plate_name), plate_crop):
            plate_snapshot_rel = posixpath.join("snapshots", "plates", plate_name)

    evidence_meta = _track_summary(rider)
    database.insert_violation_event(
        session_id=session_id, tracking_id=str(rider["track_id"]),
        source_filename=source_filename, media_type=media_type,
        frame_number=frame_number,
        helmet_confidence=rider.get("helmet_confidence"),
        no_helmet_confidence=rider.get("no_helmet_confidence"),
        heuristic_score=rider.get("heuristic_score"), plate=plate,
        snapshot_path=snapshot_rel, plate_snapshot_path=plate_snapshot_rel,
        bbox=rider["box"], evidence_json=json.dumps(evidence_meta))
    rider["plate_clean"] = plate.get("plate_clean") or "Unreadable"
    return plate


def _track_summary(track):
    history = track.get("history", [])
    return {"samples": len(history),
            "helmet_votes": sum(s["status"] == "helmet" for s in history),
            "no_helmet_votes": sum(s["status"] == "no_helmet" for s in history),
            "unknown_votes": sum(s["status"] == "unknown" for s in history),
            "first_frame": track.get("first_frame"), "last_frame": track.get("last_frame")}


def _final_status(track):
    history = track.get("history", [])
    if not history:
        return "unknown"
    if track.get("no_helmet_confirmed"):
        return "no_helmet"
    if any(s["status"] == "helmet" for s in history):
        return "helmet"
    return "unknown"


def _save_tracks(tracks, source_filename, media_type, session_id):
    for track in tracks.values():
        status = _final_status(track)
        helmet_conf = max((s.get("confidence") or 0 for s in track["history"]
                           if s["status"] == "helmet"), default=0) or None
        nohelmet_conf = max((s.get("no_helmet_confidence") or 0 for s in track["history"]
                             if s["status"] == "no_helmet"), default=0) or None
        snapshot_rel = None
        snapshot = track.get("snapshot_image")
        if snapshot is not None and snapshot.size:
            track_dir = os.path.join(config.SNAPSHOT_DIR, "tracks")
            os.makedirs(track_dir, exist_ok=True)
            snapshot_name = f"{session_id}_{track['id']}.jpg"
            if cv2.imwrite(os.path.join(track_dir, snapshot_name), snapshot):
                snapshot_rel = posixpath.join("snapshots", "tracks", snapshot_name)
        database.save_track(source_filename, media_type, session_id, str(track["id"]), status,
                            track["first_frame"], track["last_frame"],
                            helmet_confidence=helmet_conf, no_helmet_confidence=nohelmet_conf,
                            evidence_json=json.dumps(_track_summary(track)), bbox=track["box"],
                            snapshot_path=snapshot_rel)


def _person_snapshot(image, rider):
    """Crop a person with a small margin and a visible box/status marker."""
    height, width = image.shape[:2]
    x, y, box_w, box_h = rider["box"]
    pad_x, pad_y = max(8, int(box_w * 0.12)), max(8, int(box_h * 0.08))
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(width, x + box_w + pad_x), min(height, y + box_h + pad_y)
    crop = image[top:bottom, left:right].copy()
    color = STATUS_COLOR["unknown"]
    cv2.rectangle(crop, (max(0, x-left), max(0, y-top)),
                  (min(right-left-1, x+box_w-left), min(bottom-top-1, y+box_h-top)), color, 2)
    track_id = rider.get("track_id")
    label = f"TRACK {track_id}"
    cv2.putText(crop, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return crop


def _assign_tracks(riders, tracks, next_id, frame_idx, sample_idx):
    available = set(tracks)
    for rider in sorted(riders, key=lambda r: r["box"][2] * r["box"][3], reverse=True):
        x, y, w, h = rider["box"]
        cx, cy = x + w/2, y + h/2
        choices = []
        for track_id in available:
            track = tracks[track_id]
            tx, ty, tw, th = track["box"]
            tcx, tcy = tx + tw/2, ty + th/2
            dist = ((cx-tcx)**2 + (cy-tcy)**2)**0.5
            size = max(w, h, tw, th, 1)
            iou = _box_iou(rider["box"], track["box"])
            if iou >= config.TRACKING_IOU_THRESHOLD or dist <= config.TRACKING_CENTER_DISTANCE_FACTOR * size:
                choices.append((iou, -dist/size, track_id))
        if choices:
            _, _, track_id = max(choices)
            available.remove(track_id)
            track = tracks[track_id]
            rider["track_id"] = track_id
            track["box"] = rider["box"]
            track["last_frame"] = frame_idx
            track["last_sample"] = sample_idx
        else:
            track_id = next_id
            next_id += 1
            rider["track_id"] = track_id
            tracks[track_id] = {"id": track_id, "box": rider["box"],
                                "first_frame": frame_idx, "last_frame": frame_idx,
                                "last_sample": sample_idx, "history": [],
                                "last_event_time": None, "event_emitted": False}
    return next_id


def _update_track(track, rider, frame_idx, fps, image, source_filename, media_type, session_id):
    sample = {"frame": frame_idx, "status": rider["status"],
              "confidence": rider.get("confidence"),
              "no_helmet_confidence": rider.get("no_helmet_confidence"),
              "overlap": rider.get("overlap")}
    track["history"].append(sample)
    x, y, box_w, box_h = rider["box"]
    person_area = box_w * box_h
    if person_area > track.get("snapshot_area", 0):
        track["snapshot_image"] = _person_snapshot(image, rider)
        track["snapshot_area"] = person_area
    if rider["status"] == "no_helmet":
        count = 0
        for previous in reversed(track["history"]):
            if previous["status"] != "no_helmet":
                break
            count += 1
        now = frame_idx / max(float(fps), 1.0)
        cooldown_elapsed = (track["last_event_time"] is None or
                            now - track["last_event_time"] >= config.VIOLATION_COOLDOWN_SECONDS)
        if count >= config.HELMET_CONFIRMATION_FRAMES and cooldown_elapsed:
            enriched = dict(rider)
            enriched["history"] = track["history"]
            _handle_violation(image, enriched, source_filename, media_type, session_id, frame_idx)
            track["last_event_time"] = now
            track["event_emitted"] = True
            track["no_helmet_confirmed"] = True


def process_image(image_path, source_filename=None, session_id=None):
    source_filename = source_filename or os.path.basename(image_path)
    session_id = session_id or uuid.uuid4().hex
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    riders = classify_riders(image, _helmet_detector, _person_detector)
    tracks = {}
    for track_id, rider in enumerate(riders, 1):
        if (rider["status"] == "no_helmet" and
                (rider.get("no_helmet_confidence") or 0) < config.NO_HELMET_HIGH_CONFIDENCE_THRESHOLD):
            rider["model_status"] = "no_helmet"
            rider["status"] = "unknown"
        rider["track_id"] = track_id
        rider["confidence"] = rider.get("confidence") if rider["status"] == "helmet" else None
        track = {"id": track_id, "box": rider["box"], "first_frame": None,
                 "last_frame": None, "history": []}
        track["history"].append({"status": rider["status"], "confidence": rider.get("confidence"),
                                 "no_helmet_confidence": rider.get("no_helmet_confidence")})
        track["snapshot_image"] = _person_snapshot(image, rider)
        tracks[track_id] = track
        _debug_record(session_id, source_filename, None, rider)
        if (rider["status"] == "no_helmet" and
                (rider.get("no_helmet_confidence") or 0) >= config.NO_HELMET_HIGH_CONFIDENCE_THRESHOLD):
            track["no_helmet_confirmed"] = True
            _handle_violation(image, rider, source_filename, "image", session_id)
    _save_tracks(tracks, source_filename, "image", session_id)
    out_name = f"{os.path.splitext(source_filename)[0]}_annotated.jpg"
    if not cv2.imwrite(os.path.join(config.OUTPUT_DIR, out_name), _annotate(image, riders)):
        raise ValueError("Could not save annotated image output")
    counts = {status: sum(r["status"] == status for r in riders)
              for status in ("helmet", "no_helmet", "unknown")}
    return {"session_id": session_id, "source_filename": source_filename,
            "tracks_found": len(riders), "helmeted": counts["helmet"],
            "no_helmet": counts["no_helmet"], "unknown": counts["unknown"],
            "violations": sum(1 for r in riders if r["status"] == "no_helmet"),
            "annotated_output": os.path.join("outputs", out_name)}


def process_video(video_path, source_filename=None, frame_skip=config.VIDEO_FRAME_SKIP, session_id=None,
                 progress_callback=None):
    if frame_skip < 1:
        raise ValueError("VIDEO_FRAME_SKIP must be at least 1")
    source_filename = source_filename or os.path.basename(video_path)
    session_id = session_id or uuid.uuid4().hex
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_name = f"{os.path.splitext(source_filename)[0]}_annotated.mp4"
    output_path = os.path.join(config.OUTPUT_DIR, output_name)
    writer = None
    frame_idx = sample_idx = 0
    tracks, retired_tracks, next_id, last_riders = {}, [], 1, []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_skip == 0:
                expired = [tid for tid, tr in tracks.items()
                           if sample_idx - tr["last_sample"] > config.TRACKING_MAX_MISSED_FRAMES]
                for tid in expired:
                    retired_tracks.append(tracks.pop(tid))
                riders = classify_riders(frame, _helmet_detector, _person_detector)
                next_id = _assign_tracks(riders, tracks, next_id, frame_idx, sample_idx)
                for rider in riders:
                    track = tracks[rider["track_id"]]
                    _update_track(track, rider, frame_idx, fps, frame, source_filename,
                                  "video", session_id)
                    _debug_record(session_id, source_filename, frame_idx, rider)
                last_riders = riders
                sample_idx += 1
            annotated = _annotate(frame, last_riders)
            if writer is None:
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]), True)
                if not writer.isOpened():
                    raise ValueError("Could not create annotated video output")
            writer.write(annotated)
            frame_idx += 1
            if progress_callback and (frame_idx % max(1, int(fps)) == 0):
                progress_callback(frame_idx, total_frames)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frame_idx == 0:
        raise ValueError("Video contains no readable frames")
    if progress_callback:
        progress_callback(frame_idx, total_frames, complete=True)
    all_tracks = {str(t["id"]): t for t in retired_tracks + list(tracks.values())}
    _save_tracks(all_tracks, source_filename, "video", session_id)
    counts = {status: sum(_final_status(t) == status for t in all_tracks.values())
              for status in ("helmet", "no_helmet", "unknown")}
    with database.get_conn() as conn:
        violations = conn.execute("SELECT COUNT(*) FROM violation_events WHERE session_id=?",
                                  (session_id,)).fetchone()[0]
    return {"session_id": session_id, "source_filename": source_filename,
            "frames_processed": frame_idx, "tracks_found": len(all_tracks),
            "helmeted": counts["helmet"], "no_helmet": counts["no_helmet"],
            "unknown": counts["unknown"], "violations": violations,
            "annotated_output": os.path.join("outputs", output_name)}


def _box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw, ih = max(0, min(ax+aw, bx+bw)-max(ax,bx)), max(0, min(ay+ah, by+bh)-max(ay,by))
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0
