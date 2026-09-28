"""
Flask entry point.

    python app.py

then open http://localhost:5000
"""
import os
import uuid
import atexit
from pathlib import Path
from threading import Lock

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, abort
from pathlib import PurePosixPath

import config
import database
import pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB, video files add up
processing_progress = {"active": False, "phase": "", "percent": None, "frame": 0, "total_frames": 0}
progress_lock = Lock()

database.init_db()


def clear_runtime_data():
    """Keep uploads and analysis results only for this app process lifetime."""
    data_root = (Path(config.BASE_DIR) / "data").resolve()
    for directory in (config.UPLOAD_DIR, config.SNAPSHOT_DIR, config.OUTPUT_DIR):
        root = Path(directory).resolve()
        if root.parent != data_root or root.name not in {"uploads", "snapshots", "outputs"}:
            raise RuntimeError(f"Refusing to clear unexpected runtime directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        for item in root.rglob("*"):
            if item.is_file() or item.is_symlink():
                item.unlink()
    database.clear_all()


# Remove stale results left by a prior run, then discard this run's data on exit.
clear_runtime_data()
atexit.register(clear_runtime_data)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    return jsonify(pipeline.detector_status())


@app.route("/api/stats")
def api_stats():
    return jsonify(database.get_stats(request.args.get("session_id")))


@app.route("/api/violations")
def api_violations():
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return jsonify({"error": "limit and offset must be integers"}), 400
    rows = database.get_violations(limit=limit, offset=offset,
                                   session_id=request.args.get("session_id"))
    for row in rows:
        if row.get("snapshot_path"):
            relative_path = snapshot_relative_path(row["snapshot_path"])
            if relative_path:
                row["snapshot_url"] = url_for("snapshot_media", filename=relative_path)
        if row.get("plate_snapshot_path"):
            relative_path = snapshot_relative_path(row["plate_snapshot_path"])
            if relative_path:
                row["plate_snapshot_url"] = url_for("snapshot_media", filename=relative_path)
    return jsonify(rows)


@app.route("/api/tracks")
def api_tracks():
    helmet_status = request.args.get("status") or None
    if helmet_status not in {None, "helmet", "no_helmet", "unknown"}:
        return jsonify({"error": "status must be helmet, no_helmet, or unknown"}), 400
    rows = database.get_tracks(request.args.get("session_id"), helmet_status)
    for row in rows:
        if row.get("snapshot_path"):
            relative_path = snapshot_relative_path(row["snapshot_path"])
            if relative_path:
                row["snapshot_url"] = url_for("snapshot_media", filename=relative_path)
    return jsonify(rows)


@app.route("/api/progress")
def api_progress():
    with progress_lock:
        return jsonify(dict(processing_progress))


def snapshot_relative_path(stored_path):
    """Map known DB snapshot paths to a path rooted only inside SNAPSHOT_DIR."""
    normalized = str(stored_path).replace("\\", "/").lstrip("/")
    for prefix in ("data/snapshots/", "snapshots/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path.as_posix()


@app.route("/snapshots/<path:filename>")
def snapshot_media(filename):
    relative_path = snapshot_relative_path(filename)
    if not relative_path:
        abort(404)
    return send_from_directory(config.SNAPSHOT_DIR, relative_path, conditional=True)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    is_image = ext in config.ALLOWED_IMAGE_EXT
    is_video = ext in config.ALLOWED_VIDEO_EXT
    if not (is_image or is_video):
        return jsonify({"error": f"Unsupported file type '{ext}'. Use an image or a video."}), 400

    with progress_lock:
        processing_progress.update(active=True, phase="Saving upload", percent=None,
                                   frame=0, total_frames=0)
    safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
    save_path = os.path.join(config.UPLOAD_DIR, safe_name)
    file.save(save_path)
    session_id = uuid.uuid4().hex
    media_type = "image" if is_image else "video"
    database.start_session(session_id, file.filename, media_type)

    try:
        if is_image:
            with progress_lock:
                processing_progress.update(phase="Analyzing image", percent=None)
            result = pipeline.process_image(save_path, source_filename=file.filename, session_id=session_id)
        else:
            with progress_lock:
                processing_progress.update(phase="Analyzing video", percent=0)

            def update_video_progress(frame, total_frames, complete=False):
                percent = round(100 * frame / total_frames) if total_frames > 0 else None
                with progress_lock:
                    processing_progress.update(phase="Analyzing video", percent=100 if complete else min(percent, 99) if percent is not None else None,
                                               frame=frame, total_frames=total_frames)

            result = pipeline.process_video(save_path, source_filename=file.filename,
                                            session_id=session_id,
                                            progress_callback=update_video_progress)
        database.finish_session(session_id)
    except Exception as exc:  # noqa: BLE001 - surface the real error to the dashboard
        database.finish_session(session_id, "failed")
        with progress_lock:
            processing_progress.update(active=False, phase="Failed")
        return jsonify({"error": str(exc)}), 500

    with progress_lock:
        processing_progress.update(active=False, phase="Complete", percent=100)

    result["annotated_output_url"] = url_for("media", filename=result["annotated_output"].replace("\\", "/"))
    result["media_type"] = media_type
    return jsonify(result)


@app.route("/media/<path:filename>")
def media(filename):
    filename = filename.replace("\\", "/")
    return send_from_directory(os.path.join(config.BASE_DIR, "data"), filename)


if __name__ == "__main__":
    # Disable the debug reloader so startup/shutdown cleanup runs once per server.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
