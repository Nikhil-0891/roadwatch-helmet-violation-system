"""Local smoke checks for RoadWatch; run with `python test_system.py`."""
import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import config
import database
from detection.helmet_detector import _iou_overlap_fraction
from detection.helmet_detector import classify_riders
from detection.plate_reader import clean_plate_text, locate_plate_candidates, _ocr_text, _preprocess_for_ocr
from pipeline import _box_iou, detector_status, _helmet_detector
from app import app


def client_for_tests():
    return app.test_client()


class RoadWatchChecks(unittest.TestCase):
    def test_model_load_and_inference_contract(self):
        status = detector_status()
        if status["helmet_model_loaded"]:
            detections = _helmet_detector.detect(np.zeros((320, 480, 3), dtype=np.uint8))
            for detection in detections:
                self.assertGreaterEqual(detection["confidence"], 0.0)
                self.assertLessEqual(detection["confidence"], 1.0)
        else:
            self.assertTrue(status["error"])
        with open(config.HELMET_NAMES, encoding="utf-8") as names_file:
            labels = {line.strip().lower() for line in names_file if line.strip()}
        self.assertEqual(status["direct_two_class"], bool(labels & config.NO_HELMET_LABELS))

    def test_indian_plate_validation(self):
        self.assertEqual(clean_plate_text("AP09XY1234"), ("AP09XY1234", True))
        self.assertEqual(clean_plate_text("KA01AB1234"), ("KA01AB1234", True))
        self.assertEqual(clean_plate_text("TS09AB1234"), ("TS09AB1234", True))
        self.assertEqual(clean_plate_text("TN30S0398"), ("TN30S0398", True))
        self.assertEqual(clean_plate_text("NOTAPLATE"), (None, False))

    def test_plate_detector_on_blank_frame(self):
        blank = np.zeros((160, 320, 3), dtype=np.uint8)
        self.assertEqual(locate_plate_candidates(blank), [])
        from detection.plate_reader import read_best_plate
        result = read_best_plate(blank)
        self.assertEqual(result["plate_status"], "no_plate_candidate")

    def test_tesseract_independently(self):
        plate = np.full((120, 700), 255, dtype=np.uint8)
        cv2.putText(plate, "KA01AB1234", (15, 92), cv2.FONT_HERSHEY_SIMPLEX,
                    2.1, (0, 0, 0), 4, cv2.LINE_AA)
        raw, confidence, error = _ocr_text(_preprocess_for_ocr(plate))
        self.assertIsNone(error, f"Tesseract failed at {config.TESSERACT_CMD}")
        self.assertTrue(raw)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_tracking_geometry(self):
        self.assertAlmostEqual(_box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(_box_iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)
        self.assertAlmostEqual(_iou_overlap_fraction((0, 0, 20, 20), (5, 5, 10, 10)), 1.0)

    def test_single_class_absence_is_unknown(self):
        class FakeHelmet:
            available = True
            direct_two_class = False
            def detect(self, image): return []
        class FakePeople:
            def detect(self, image):
                return [{"box": (10, 10, 40, 100), "confidence": None, "detector_score": 1.4}]
        rider = classify_riders(np.zeros((140, 80, 3), dtype=np.uint8), FakeHelmet(), FakePeople())[0]
        self.assertEqual(rider["status"], "unknown")
        self.assertIsNone(rider["confidence"])
        self.assertIsNone(rider["no_helmet_confidence"])

    def test_single_class_positive_helmet_overlap(self):
        class FakeHelmet:
            available = True
            direct_two_class = False
            def detect(self, image):
                return [{"box": (12, 12, 22, 18), "label": "Helmet", "confidence": 0.83}]
        class FakePeople:
            def detect(self, image):
                return [{"box": (10, 10, 40, 100), "confidence": None, "detector_score": 2.0}]
        rider = classify_riders(np.zeros((140, 80, 3), dtype=np.uint8), FakeHelmet(), FakePeople())[0]
        self.assertEqual(rider["status"], "helmet")
        self.assertAlmostEqual(rider["confidence"], 0.83)
        self.assertIsNotNone(rider["helmet_box"])

    def test_additive_db_and_snapshot_creation(self):
        import pipeline
        with tempfile.TemporaryDirectory() as folder:
            old_db, old_snapshot = config.DB_PATH, config.SNAPSHOT_DIR
            config.DB_PATH = os.path.join(folder, "test.db")
            config.SNAPSHOT_DIR = folder
            try:
                database.init_db()
                database.start_session("test-session", "unit.jpg", "image")
                with patch.object(pipeline, "read_best_plate", return_value={
                    "plate_clean": None, "plate_raw": None, "matched_template": False,
                    "plate_confidence": 0.0, "plate_status": "unreadable",
                }):
                    pipeline._handle_violation(np.full((100, 100, 3), 80, dtype=np.uint8),
                                               {"box": (5, 5, 40, 40), "confidence": None,
                                                "no_helmet_confidence": 0.99, "track_id": 1,
                                                "history": []}, "unit.jpg", "image", "test-session")
                rows = database.get_violations()
                self.assertEqual(len(rows), 1)
                self.assertTrue(os.path.isfile(os.path.join(folder, os.path.basename(rows[0]["snapshot_path"]))))
                self.assertIsNone(rows[0]["plate_clean"])
            finally:
                config.DB_PATH, config.SNAPSHOT_DIR = old_db, old_snapshot

    def test_legacy_database_migrates_without_dropping_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            old_db = config.DB_PATH
            config.DB_PATH = os.path.join(folder, "legacy.db")
            try:
                import sqlite3
                conn = sqlite3.connect(config.DB_PATH)
                conn.execute("""CREATE TABLE detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, source_filename TEXT NOT NULL,
                    media_type TEXT NOT NULL, frame_number INTEGER,
                    captured_at TEXT NOT NULL DEFAULT (datetime('now')),
                    helmet_status TEXT NOT NULL, confidence REAL, plate_raw TEXT,
                    plate_clean TEXT, plate_matched_template INTEGER DEFAULT 0,
                    snapshot_path TEXT, bbox_x INTEGER,bbox_y INTEGER,bbox_w INTEGER,bbox_h INTEGER)""")
                conn.execute("INSERT INTO detections(source_filename,media_type,helmet_status,confidence) VALUES('old.mp4','video','no_helmet',0.8)")
                conn.commit(); conn.close()
                database.init_db()
                row = database.get_recent_detections(1)[0]
                self.assertEqual(row["source_filename"], "old.mp4")
                self.assertIsNone(row["confidence"])
                self.assertIn("tracking_id", row)
            finally:
                config.DB_PATH = old_db

    def test_dashboard_api(self):
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/api/status").status_code, 200)
        self.assertEqual(client.get("/api/stats").status_code, 200)
        self.assertEqual(client.get("/api/violations?limit=bad").status_code, 400)
        rows = client.get("/api/violations?limit=1").get_json()
        if rows and rows[0].get("snapshot_url"):
            with client.get(rows[0]["snapshot_url"]) as image:
                self.assertEqual(image.status_code, 200)
                self.assertTrue(image.data)

    def test_snapshot_route_serves_db_snapshot_and_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            old_db, old_snapshot = config.DB_PATH, config.SNAPSHOT_DIR
            config.DB_PATH = os.path.join(folder, "viewer.db")
            config.SNAPSHOT_DIR = os.path.join(folder, "snapshots")
            os.makedirs(config.SNAPSHOT_DIR)
            try:
                database.init_db()
                database.start_session("viewer-session", "clip.mp4", "video")
                image = np.full((90, 160, 3), (40, 120, 220), dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                filename = "evidence.jpg"
                with open(os.path.join(config.SNAPSHOT_DIR, filename), "wb") as snapshot:
                    snapshot.write(encoded.tobytes())
                database.insert_violation_event(
                    "viewer-session", 4, "clip.mp4", "video", frame_number=130,
                    no_helmet_confidence=0.97, snapshot_path=f"snapshots/{filename}",
                    plate={"plate_status": "unreadable", "plate_confidence": 0.0},
                )
                client = app.test_client()
                record = client.get("/api/violations?session_id=viewer-session").get_json()[0]
                self.assertEqual(record["tracking_id"], "4")
                with client.get(record["snapshot_url"]) as image_response:
                    self.assertEqual(image_response.status_code, 200)
                    self.assertEqual(image_response.data, encoded.tobytes())
                with client.get("/snapshots/missing.jpg") as missing_response:
                    self.assertEqual(missing_response.status_code, 404)
                with client.get("/snapshots/%2e%2e/viewer.db") as traversal_response:
                    self.assertEqual(traversal_response.status_code, 404)
                self.assertEqual(client.get("/").status_code, 200)
            finally:
                config.DB_PATH, config.SNAPSHOT_DIR = old_db, old_snapshot

    def test_video_violation_deduplication(self):
        import pipeline
        with tempfile.TemporaryDirectory() as folder:
            old_db, old_snapshot, old_output = config.DB_PATH, config.SNAPSHOT_DIR, config.OUTPUT_DIR
            old_skip, old_cooldown = config.VIDEO_FRAME_SKIP, config.VIOLATION_COOLDOWN_SECONDS
            config.DB_PATH = os.path.join(folder, "video.db")
            config.SNAPSHOT_DIR = os.path.join(folder, "snapshots")
            config.OUTPUT_DIR = os.path.join(folder, "outputs")
            os.makedirs(config.SNAPSHOT_DIR)
            os.makedirs(config.OUTPUT_DIR)
            source = os.path.join(folder, "input.avi")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
            if not writer.isOpened():
                config.DB_PATH, config.SNAPSHOT_DIR, config.OUTPUT_DIR = old_db, old_snapshot, old_output
                self.skipTest("This OpenCV build has no MJPG video writer")
            for _ in range(100):
                writer.write(np.full((120, 160, 3), 100, dtype=np.uint8))
            writer.release()
            database.init_db()
            database.start_session("video-session", "unit.avi", "video")
            config.VIDEO_FRAME_SKIP, config.VIOLATION_COOLDOWN_SECONDS = 1, 30
            try:
                fake_rider = {"box": (30, 10, 40, 80), "head_box": (30, 10, 40, 24),
                              "status": "no_helmet", "confidence": None,
                              "no_helmet_confidence": 0.97}
                with patch.object(pipeline, "classify_riders", return_value=[fake_rider]), \
                     patch.object(pipeline, "read_best_plate", return_value={
                         "plate_clean": None, "plate_raw": None, "matched_template": False,
                         "plate_confidence": 0.0, "plate_status": "unreadable",
                     }):
                    result = pipeline.process_video(source, "unit.avi", frame_skip=1,
                                                    session_id="video-session")
                self.assertEqual(result["frames_processed"], 100)
                self.assertEqual(result["tracks_found"], 1)
                self.assertEqual(result["violations"], 1)
                self.assertEqual(len(database.get_violations()), 1)
                self.assertEqual(result["unknown"], 0)
                self.assertEqual(database.get_stats("video-session")["unique_tracks"], 1)
                self.assertEqual(database.get_stats("video-session")["violation_events"], 1)
                self.assertEqual(client_for_tests().get("/api/stats?session_id=video-session").get_json()["unique_tracks"], 1)
                with database.get_conn() as conn:
                    self.assertEqual(conn.execute("SELECT COUNT(DISTINCT tracking_id) FROM detections WHERE session_id='video-session'").fetchone()[0], 1)
                    self.assertEqual(conn.execute("SELECT COUNT(DISTINCT tracking_id) FROM violation_events WHERE session_id='video-session'").fetchone()[0], 1)
            finally:
                config.DB_PATH, config.SNAPSHOT_DIR, config.OUTPUT_DIR = old_db, old_snapshot, old_output
                config.VIDEO_FRAME_SKIP, config.VIOLATION_COOLDOWN_SECONDS = old_skip, old_cooldown


if __name__ == "__main__":
    unittest.main(verbosity=2)
