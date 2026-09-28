"""SQLite storage for session-scoped rider tracks and confirmed violation events."""
import sqlite3
import json
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_filename TEXT NOT NULL,
    media_type TEXT NOT NULL,             -- 'image' | 'video'
    frame_number INTEGER,                 -- NULL for images
    captured_at TEXT NOT NULL DEFAULT (datetime('now')),
    helmet_status TEXT NOT NULL,          -- 'helmet' | 'no_helmet' | 'unknown'
    confidence REAL,
    helmet_confidence REAL,
    no_helmet_confidence REAL,
    heuristic_score REAL,
    session_id TEXT,
    tracking_id TEXT,
    evidence_json TEXT,
    plate_confidence REAL,
    plate_status TEXT DEFAULT 'not_read',
    plate_candidate_count INTEGER DEFAULT 0,
    ocr_attempted INTEGER DEFAULT 0,
    plate_raw TEXT,
    plate_clean TEXT,
    plate_matched_template INTEGER DEFAULT 0,
    snapshot_path TEXT,                   -- relative path under data/snapshots
    bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER
);
CREATE INDEX IF NOT EXISTS idx_helmet_status ON detections(helmet_status);
CREATE INDEX IF NOT EXISTS idx_captured_at ON detections(captured_at);
CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS processing_sessions (
    session_id TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'processing'
);
CREATE TABLE IF NOT EXISTS violation_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    tracking_id TEXT,
    source_filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    frame_number INTEGER,
    captured_at TEXT NOT NULL DEFAULT (datetime('now')),
    helmet_confidence REAL,
    no_helmet_confidence REAL,
    heuristic_score REAL,
    plate_raw TEXT,
    plate_clean TEXT,
    plate_matched_template INTEGER DEFAULT 0,
    plate_confidence REAL,
    plate_status TEXT DEFAULT 'not_read',
    plate_candidate_count INTEGER DEFAULT 0,
    ocr_attempted INTEGER DEFAULT 0,
    plate_debug TEXT,
    snapshot_path TEXT,
    plate_snapshot_path TEXT,
    bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER,
    evidence_json TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Additive migration: retain all existing detection history.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(detections)")}
        for name, declaration in (("plate_confidence", "REAL"),
                                  ("plate_status", "TEXT DEFAULT 'not_read'"),
                                  ("helmet_confidence", "REAL"),
                                  ("no_helmet_confidence", "REAL"),
                                  ("heuristic_score", "REAL"), ("session_id", "TEXT"),
                                  ("tracking_id", "TEXT"), ("evidence_json", "TEXT"),
                                  ("plate_candidate_count", "INTEGER DEFAULT 0"),
                                  ("ocr_attempted", "INTEGER DEFAULT 0")):
            if name not in existing:
                conn.execute(f"ALTER TABLE detections ADD COLUMN {name} {declaration}")
        event_columns = {row[1] for row in conn.execute("PRAGMA table_info(violation_events)")}
        for name, declaration in (("plate_candidate_count", "INTEGER DEFAULT 0"),
                                  ("ocr_attempted", "INTEGER DEFAULT 0")):
            if name not in event_columns:
                conn.execute(f"ALTER TABLE violation_events ADD COLUMN {name} {declaration}")
        conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_detections_session_track
            ON detections(session_id, tracking_id)
            WHERE session_id IS NOT NULL AND tracking_id IS NOT NULL""")
        conn.execute("UPDATE detections SET confidence=NULL WHERE confidence < 0 OR confidence > 1")
        migrated = conn.execute("SELECT 1 FROM schema_migrations WHERE version=1").fetchone()
        if not migrated:
            # With the bundled one-class model, historical no-helmet scores
            # came from HOG's margin and are not model probabilities.
            conn.execute("UPDATE detections SET confidence=NULL WHERE helmet_status='no_helmet'")
            conn.execute("INSERT INTO schema_migrations(version) VALUES (1)")


def insert_detection(source_filename, media_type, helmet_status, confidence,
                      frame_number=None, plate_raw=None, plate_clean=None,
                      plate_matched_template=False, snapshot_path=None, bbox=None,
                      plate_confidence=None, plate_status="not_read", session_id=None,
                      tracking_id=None, helmet_confidence=None, no_helmet_confidence=None,
                      heuristic_score=None, evidence_json=None):
    bbox = bbox or (None, None, None, None)
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO detections
               (source_filename, media_type, frame_number, helmet_status, confidence,
                plate_raw, plate_clean, plate_matched_template, snapshot_path,
                plate_confidence, plate_status, session_id, tracking_id,
                helmet_confidence, no_helmet_confidence, heuristic_score, evidence_json,
                plate_candidate_count, ocr_attempted,
                bbox_x, bbox_y, bbox_w, bbox_h)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_filename, media_type, frame_number, helmet_status, confidence,
             plate_raw, plate_clean, int(plate_matched_template), snapshot_path,
             plate_confidence, plate_status, session_id, tracking_id, helmet_confidence,
             no_helmet_confidence, heuristic_score, evidence_json, 0, 0, *bbox),
        )
        return cur.lastrowid


def start_session(session_id, source_filename, media_type):
    with get_conn() as conn:
        conn.execute("INSERT INTO processing_sessions(session_id,source_filename,media_type) VALUES(?,?,?)",
                     (session_id, source_filename, media_type))


def finish_session(session_id, status="complete"):
    with get_conn() as conn:
        conn.execute("UPDATE processing_sessions SET finished_at=datetime('now'),status=? WHERE session_id=?",
                     (status, session_id))


def latest_session_id():
    with get_conn() as conn:
        row = conn.execute("SELECT session_id FROM processing_sessions ORDER BY started_at DESC, rowid DESC LIMIT 1").fetchone()
        return row["session_id"] if row else None


def save_track(source_filename, media_type, session_id, tracking_id, helmet_status,
               first_frame, last_frame, helmet_confidence=None, no_helmet_confidence=None,
               evidence_json=None, bbox=None, snapshot_path=None):
    return insert_detection(source_filename, media_type, helmet_status, helmet_confidence,
                            frame_number=first_frame, bbox=bbox, session_id=session_id,
                            snapshot_path=snapshot_path,
                            tracking_id=tracking_id, helmet_confidence=helmet_confidence,
                            no_helmet_confidence=no_helmet_confidence, evidence_json=evidence_json)


def insert_violation_event(session_id, tracking_id, source_filename, media_type,
                           frame_number=None, helmet_confidence=None,
                           no_helmet_confidence=None, heuristic_score=None,
                           plate=None, snapshot_path=None, plate_snapshot_path=None,
                           bbox=None, evidence_json=None):
    plate = plate or {}
    bbox = bbox or (None, None, None, None)
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO violation_events
            (session_id,tracking_id,source_filename,media_type,frame_number,
             helmet_confidence,no_helmet_confidence,heuristic_score,plate_raw,plate_clean,
             plate_matched_template,plate_confidence,plate_status,plate_debug,snapshot_path,
             plate_snapshot_path,plate_candidate_count,ocr_attempted,
             bbox_x,bbox_y,bbox_w,bbox_h,evidence_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id,tracking_id,source_filename,media_type,frame_number,helmet_confidence,
             no_helmet_confidence,heuristic_score,plate.get("plate_raw"),plate.get("plate_clean"),
             int(bool(plate.get("matched_template"))),plate.get("plate_confidence"),
             plate.get("plate_status","not_read"),json.dumps(plate.get("plate_debug")),snapshot_path,
             plate_snapshot_path,int(plate.get("plate_candidate_count",0)),
             int(bool(plate.get("ocr_attempted"))),*bbox,evidence_json))
        return cur.lastrowid


def get_violations(limit=50, offset=0, session_id=None):
    with get_conn() as conn:
        if session_id is None:
            latest = conn.execute("SELECT session_id FROM processing_sessions ORDER BY started_at DESC,rowid DESC LIMIT 1").fetchone()
            session_id = latest["session_id"] if latest else None
        if session_id:
            rows = conn.execute("SELECT *, event_id AS id FROM violation_events WHERE session_id=? ORDER BY event_id DESC LIMIT ? OFFSET ?",
                                (session_id, limit, offset)).fetchall()
        else:
            rows = conn.execute("""SELECT * FROM detections WHERE helmet_status='no_helmet' AND session_id IS NULL
               ORDER BY id DESC LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
        return [dict(r) for r in rows]


def get_recent_detections(limit=50):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_tracks(session_id=None, helmet_status=None):
    """Return tracked people for one upload, optionally filtered by helmet state."""
    with get_conn() as conn:
        if session_id is None:
            session_id = latest_session_id()
        if not session_id:
            return []
        query = "SELECT * FROM detections WHERE session_id=?"
        params = [session_id]
        if helmet_status:
            query += " AND helmet_status=?"
            params.append(helmet_status)
        query += " ORDER BY id ASC"
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_stats(session_id=None):
    with get_conn() as conn:
        if session_id is None:
            session_id = latest_session_id()
        if session_id:
            counts = {r["helmet_status"]: r["c"] for r in conn.execute(
                "SELECT helmet_status,COUNT(*) c FROM detections WHERE session_id=? GROUP BY helmet_status",
                (session_id,)).fetchall()}
            total = sum(counts.values())
            violations = conn.execute("SELECT COUNT(*) c FROM violation_events WHERE session_id=?", (session_id,)).fetchone()["c"]
            plate_count = conn.execute("SELECT COALESCE(SUM(plate_candidate_count),0) c FROM violation_events WHERE session_id=?", (session_id,)).fetchone()["c"]
            ocr_attempts = conn.execute("SELECT COALESCE(SUM(ocr_attempted),0) c FROM violation_events WHERE session_id=?", (session_id,)).fetchone()["c"]
            plates_read = conn.execute("SELECT COUNT(*) c FROM violation_events WHERE session_id=? AND plate_matched_template=1", (session_id,)).fetchone()["c"]
            return _stats_payload(total, violations, counts.get("helmet",0), counts.get("no_helmet",0),
                                  counts.get("unknown",0), plate_count, plates_read, ocr_attempts, session_id)
        total = conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
        violations = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE helmet_status = 'no_helmet'"
        ).fetchone()["c"]
        helmeted = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE helmet_status = 'helmet'"
        ).fetchone()["c"]
        plates_read = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE plate_matched_template = 1"
        ).fetchone()["c"]
        unknown = max(0, total-helmeted-violations)
        compliance_rate = round(100 * helmeted / total, 1) if total and unknown == 0 else None
        if compliance_rate is not None:
            compliance_rate = min(100.0, max(0.0, compliance_rate))
        plates_attempted = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE helmet_status='no_helmet' AND plate_status != 'not_read'"
        ).fetchone()["c"]
        return {
            "total_scanned": total,
            "violations": violations,
            "helmeted": helmeted,
            "plates_read": plates_read,
            "no_helmet": violations,
            "plate_read_rate": round(100 * plates_read / plates_attempted, 1) if plates_attempted else 0.0,
            "compliance_rate": compliance_rate,
            "unknown": unknown, "unique_tracks": total,
            "violation_events": violations, "plates_detected": plates_attempted,
            "ocr_failures": max(0, plates_attempted-plates_read),
            "session_id": None,
        }


def _stats_payload(total, events, helmeted, no_helmet, unknown, plates_detected, plates_read,
                   ocr_attempts=0, session_id=None):
    compliance = round(100*helmeted/total,1) if total and unknown == 0 else None
    return {"total_scanned":total,"unique_tracks":total,"helmeted":helmeted,
            "no_helmet":no_helmet,"unknown":unknown,"violations":events,
            "violation_events":events,"plates_detected":plates_detected,"plates_read":plates_read,
            "ocr_attempts":ocr_attempts,"ocr_failures":max(0,ocr_attempts-plates_read),
            "plate_read_rate":round(100*plates_read/plates_detected,1) if plates_detected else 0.0,
            "compliance_rate":compliance,
            "session_id":session_id}


def clear_all():
    with get_conn() as conn:
        conn.execute("DELETE FROM violation_events")
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM processing_sessions")
