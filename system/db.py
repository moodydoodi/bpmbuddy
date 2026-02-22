"""
Telemetry database interface for BPM-Buddy.
Utilizes SQLite in Write-Ahead Logging (WAL) mode to allow concurrent 
read/write operations between the backend loop and the Streamlit dashboard.
"""
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB_PATH = Path("outputs/live_session.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telemetry (
  ts REAL NOT NULL,
  mode TEXT,
  status TEXT,
  activity TEXT,
  energy REAL,
  cadence_raw REAL,
  cadence_smooth REAL,
  conf REAL,
  target_bpm REAL,
  selected_title TEXT,
  selected_genre TEXT,
  match_mode TEXT,
  stretch_rate REAL,
  harm_penalty REAL,
  score REAL,
  track_id TEXT,
  note TEXT
);

CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry(ts);
"""

def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Establishes a concurrent database connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db(con: sqlite3.Connection) -> None:
    """Initializes the database schema."""
    con.executescript(SCHEMA_SQL)
    con.commit()

def insert_telemetry(con: sqlite3.Connection, row: Dict[str, Any]) -> None:
    """Inserts a single telemetry record into the database."""
    init_db(con)
    data = {
        "ts": float(row.get("ts", time.time())),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "activity": row.get("activity"),
        "energy": row.get("energy"),
        "cadence_raw": row.get("cadence_raw"),
        "cadence_smooth": row.get("cadence_smooth"),
        "conf": row.get("conf"),
        "target_bpm": row.get("target_bpm"),
        "selected_title": row.get("selected_title"),
        "selected_genre": row.get("selected_genre"),
        "match_mode": row.get("match_mode"),
        "stretch_rate": row.get("stretch_rate"),
        "harm_penalty": row.get("harm_penalty"),
        "score": row.get("score"),
        "track_id": row.get("track_id"),
        "note": row.get("note"),
    }

    con.execute(
        """
        INSERT INTO telemetry (
          ts, mode, status, activity, energy,
          cadence_raw, cadence_smooth, conf,
          target_bpm, selected_title, selected_genre,
          match_mode, stretch_rate, harm_penalty, score,
          track_id, note
        ) VALUES (
          :ts, :mode, :status, :activity, :energy,
          :cadence_raw, :cadence_smooth, :conf,
          :target_bpm, :selected_title, :selected_genre,
          :match_mode, :stretch_rate, :harm_penalty, :score,
          :track_id, :note
        )
        """,
        data,
    )
    con.commit()

def set_status(
    msg: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    mode: Optional[str] = None,
    activity: Optional[str] = None,
    energy: Optional[float] = None,
    cadence_smooth: Optional[float] = None,
    conf: Optional[float] = None,
    target_bpm: Optional[float] = None,
    selected_title: Optional[str] = None,
    selected_genre: Optional[str] = None,
    match_mode: Optional[str] = None,
    stretch_rate: Optional[float] = None,
    harm_penalty: Optional[float] = None,
    score: Optional[float] = None,
    track_id: Optional[str] = None,
    note: Optional[str] = None,
    also_print: bool = True,
) -> None:
    """Helper method to update the system status and log it to telemetry."""
    if also_print:
        print(msg)

    con = connect(db_path)
    try:
        insert_telemetry(con, {
            "ts": time.time(),
            "mode": mode,
            "status": msg,
            "activity": activity,
            "energy": energy,
            "cadence_smooth": cadence_smooth,
            "conf": conf,
            "target_bpm": target_bpm,
            "selected_title": selected_title,
            "selected_genre": selected_genre,
            "match_mode": match_mode,
            "stretch_rate": stretch_rate,
            "harm_penalty": harm_penalty,
            "score": score,
            "track_id": track_id,
            "note": note,
        })
    finally:
        try:
            con.close()
        except Exception:
            pass

def fetch_last(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieves the most recent telemetry entry."""
    con = connect(db_path)
    try:
        init_db(con)
        cur = con.execute("SELECT * FROM telemetry ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return {}

        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    finally:
        try:
            con.close()
        except Exception:
            pass