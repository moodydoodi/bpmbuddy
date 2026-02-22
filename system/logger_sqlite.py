"""
Dedicated SQLite logging module for session analytics.
Stores high-resolution telemetry including reinforcement learning state actions.
"""
import os
import sqlite3
from typing import Any, Dict

class SQLiteLogger:
    """
    Manages the persistent session logging for offline evaluation.
    Utilizes Write-Ahead Logging (WAL) to ensure thread-safe, concurrent access.
    """
    
    def __init__(self, db_path: str) -> None:
        """
        Initializes the database connection and sets appropriate PRAGMA rules.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL;")
        self.con.execute("PRAGMA synchronous=NORMAL;")
        self.con.execute("PRAGMA busy_timeout=2500;")
        self._init()

    def _init(self) -> None:
        """
        Initializes the schema and handles basic column migrations.
        """
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS session_log (
            ts REAL,
            mode TEXT,
            activity TEXT,
            energy REAL,
            cadence_raw REAL,
            cadence_smooth REAL,
            conf REAL,
            target_bpm REAL,
            rl_action INTEGER,
            selected_id TEXT,
            selected_title TEXT,
            selected_genre TEXT,
            match_mode TEXT,
            stretch_rate REAL,
            harm_penalty REAL,
            score REAL,
            song_bpm REAL,
            effective_bpm REAL,
            status TEXT
        )
        """)
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_session_ts ON session_log(ts)")
        self.con.commit()

        cur = self.con.execute("PRAGMA table_info(session_log)")
        existing_columns = {r[1] for r in cur.fetchall()}
        
        if "song_bpm" not in existing_columns:
            self.con.execute("ALTER TABLE session_log ADD COLUMN song_bpm REAL")
        if "effective_bpm" not in existing_columns:
            self.con.execute("ALTER TABLE session_log ADD COLUMN effective_bpm REAL")
            
        self.con.commit()

    def insert(self, row: Dict[str, Any]) -> None:
        """Inserts a populated data row into the session log."""
        self.con.execute("""
        INSERT INTO session_log (
            ts, mode, activity, energy,
            cadence_raw, cadence_smooth, conf,
            target_bpm, rl_action,
            selected_id, selected_title, selected_genre,
            match_mode, stretch_rate, harm_penalty, score,
            song_bpm, effective_bpm,
            status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row.get("ts"), row.get("mode"), row.get("activity"), row.get("energy"),
            row.get("cadence_raw"), row.get("cadence_smooth"), row.get("conf"),
            row.get("target_bpm"), row.get("rl_action"), row.get("selected_id"),
            row.get("selected_title"), row.get("selected_genre"), row.get("match_mode"),
            row.get("stretch_rate"), row.get("harm_penalty"), row.get("score"),
            row.get("song_bpm"), row.get("effective_bpm"), row.get("status"),
        ))

    def commit(self) -> None:
        """Flushes pending transactions to disk."""
        self.con.commit()

    def close(self) -> None:
        """Safely commits data and closes the database connection."""
        try:
            self.con.commit()
            self.con.close()
        except Exception:
            pass