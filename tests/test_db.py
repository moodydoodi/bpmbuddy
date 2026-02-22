"""
Unit tests for the SQLite logging infrastructure.
"""
import sys
import os
import tempfile
import sqlite3
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logger_sqlite import SQLiteLogger

def test_logger_insertion():
    """
    Test if the logger correctly creates a DB schema and inserts a row
    without throwing locking errors. Uses a temporary directory.
    """
    # Create an isolated temporary directory for the test database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_session.db")
        logger = SQLiteLogger(db_path)
        
        # Create mock telemetry data
        test_row = {
            "ts": 123456.78,
            "mode": "sim",
            "activity": "Jogging",
            "energy": 0.85,
            "target_bpm": 160.0
        }
        
        # Execute insert and commit
        logger.insert(test_row)
        logger.commit()
        logger.close()
        
        # Verify insertion via raw sqlite3 connection
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT ts, activity, energy FROM session_log")
        rows = cur.fetchall()
        con.close()
        
        # Assertions
        assert len(rows) == 1, "Exactly one row should be present"
        assert rows[0][0] == 123456.78, "Timestamp mismatch"
        assert rows[0][1] == "Jogging", "Activity mismatch"
        assert rows[0][2] == 0.85, "Energy mismatch"