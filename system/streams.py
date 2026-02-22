"""
Data Acquisition Module.
Manages network-based live sensor ingestion (Phyphox) and offline simulation streams.
"""
import os
import time
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional


@dataclass
class ReplayFrame:
    """Data structure representing a single timeframe in simulated telemetry."""
    ts: float
    activity: str
    cadence_raw: Optional[float]
    cadence_smooth: Optional[float]
    conf: float
    energy: float


class PhyphoxStream:
    """
    Robust REST-client for real-time sensor data extraction from the Phyphox app.
    Handles network timeouts and data sanitization.
    """
    
    def __init__(self, base_url: str):
        self.base_url = str(base_url).rstrip("/")
        self.last_t = -1e9
        self.error_count = 0
        print(f"📡 Connecting to sensor: {self.base_url}")

    def fetch_data(self) -> List[Tuple[float, float, float, float]]:
        """
        Polls the Phyphox endpoint for new accelerometer records.
        """
        try:
            pipe = "%7C"
            url = (f"{self.base_url}/get?acc_time={self.last_t}"
                   f"&accX={self.last_t}{pipe}acc_time"
                   f"&accY={self.last_t}{pipe}acc_time"
                   f"&accZ={self.last_t}{pipe}acc_time")
                   
            r = requests.get(url, timeout=0.5)
            if r.status_code != 200:
                return []

            data = r.json()
            buffer = data.get("buffer", {})

            def get_array(key: str) -> list:
                obj = buffer.get(key)
                if isinstance(obj, dict) and "buffer" in obj:
                    return obj["buffer"]
                return obj if isinstance(obj, list) else []

            ts = get_array("acc_time")
            xs = get_array("accX")
            ys = get_array("accY")
            zs = get_array("accZ")

            if len(ts) > 0:
                self.error_count = 0
                out = []
                for i in range(min(len(ts), len(xs), len(ys), len(zs))):
                    self.last_t = float(ts[i])
                    out.append((self.last_t, float(xs[i]), float(ys[i]), float(zs[i])))
                return out
                
            return []

        except Exception:
            self.error_count += 1
            if self.error_count % 30 == 0:
                print(f"⚠️ No data from {self.base_url} (Is Phyphox running?)")
            return []


class ReplayStream:
    """
    Offline simulator reading historical or synthetic session data from a CSV file.
    Provides identical interfaces to PhyphoxStream for backend abstraction.
    """
    
    def __init__(self, csv_path: str, speed: float = 1.0, hz: float = 10.0):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Simulation file missing: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.speed = float(max(0.05, speed))
        self.hz = float(hz)
        self.start_wall = time.time()
        self.i = 0
        self.n = len(self.df)
        print(f"📼 Simulation loaded: {self.n} rows from {csv_path}")

    def fetch_frame(self) -> Optional[ReplayFrame]:
        """Calculates temporal progression and fetches the appropriate simulated frame."""
        if self.n <= 0:
            return None

        elapsed = (time.time() - self.start_wall) * self.speed
        target_i = int(elapsed * self.hz)
        target_i = min(target_i, self.n - 1)

        if self.i > target_i:
            return None

        row = self.df.iloc[self.i]
        self.i += 1

        def parse_col(name: str, default: Any = None) -> Any:
            v = row.get(name, default)
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return default
                return float(v)
            except Exception:
                return default

        return ReplayFrame(
            ts=parse_col("ts", time.time()),
            activity=str(row.get("activity", "Unknown")),
            cadence_raw=parse_col("cadence_raw", None),
            cadence_smooth=parse_col("cadence_smooth", None),
            conf=parse_col("conf", 0.0),
            energy=parse_col("energy", 0.0)
        )