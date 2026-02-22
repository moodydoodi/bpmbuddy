"""
Digital Signal Processing (DSP) and feature extraction module.
Provides robust algorithms for estimating physical movement energy 
and running cadence (SPM) using autocorrelation.
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CadenceState:
    """Maintains the temporal state of the user's running cadence."""
    cadence_smooth: Optional[float] = None
    last_good_t: float = 0.0

def energy_from_acc(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Computes a robust energy estimate using the standard deviation 
    of the 3-axis acceleration magnitude.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)

    if x.size < 5 or y.size < 5 or z.size < 5:
        return 0.0

    mag = np.sqrt(x * x + y * y + z * z)
    mag = mag[np.isfinite(mag)]

    if mag.size < 5:
        return 0.0

    e = float(np.std(mag))
    return e if np.isfinite(e) else 0.0

def estimate_cadence_acf(x: np.ndarray, y: np.ndarray, z: np.ndarray, sr: float) -> Tuple[Optional[float], float]:
    """
    Estimates cadence (Steps Per Minute) utilizing the Autocorrelation Function (ACF).
    Includes Hanning windowing to reduce spectral leakage.
    """
    if len(x) < int(sr * 3.0):
        return None, 0.0

    mag = np.sqrt(x * x + y * y + z * z)
    mag = mag - np.mean(mag)
    mag = mag * np.hanning(len(mag))

    acf = np.correlate(mag, mag, mode="full")[len(mag)-1:]
    if len(acf) < 10:
        return None, 0.0

    min_spm = 55.0
    max_spm = 185.0
    min_lag = int(sr * 60.0 / max_spm)
    max_lag = int(sr * 60.0 / min_spm)
    max_lag = min(max_lag, len(acf) - 1)
    
    if max_lag <= min_lag + 2:
        return None, 0.0

    seg = acf[min_lag:max_lag+1]
    peak_i = int(np.argmax(seg))
    peak_lag = peak_i + min_lag
    peak_val = float(seg[peak_i])

    baseline = float(np.median(seg))
    conf_raw = (peak_val - baseline) / (peak_val + 1e-9)
    conf = float(np.clip(conf_raw, 0.0, 1.0))

    spm = 60.0 * sr / float(peak_lag)
    return float(spm), conf

def harmonics_disambiguate(
    spm_raw: Optional[float],
    prev: Optional[float],
    energy: float,
    sticky_range=(0.75, 1.35),
) -> Optional[float]:
    """
    Prevents octave errors (e.g., misinterpreting 80 SPM as 160 SPM) 
    by penalizing unrealistic cadence leaps based on physical energy.
    """
    if spm_raw is None:
        return None

    cands = [c for c in (spm_raw / 2.0, spm_raw, spm_raw * 2.0) if 50.0 <= c <= 210.0]
    if not cands:
        return None

    if prev is None:
        prior = 100.0 if energy < 0.45 else 160.0
        return min(cands, key=lambda v: abs(v - prior))

    lo, hi = sticky_range

    def cost(v: float) -> float:
        base = abs(v - prev)
        ratio = v / (prev + 1e-9)
        penalty = 0.0 if (lo <= ratio <= hi) else 20.0
        if energy < 0.40 and v > 135.0:
            penalty += 50.0
        return base + penalty

    return min(cands, key=cost)

def smooth_cadence(
    state: CadenceState,
    spm_used: Optional[float],
    conf: float,
    conf_update_th: float = 0.25,
    cadence_lost_after_s: float = 4.0,
    alpha_base: float = 0.15,
) -> CadenceState:
    """Applies an Exponential Moving Average (EMA) to stabilize cadence readings."""
    now = time.time()
    if spm_used is not None and conf >= conf_update_th:
        w = (conf - conf_update_th) / max(1e-6, (1.0 - conf_update_th))
        alpha = alpha_base * (0.4 + 0.6 * w)
        if state.cadence_smooth is None:
            state.cadence_smooth = float(spm_used)
        else:
            state.cadence_smooth = float(alpha * spm_used + (1.0 - alpha) * state.cadence_smooth)
        state.last_good_t = now
    else:
        if state.cadence_smooth is not None and (now - state.last_good_t) >= cadence_lost_after_s:
            state.cadence_smooth = None
    return state