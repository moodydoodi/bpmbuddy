"""
Unit Tests for the BPM-Buddy DSP Module.
Validates cadence estimation (Autocorrelation) using synthetic signals.
"""
import sys
from pathlib import Path

# Ensure Python can find the 'system' package from the root directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from system.features import estimate_cadence_acf

def test_estimate_cadence_sine():
    """Tests if a perfect sine wave is correctly identified as a specific cadence."""
    sr = 20.0
    spm_target = 120.0
    f = spm_target / 60.0
    
    # Generate 6 seconds of synthetic running signal
    t = np.arange(0, 6.0, 1.0 / sr)
    y = np.sin(2 * np.pi * f * t) # The actual "step" motion
    x = np.random.normal(0, 0.01, size=len(t)) # Slight noise
    z = np.random.normal(0, 0.01, size=len(t)) # Slight noise

    spm, conf = estimate_cadence_acf(x, y, z, sr)
    
    # Verify the result is highly plausible
    assert spm is not None, "Cadence should not be None for a valid signal"
    assert abs(spm - spm_target) < 5.0, f"Expected approx {spm_target} SPM, got {spm}"
    assert conf > 0.1, "Confidence should be high for a clean sine wave"

def test_estimate_short_signal():
    """Tests if the system gracefully handles signals that are too short to analyze."""
    sr = 20.0
    # Only 2 seconds of data (too short for autocorrelation)
    t = np.arange(0, 2.0, 1.0 / sr)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)

    spm, conf = estimate_cadence_acf(x, y, z, sr)
    
    # Expectation: The system detects insufficient data and returns None without crashing
    assert spm is None, "SPM should be None for a truncated signal"
    assert conf == 0.0, "Confidence should be 0.0"