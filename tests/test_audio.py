"""
Unit tests for the audio engine and musical heuristic helpers.
"""
import sys
import os
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_engine import best_bpm_error, harmonic_dist, key_to_pc

def test_best_bpm_error_1_to_1():
    """Test standard 1:1 BPM matching (cadence matches song)."""
    err, mode, target = best_bpm_error(160.0, 160.0)
    assert err == 0.0, "Error should be 0 for exact match"
    assert mode == "1:1", "Mode should be 1:1"
    assert target == 160.0, "Target BPM should remain 160.0"

def test_best_bpm_error_2_to_1():
    """Test halftime 2:1 BPM matching (runner at 160 SPM, song at 80 BPM)."""
    err, mode, target = best_bpm_error(80.0, 160.0)
    assert err == 0.0, "Error should be 0 for perfect halftime match"
    assert mode == "2:1", "Mode should be 2:1 (halftime)"
    assert target == 80.0, "Target BPM should be halved to match the song"

def test_harmonic_distance():
    """Test the harmonic penalty calculation for key matching."""
    assert harmonic_dist(0, 0) == 0.0, "Same key should yield 0 penalty"
    assert harmonic_dist(None, 5) == 1.0, "Unknown key should yield neutral penalty of 1.0"
    assert harmonic_dist(0, 7) == 0.2, "Neighboring keys (perfect fifth) should yield low penalty"
    assert harmonic_dist(0, 6) == 2.0, "Dissonant keys (tritone) should yield high penalty"

def test_key_to_pc():
    """Test string to pitch class integer conversion."""
    assert key_to_pc("C") == 0
    assert key_to_pc("C#") == 1
    assert key_to_pc("EB") == 3
    assert key_to_pc("Invalid") is None