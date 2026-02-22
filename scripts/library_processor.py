"""
Music Library Processor.
Analyzes raw MP3 files in the audio cache to extract BPM, Key, Energy, 
and Beat-Grids using librosa, generating the central library database.
"""
import sys
from pathlib import Path

# Ensure Python can find the 'system' package from the root directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import librosa
import numpy as np
from tqdm import tqdm
import warnings
from system import config as cfg

warnings.filterwarnings('ignore')

def analyze_track(filepath: str) -> dict:
    """Extracts musical features from an audio file using advanced DSP."""
    try:
        # Load up to 120s for accurate analysis
        y, sr = librosa.load(filepath, duration=120)
        
        # 1. Harmonic-Percussive Source Separation (HPSS)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 2. BPM & Beats (on percussive track only)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # 3. Key (on harmonic track only)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        key_idx = int(np.argmax(np.mean(chroma, axis=1)))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # 4. Energy Calculation (RMS + Punch + Contrast)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        punch = float(np.mean(onset_env))
        contrast = float(np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr)))
        
        energy_score = (punch * 0.6) + (rms * 2.0 * 0.3) + (contrast * 0.02 * 0.1)
        
        # 5. BPM Normalization
        bpm_val = float(tempo)
        if energy_score > 1.0 and bpm_val < 100:
            bpm_val *= 2
        elif energy_score < 0.4 and bpm_val > 150:
            bpm_val /= 2
            
        return {
            "bpm_norm": bpm_val,
            "key": keys[key_idx],
            "energy": energy_score, 
            "duration": librosa.get_duration(y=y, sr=sr),
            "beats": beat_times.tolist()
        }
    except Exception as e:
        return {}

def main() -> None:
    print("--- HIGH-END LIBRARY PROCESSOR ---")
    print("Extracting Features: HPSS, Spectral Contrast, Beat-Grid")
    
    if not os.path.exists(cfg.AUDIO_ROOT):
        print(f"Error: Directory {cfg.AUDIO_ROOT} not found!")
        return

    files = [f for f in os.listdir(cfg.AUDIO_ROOT) if f.endswith((".mp3", ".wav"))]
    library = []
    
    print(f"Starting deep analysis for {len(files)} files...")
    print("Tip: This takes about 2-3 seconds per song.")
    
    for f in tqdm(files):
        path = os.path.join(cfg.AUDIO_ROOT, f)
        tid = f.replace(".mp3", "").replace(".wav", "")
        
        feats = analyze_track(path)
        if not feats:
            continue
            
        entry = {
            "id": tid,
            "filename": f,
            "path": path,
            "title": f,
            "artist": "Unknown",
            "genre": "unknown",
            "bpm_norm": feats["bpm_norm"],
            "bpm": feats["bpm_norm"],
            "key": feats["key"],
            "energy": feats["energy"],
            "beats": feats["beats"]
        }
        library.append(entry)

    # Energy Normalization (Min-Max Scaling to 0.0 - 1.0)
    energies = [x['energy'] for x in library]
    if energies:
        max_e, min_e = max(energies), min(energies)
        print(f"Normalizing energy (Range: {min_e:.2f} - {max_e:.2f})...")
        for x in library:
            norm = (x['energy'] - min_e) / (max_e - min_e + 1e-6)
            x['energy'] = round(norm, 4)

    # Save to JSON
    os.makedirs(os.path.dirname(cfg.LIBRARY_FILE), exist_ok=True)
    with open(cfg.LIBRARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(library, f, indent=2)
        
    print(f"\nDONE! {len(library)} tracks analyzed and saved to {cfg.LIBRARY_FILE}.")

if __name__ == "__main__":
    main()