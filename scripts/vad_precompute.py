"""
Voice Activity Detection (VAD) Pre-computation Script.
Analyzes the music library offline to identify vocal sections, creating 
'safe zones' for DJ transitions to avoid vocal clashing.
"""
import sys
from pathlib import Path

# Ensure Python can find the 'system' package from the root directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import ssl
import warnings
import numpy as np
import soundfile as sf
import torch
from typing import Dict, List, Tuple

from system import config as cfg

# Bypass SSL verification for legacy torch hub downloads
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

def load_silero():
    """Loads the Silero VAD model, prioritizing a local JIT file."""
    if Path(cfg.MODEL_VAD_PATH).exists():
        print(f"✅ Loading local VAD model from: {cfg.MODEL_VAD_PATH}")
        try:
            model = torch.jit.load(cfg.MODEL_VAD_PATH)
            return model, get_speech_timestamps_local
        except Exception as e:
            print(f"❌ Error loading local file: {e}")

    print("⚠️ Local model not found. Attempting Torch Hub download...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, _, read_audio, _, collect_chunks) = utils
        return model, get_speech_timestamps
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

def get_speech_timestamps_local(
    audio: torch.Tensor, model: torch.nn.Module, 
    threshold: float = 0.5, sampling_rate: int = 16000
) -> List[Dict[str, float]]:
    """Local implementation of speech detection."""
    if len(audio.shape) == 1: 
        audio = audio.unsqueeze(0)
    
    window_size = 512 if sampling_rate == 16000 else 256
    model.reset_states()
    speech_probs = []
    
    for i in range(0, audio.shape[1], window_size):
        chunk = audio[:, i: i+window_size]
        if chunk.shape[1] < window_size:
            pad = torch.zeros(1, window_size - chunk.shape[1])
            chunk = torch.cat([chunk, pad], dim=1)
            
        with torch.no_grad():
            out = model(chunk, torch.tensor([sampling_rate]))
        
        if out.ndim == 2 and out.shape[1] == 2:
            prob = out[0][1].item()
        elif out.ndim == 2 and out.shape[1] == 1:
            prob = out[0][0].item()
        else:
            prob = out.item()
            
        speech_probs.append(prob)

    triggered = False
    speeches = []
    current_speech = {}
    frame_dur = window_size / sampling_rate
    
    for i, prob in enumerate(speech_probs):
        time_sec = i * frame_dur
        if prob >= threshold and not triggered:
            triggered = True
            current_speech['start'] = time_sec
            
        if prob < (threshold - 0.15) and triggered:
            triggered = False
            current_speech['end'] = time_sec
            speeches.append(current_speech)
            current_speech = {}
            
    if triggered:
        current_speech['end'] = len(speech_probs) * frame_dur
        speeches.append(current_speech)
        
    return [s for s in speeches if (s['end'] - s['start']) > 0.25]

def read_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    """Reads audio and safely resamples to 16kHz Mono."""
    try:
        x, sr = sf.read(path, always_2d=False)
    except Exception as e:
        print(f"Read error {path}: {e}")
        return np.array([]), 0

    if x.ndim > 1: 
        x = x.mean(axis=1)
    
    if sr != 16000 and len(x) > 0:
        t_old = np.linspace(0, 1, len(x), endpoint=False)
        t_new = np.linspace(0, 1, int(len(x) * 16000 / sr), endpoint=False)
        x = np.interp(t_new, t_old, x).astype(np.float32)
        sr = 16000
    
    return x.astype(np.float32), sr

def main() -> None:
    if not Path(cfg.LIBRARY_FILE).exists():
        print(f"Error: {cfg.LIBRARY_FILE} not found.")
        return

    model, get_speech_timestamps = load_silero()
    model.eval()

    try:
        with open(cfg.LIBRARY_FILE, "r", encoding="utf-8") as f:
            lib = json.load(f)
    except Exception as e:
        print(f"JSON Error: {e}")
        return

    changed = 0
    print(f"Starting VAD analysis for {len(lib)} tracks...")

    for i, tr in enumerate(lib):
        fn = tr.get("filename")
        if not fn: continue
        
        audio_path = Path(cfg.AUDIO_ROOT) / fn
        if not audio_path.exists() and "path" in tr and os.path.exists(tr["path"]):
            audio_path = Path(tr["path"])
        elif not audio_path.exists():
            continue

        if "mix_segments" in tr:
            continue

        x, sr = read_mono_16k(str(audio_path))
        if len(x) == 0: continue

        dur = len(x) / sr
        wav = torch.from_numpy(x)
        
        ts_raw = get_speech_timestamps(wav, model, sampling_rate=sr)
        speech = [{"start": round(float(s["start"])/sr if s["start"] > 5000 else float(s["start"]), 3), 
                   "end": round(float(s["end"])/sr if s["end"] > 5000 else float(s["end"]), 3)} for s in ts_raw]
        
        cur = 0.0
        mix = []
        for s in sorted(speech, key=lambda i: i["start"]):
            if s["start"] - cur >= 4.0:
                mix.append({"start": round(cur, 2), "end": round(s["start"], 2)})
            cur = max(cur, s["end"])
        if dur - cur >= 4.0:
            mix.append({"start": round(cur, 2), "end": round(dur, 2)})

        tr["speech_segments"] = speech
        tr["mix_segments"] = mix
        
        changed += 1
        print(f"[{i+1}/{len(lib)}] {fn[:20]}... -> Speech: {len(speech)}, Mix-Zones: {len(mix)}")

        if changed % 10 == 0:
             with open(cfg.LIBRARY_FILE, "w", encoding="utf-8") as f:
                json.dump(lib, f, ensure_ascii=False, indent=2)

    if changed:
        with open(cfg.LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(lib, f, ensure_ascii=False, indent=2)
        print(f"✅ Done! {changed} tracks updated.")
    else:
        print("Everything up to date.")

if __name__ == "__main__":
    main()