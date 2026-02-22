"""
Audio manipulation and playback engine.
Handles dynamic time-stretching (via phase vocoder) and applies 
beat alignment and Voice Activity Detection (VAD) rules for seamless transitions.
"""
import os
import json
import time
import hashlib
import threading
import pygame
import numpy as np
from system.db import set_status
import system.config as cfg

from dataclasses import dataclass
from typing import Any, Dict, Optional

# --- Dependencies Check ---
try:
    import soundfile as sf
    import pyrubberband as pyrb
    AUDIO_DEPS_OK = True
except Exception:
    AUDIO_DEPS_OK = False

@dataclass
class Selection:
    """Container for track metadata and calculated mixing targets."""
    track: Dict[str, Any]
    desired_bpm: float
    match_mode: str
    stretch_rate: float
    harm_pen: float
    score: float

# --- Harmonic Logic ---
KEY_MAP = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4, 'F': 5,
    'F#': 6, 'GB': 6, 'G': 7, 'G#': 8, 'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
}
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")

def key_to_pc(k: str) -> Optional[int]:
    """Converts a key string to an integer pitch class representation."""
    if not k:
        return None
    try:
        s = str(k).split(' ')[0].upper().replace('♭', 'B')
        return KEY_MAP.get(s, None)
    except Exception:
        return None

def harmonic_dist(a: Optional[int], b: Optional[int]) -> float:
    """Calculates harmonic distance using a simplified circle of fifths logic."""
    if a is None or b is None:
        return 1.0
    if a == b:
        return 0.0
    d = abs(a - b) % 12
    return 0.2 if d in [1, 11, 5, 7] else 2.0

def best_bpm_error(song_bpm: float, cadence_spm: float) -> tuple:
    """Determines optimal match mode (1:1 or halftime 2:1) and respective error."""
    e11 = abs(song_bpm - cadence_spm)
    e21 = abs(song_bpm * 2.0 - cadence_spm)
    if e21 <= e11 + 2.0:
        return e21, "2:1", cadence_spm / 2.0
    return e11, "1:1", cadence_spm

def load_library(path: str) -> list:
    """Loads analyzed music database safely."""
    set_status("📚 Loading music database...")
    if not os.path.exists(path):
        set_status("⚠️ Library file not found.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        set_status(f"❌ Library error: {e}")
        return []

# --- Core Audio Logic ---
class DJBrain:
    """Orchestrates track selection, audio caching, and threaded playback."""
    
    def __init__(self, library: list):
        self.library = library or []
        self.audio_index = {}
        
        try:
            root_dir = getattr(cfg, "AUDIO_ROOT", "")
            if root_dir and os.path.isdir(root_dir):
                for root, _, files in os.walk(root_dir):
                    for fn in files:
                        if fn.lower().endswith(AUDIO_EXTS):
                            self.audio_index[fn] = os.path.join(root, fn)
            else:
                set_status("⚠️ AUDIO_ROOT is missing or not a directory.")
        except Exception as e:
            set_status(f"⚠️ Audio index error: {e}")

        try:
            os.makedirs(cfg.AUDIO_CACHE_DIR, exist_ok=True)
        except Exception:
            pass

        # Mixer Init
        self.audio_enabled = False
        self.ch_a = None
        self.ch_b = None
        self.cur = None
        self.nxt = None
        try:
            pygame.mixer.init(frequency=44100)
            self.ch_a = pygame.mixer.Channel(0)
            self.ch_b = pygame.mixer.Channel(1)
            self.cur = self.ch_a
            self.nxt = self.ch_b
            self.audio_enabled = True
        except Exception as e:
            self.audio_enabled = False
            set_status(f"⚠️ Audio output disabled: {e}")

        self.current_track = None
        self.current_effective_bpm = None
        self.last_switch_t = 0.0
        self.track_start_time = 0.0
        self._lock = threading.Lock()
        self.is_processing = False

    def _find_audio_path(self, track: dict) -> Optional[str]:
        """Resolves local file path based on track metadata."""
        if not track:
            return None

        p = track.get("path")
        if p:
            try:
                p2 = os.path.normpath(str(p))
                if os.path.isabs(p2) and os.path.exists(p2): return p2
                if os.path.exists(p2): return p2

                base = os.path.dirname(os.path.abspath(__file__))
                cand_proj = os.path.normpath(os.path.join(base, "..", "..", p2))
                if os.path.exists(cand_proj): return cand_proj

                root_dir = getattr(cfg, "AUDIO_ROOT", "")
                if root_dir:
                    cand = os.path.normpath(os.path.join(root_dir, os.path.basename(p2)))
                    if os.path.exists(cand): return cand
            except Exception:
                pass

        fname = str(track.get("filename", "")).strip()
        if fname and fname in self.audio_index:
            return self.audio_index[fname]

        tid = str(track.get("id", "")).strip()
        if tid:
            for k, v in self.audio_index.items():
                if tid in k: return v
        return None

    def _process_audio(self, filepath: str, rate: float, start_offset: float = 0.0) -> tuple:
        """Applies offline time-stretching and caches the resultant audio."""
        if (not cfg.ENABLE_TIME_STRETCH) or (not AUDIO_DEPS_OK):
            return filepath, False

        try: rate = max(0.8, min(1.2, float(rate)))
        except Exception: return filepath, False
        try: start_offset = max(0.0, float(start_offset))
        except Exception: start_offset = 0.0

        file_hash = hashlib.md5(f"{filepath}_{rate:.3f}_{start_offset:.2f}".encode()).hexdigest()[:12]
        cache_path = os.path.join(cfg.AUDIO_CACHE_DIR, f"{file_hash}.wav")

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 10_000:
            return cache_path, True

        try:
            y, sr = sf.read(filepath, always_2d=False)
            y = np.asarray(y)
            if y.dtype.kind != "f":
                y = y.astype(np.float32) / np.iinfo(y.dtype).max
            else:
                y = y.astype(np.float32)

            if start_offset > 0.0:
                start_sample = int(start_offset * sr)
                if 0 < start_sample < len(y): y = y[start_sample:]

            if abs(rate - 1.0) > 0.02:
                y = pyrb.time_stretch(y, sr, rate)

            sf.write(cache_path, y, sr)
            return cache_path, True
        except Exception as e:
            set_status(f"⚠️ Stretch failed -> playing original ({e})")
            return filepath, False

    def get_smart_wait_time(self, current_bpm: float) -> float:
        """Calculates hold time to ensure transitions hit on a beat or VAD safe zone."""
        if not self.current_track or self.track_start_time <= 0: return 0.0
        try: current_bpm = float(current_bpm)
        except Exception: return 0.0
        if current_bpm <= 0: return 0.0

        elapsed = time.time() - self.track_start_time

        # Beat Sync
        beats = self.current_track.get("beats", [])
        next_beat_time = 0.0
        if isinstance(beats, list) and beats:
            future = [float(b) for b in beats if float(b) > (elapsed + 0.1)]
            next_beat_time = future[0] if future else elapsed + (4 * (60.0 / current_bpm))
        else:
            bar_len = (60.0 / current_bpm) * 4.0
            next_beat_time = elapsed + (bar_len - (elapsed % bar_len))

        wait_for_beat = max(0.0, next_beat_time - elapsed)

        # VAD Verification
        mix_segments = self.current_track.get("mix_segments", [])
        if not mix_segments: return wait_for_beat

        if isinstance(mix_segments, dict) and isinstance(mix_segments.get("safe_points", []), list):
            future_safe = [float(s) for s in mix_segments["safe_points"] if float(s) > next_beat_time]
            if future_safe:
                wait_for_vad = max(0.0, future_safe[0] - elapsed)
                if wait_for_vad <= 12.0:
                    set_status(f"🎤 Waiting for safe point ({wait_for_vad:.1f}s)...")
                    return wait_for_vad
            return wait_for_beat

        if isinstance(mix_segments, list):
            is_safe = False
            next_safe_start = None
            for seg in mix_segments:
                if not isinstance(seg, dict): continue
                s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
                if s <= next_beat_time <= e:
                    is_safe = True; break
                if s > next_beat_time and (next_safe_start is None or s < next_safe_start):
                    next_safe_start = s

            if is_safe: return wait_for_beat
            if next_safe_start is not None:
                wait_for_vad = max(0.0, next_safe_start - elapsed)
                if wait_for_vad <= 12.0:
                    set_status(f"🎤 Waiting for vocal end ({wait_for_vad:.1f}s)...")
                    return wait_for_vad

        return wait_for_beat

    def select_track(self, target_bpm: float, high_energy: bool, genres: list, current_key_pc=None) -> Optional[dict]:
        """Heuristic evaluation to find the best matching track from library."""
        try: target_bpm = float(target_bpm)
        except Exception: return None
        if target_bpm <= 0: return None

        genres_set = set([g.lower() for g in (genres or [])])
        cands = []

        for tr in self.library:
            if not isinstance(tr, dict): continue
            bpm = tr.get("bpm_norm")
            energy = tr.get("energy")
            
            try: bpm = float(bpm) if bpm is not None else None
            except Exception: bpm = None
            
            if not bpm or bpm <= 0: continue
            if genres_set and str(tr.get("genre", "unknown")).lower() not in genres_set: continue

            key_pc = key_to_pc(tr.get("key"))
            harm_pen = harmonic_dist(current_key_pc, key_pc) if current_key_pc is not None else 0.0

            if cfg.ENABLE_TIME_STRETCH and AUDIO_DEPS_OK:
                rate1 = target_bpm / bpm
                rate2 = (target_bpm / 2.0) / bpm
                v1, v2 = (0.8 <= rate1 <= 1.2), (0.8 <= rate2 <= 1.2)
                if not (v1 or v2): continue
                if v1 and (not v2 or abs(1 - rate1) < abs(1 - rate2)):
                    des, mode, err = target_bpm, "1:1", 0.0
                else:
                    des, mode, err = target_bpm / 2.0, "2:1", 0.0
            else:
                err, mode, des = best_bpm_error(bpm, target_bpm)
                if err > 15: continue

            stretch_cost = abs(1.0 - (des / bpm)) * 100.0 if (cfg.ENABLE_TIME_STRETCH and AUDIO_DEPS_OK) else float(err)
            if mode == "1:1": stretch_cost -= 5.0

            score = stretch_cost + (harm_pen * 15.0)

            try:
                if energy is not None:
                    score += (-10.0 * float(energy)) if high_energy else (20.0 * float(energy))
                elif not high_energy: score += 10.0
            except Exception: pass

            cands.append({ "track": tr, "desired": float(des), "score": float(score), "mode": mode, "err": float(stretch_cost) })

        cands.sort(key=lambda x: x["score"])
        return cands[0] if cands else None

    def play_transition_threaded(self, selection: dict, fade_ms: int):
        """Asynchronous execution of the track loading and crossfading."""
        if not selection: return
        with self._lock:
            if self.is_processing: return
            self.is_processing = True
        threading.Thread(target=self._execute, args=(selection, fade_ms), daemon=True).start()

    def stop(self):
        """Halts PyGame playback."""
        if not self.audio_enabled: return
        try:
            if self.cur: self.cur.stop()
            if self.nxt: self.nxt.stop()
        except Exception: pass

    def _execute(self, selection: dict, fade_ms: int):
        """Internal playback logic."""
        try:
            tr = selection.get("track")
            path = self._find_audio_path(tr)
            if not path or not os.path.exists(path):
                set_status(f"⚠️ Audio missing: {tr.get('title')}")
                return
            if not self.audio_enabled:
                set_status("⚠️ Audio output disabled.")
                return

            target = float(selection.get("desired", 0.0))
            bpm0 = float(tr.get("bpm_norm", 0.0))
            if bpm0 <= 0 or target <= 0: return

            rate = target / bpm0
            set_status(f"🎧 Mixing: {tr.get('title')} ({target:.1f} BPM)...")

            # Offset based on beat grid to bypass silence
            start_offset = float(tr.get("beats", [0.0])[0]) if tr.get("beats") else 0.0
            final_path, _ = self._process_audio(path, rate, start_offset)

            if cfg.ENABLE_BEAT_ALIGN and self.current_effective_bpm and self.cur and self.cur.get_busy():
                wait = self.get_smart_wait_time(self.current_effective_bpm)
                if 0.05 < wait < 4.0:
                    if wait < 1.5: set_status(f"🥁 Waiting for beat ({wait:.1f}s)...")
                    time.sleep(wait)

            snd = pygame.mixer.Sound(final_path)
            self.nxt.play(snd, fade_ms=int(fade_ms))
            if self.cur.get_busy(): self.cur.fadeout(int(fade_ms))

            self.cur, self.nxt = self.nxt, self.cur
            self.current_track = tr
            self.current_effective_bpm = target
            self.last_switch_t = time.time()
            self.track_start_time = time.time()

            set_status(f"🎵 {tr.get('title')}")

        except Exception as e:
            set_status(f"❌ Mixer error: {e}")
        finally:
            with self._lock:
                self.is_processing = False

class AudioEngine:
    """Wrapper API utilized by the main application logic."""
    def __init__(self, audio_root: str, out_dir: str = "outputs", enable_time_stretch: bool = True,
                 enable_beat_align: bool = True, enable_harmonic_mix: bool = True,
                 sample_rate: int = 44100, cache_dir: Optional[str] = None):
        cfg.AUDIO_ROOT = audio_root
        cfg.ENABLE_TIME_STRETCH = bool(enable_time_stretch)
        cfg.ENABLE_BEAT_ALIGN = bool(enable_beat_align)
        self.enable_harmonic_mix = bool(enable_harmonic_mix)
        
        cfg.AUDIO_CACHE_DIR = cache_dir or os.path.join(out_dir, "cache_stretch")
        os.makedirs(cfg.AUDIO_CACHE_DIR, exist_ok=True)
        self._dj = DJBrain(library=[])

    @property
    def is_processing(self) -> bool: return bool(self._dj.is_processing)
    @property
    def current_track(self): return self._dj.current_track
    @property
    def current_effective_bpm(self): return self._dj.current_effective_bpm
    @property
    def last_switch_t(self) -> float: return float(self._dj.last_switch_t)
    @last_switch_t.setter
    def last_switch_t(self, v: float):
        try: self._dj.last_switch_t = float(v)
        except Exception: self._dj.last_switch_t = 0.0

    def stop(self): return self._dj.stop()
    def play(self, sel: Selection, fade_ms: int = 1600):
        selection_dict = {
            "track": sel.track, "desired": float(sel.desired_bpm), "mode": str(sel.match_mode),
            "err": 0.0, "score": float(sel.score), "stretch_rate": float(sel.stretch_rate),
        }

        self._dj.play_transition_threaded(selection_dict, fade_ms)
