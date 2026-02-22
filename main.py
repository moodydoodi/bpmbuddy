"""
Main Orchestrator for BPM-Buddy.
Optimized for smooth simulation handling and dynamic pace tracking.
Ensures music 'chases' the runner with realistic inertia.
"""
import os
import json
import time
import collections
import sys
from typing import Optional

import numpy as np
import torch

# Package internal imports
from system import config as cfg
from system.audio_engine import AudioEngine, Selection
from system.features import (
    CadenceState, 
    energy_from_acc, 
    estimate_cadence_acf, 
    harmonics_disambiguate, 
    smooth_cadence
)
from system.streams import PhyphoxStream, ReplayStream
from system.logger_sqlite import SQLiteLogger
from system.policy import DecisionPolicy

def get_operating_mode() -> str:
    if os.path.exists(cfg.CONTROL_JSON):
        try:
            with open(cfg.CONTROL_JSON, "r") as f:
                return json.load(f).get("mode", "live")
        except Exception: pass
    return "live"

def main() -> None:
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    print("\n=== BPM-BUDDY SYSTEM (Physical Consistency Mode) ===")

    # 1. Initialization
    library_data = []
    if os.path.exists(cfg.LIBRARY_FILE):
        try:
            with open(cfg.LIBRARY_FILE, "r", encoding="utf-8") as f:
                library_data = json.load(f)
            print(f"✅ Library loaded: {len(library_data)} tracks found.")
        except Exception as e: print(f"❌ Library error: {e}")

    audio = AudioEngine(audio_root=cfg.AUDIO_ROOT, out_dir=cfg.OUT_DIR)
    audio._dj.library = library_data 
    
    log = SQLiteLogger(cfg.DB_PATH)
    policy = DecisionPolicy(use_rl=cfg.USE_RL)

    # 2. Select Stream
    mode = get_operating_mode()
    if mode in ["replay", "sim"]:
        print(f"🎬 MODE: SMOOTH SIMULATION ({os.path.basename(cfg.SIM_DATA_FILE)})")
        stream = ReplayStream(cfg.SIM_DATA_FILE, speed=1.0, hz=cfg.TARGET_HZ)
    else:
        print(f"📡 MODE: LIVE SENSOR ({cfg.DEFAULT_PHYPHOX_URL})")
        stream = PhyphoxStream(cfg.DEFAULT_PHYPHOX_URL)
    
    # 3. State & Buffers
    bx = collections.deque(maxlen=cfg.CAD_WINDOW)
    by = collections.deque(maxlen=cfg.CAD_WINDOW)
    bz = collections.deque(maxlen=cfg.CAD_WINDOW)
    
    last_resample_t = None
    last_decision_t = 0.0
    cadence_state = CadenceState()
    
    # Visual states for dashboard stability
    visual_music_bpm = 0.0
    last_score = 0.0
    last_rl_action = 0

    print("\n🚀 System active. Visualizing flow in dashboard...")

    try:
        while True:
            loop_t0 = time.time()
            
            # --- DATA INGESTION ---
            if hasattr(stream, 'fetch_data'):
                samples = stream.fetch_data() # Live Mode
            else:
                frame = stream.fetch_frame() # Replay Mode
                samples = []
                if frame:
                    # IMPROVED SYNTHESIS: Less jitter for a cleaner DSP result
                    fake_t = (last_resample_t + (1.0 / cfg.TARGET_HZ)) if last_resample_t else time.time()
                    freq = (frame.cadence_smooth or 0.0) / 60.0
                    
                    # Simulate a cleaner, more rhythmic vertical acceleration (Y-axis)
                    # We add very subtle noise only (0.05) to keep the ACF peak sharp
                    clean_noise = np.random.normal(0, 0.05)
                    syn_y = np.sin(fake_t * freq * 2 * np.pi) * (8.0 if frame.energy > 0.5 else 2.5) + clean_noise
                    samples = [(fake_t, 0.0, syn_y, 9.81)]
            
            for (t, x, y, z) in samples:
                if last_resample_t is None or (t - last_resample_t) >= (1.0 / cfg.TARGET_HZ) * 0.95:
                    last_resample_t = t
                    bx.append(x); by.append(y); bz.append(z)

            if len(bx) < cfg.CNN_WINDOW:
                time.sleep(cfg.POLL_INTERVAL); continue

            # --- DIGITAL SIGNAL PROCESSING (DSP) ---
            x_arr, y_arr, z_arr = np.array(bx), np.array(by), np.array(bz)
            energy = float(energy_from_acc(x_arr, y_arr, z_arr))
            is_moving = energy > 0.08
            cadence_raw, conf = estimate_cadence_acf(x_arr, y_arr, z_arr, cfg.TARGET_HZ)
            
            if is_moving:
                activity = "Jogging" if energy > 0.45 else "Walking"
                # Stabilized harmonic disambiguation
                spm_used = harmonics_disambiguate(cadence_raw, cadence_state.cadence_smooth, energy)
                cadence_state = smooth_cadence(cadence_state, spm_used, conf, cfg.CONF_UPDATE_TH)
            else:
                activity, cadence_state = "Idle", CadenceState()

            cad_smooth = cadence_state.cadence_smooth
            target_bpm = cad_smooth if cad_smooth is not None else cfg.IDLE_TARGET_BPM

            # --- DYNAMIC PACE TRACKING (MUSIC SYNC) ---
            # Inertia effect: Music line 'chases' the runner line smoothly
            if visual_music_bpm == 0: visual_music_bpm = target_bpm
            # Factor 0.03 = Very smooth, professional tracking look
            inertia_factor = 0.03 if mode != "live" else 0.08
            visual_music_bpm += (target_bpm - visual_music_bpm) * inertia_factor

            # Telemetry Output
            cur_track = audio.current_track
            track_name = cur_track.get('title', '-') if cur_track else '-'
            sys.stdout.write(f"\r[STATUS] {activity:7s} | Run: {cad_smooth or 0:3.0f} | DJ: {visual_music_bpm:3.0f} | {track_name[:15]}")
            sys.stdout.flush()

            # --- DATA LOGGING ---
            try:
                log.insert({
                    "ts": loop_t0, "mode": mode, "activity": activity, "energy": float(energy),
                    "cadence_raw": cadence_raw, "cadence_smooth": cad_smooth, "conf": float(conf),
                    "target_bpm": float(target_bpm), "rl_action": int(last_rl_action),
                    "selected_id": cur_track.get("id") if cur_track else None,
                    "selected_title": cur_track.get("title") if cur_track else None,
                    "selected_genre": cur_track.get("genre") if cur_track else None,
                    "match_mode": "1:1", "stretch_rate": 1.0, "harm_penalty": 0.0, 
                    "score": float(last_score),
                    "song_bpm": cur_track.get("bpm_norm") if cur_track else None,
                    "effective_bpm": float(visual_music_bpm),
                    "status": "Running"
                })
                log.commit()
            except Exception: pass

            # --- DECISION LOGIC ---
            if (loop_t0 - last_decision_t) >= cfg.DECISION_EVERY_S:
                if (is_moving and conf >= cfg.CONF_DECISION_TH) or (not is_moving):
                    sel_dict = audio._dj.select_track(target_bpm, energy > 0.6, genres=[])
                    if sel_dict:
                        last_score = sel_dict.get("score", 0.0)
                        is_new = cur_track is None or sel_dict["track"]["id"] != cur_track["id"]
                        if is_new and (loop_t0 - audio.last_switch_t) >= cfg.SWITCH_COOLDOWN_S:
                            sel_obj = Selection(
                                track=sel_dict["track"], desired_bpm=sel_dict["desired"],
                                match_mode=sel_dict["mode"], stretch_rate=sel_dict["desired"]/sel_dict["track"]["bpm_norm"],
                                score=sel_dict["score"], harm_pen=0.0
                            )
                            audio.play(sel_obj, fade_ms=2000)
                            last_decision_t = loop_t0

            time.sleep(max(0.0, cfg.POLL_INTERVAL - (time.time() - loop_t0)))

    except KeyboardInterrupt: print("\n[Exit] Shutdown.")
    finally: log.close(); audio.stop()

if __name__ == "__main__":
    main()