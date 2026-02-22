"""
Synthetic sensor data generator for the BPM-Buddy simulation mode.
Creates highly realistic, flowy accelerometer readings via random walk interpolation.
"""
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "demo_session.csv")
DURATION_SEC = 100  
SAMPLE_RATE = 10.0  

def generate() -> None:
    """Generates the demographic CSV utilized by the ReplayStream."""
    print(f"Generating synthetic evaluation data ({DURATION_SEC}s)...")
    
    phases = [
        (5,  0,   0,   "Standing", 0.05, 0.05),
        (20,  80,  115, "Walking",  0.20, 0.45),
        (30,  150, 160, "Jogging",  0.60, 0.70),
        (20,  170, 185, "Jogging",  0.80, 0.98),
        (20,  140, 90,  "Walking",  0.60, 0.30),
        (5,  0,   0,   "Standing", 0.10, 0.02) 
    ]
    
    records = []
    t = 0.0
    
    for duration, spm_start, spm_end, activity, e_start, e_end in phases:
        steps = int(duration * SAMPLE_RATE)
        
        for i in range(steps):
            p = i / steps
            
            current_spm = spm_start + (spm_end - spm_start) * p
            current_energy = e_start + (e_end - e_start) * p
            
            if current_spm > 0:
                current_spm += np.random.normal(0, 1.5) 
                current_energy += np.random.normal(0, 0.02)
            
            current_energy = max(0.0, min(1.0, current_energy))
            current_spm = max(0.0, current_spm)
            conf = 0.95 if current_spm > 60 else 0.10
            
            records.append({
                "ts": round(t, 2),
                "activity": activity,
                "cadence_smooth": round(current_spm, 2),
                "energy": round(current_energy, 4),
                "conf": conf
            })
            
            t += 1.0 / SAMPLE_RATE

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Success: File '{OUTPUT_FILE}' generated ({len(df)} records).")

if __name__ == "__main__":
    generate()