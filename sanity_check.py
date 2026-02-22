"""
Diagnostic utility tool for the BPM-Buddy project environment.
Verifies the existence of critical models, datasets, dependencies, 
and path structures prior to application launch.
"""
import sys
from pathlib import Path

# Ensure Python can find the 'system' package from the root directory
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import shutil
import pygame
from system import config as cfg

EXPECTED_FILES = {
    "CNN Model": cfg.MODEL_CNN_PATH,
    "RL Model": cfg.MODEL_RL_PATH,
    "Music Database": cfg.LIBRARY_FILE,
    "Demo Data": cfg.SIM_DATA_FILE,
    "Backend Script": "main.py",
    "Dashboard Script": "dashboard.py"
}

def check_step(name: str, filepath: str) -> bool:
    """Verifies absolute file existence."""
    if os.path.exists(filepath):
        print(f"✅ {name:20} -> FOUND")
        return True
    
    print(f"❌ {name:20} -> MISSING! (Expected: {filepath})")
    return False

def run_diagnostics() -> None:
    """Executes the master diagnostic suite."""
    print("=" * 50)
    print("   BPM-BUDDY SYSTEM DIAGNOSTICS")
    print("=" * 50 + "\n")
    
    # 1. Structure Check: Create folders dynamically if they don't exist
    for folder in [cfg.OUT_DIR, cfg.AUDIO_ROOT, cfg.AUDIO_CACHE_DIR, cfg.MODELS_DIR]:
        os.makedirs(folder, exist_ok=True)

    # 2. File Check
    results = {}
    for label, path in EXPECTED_FILES.items():
        results[label] = check_step(label, path)

    print("\n" + "-" * 30)
    
    # 3. Audio System Check
    try:
        pygame.mixer.init()
        pygame.mixer.quit()
        print("✅ Audio System        -> READY")
    except Exception as e:
        print(f"❌ Audio System        -> ERROR: {e}")

    # 4. Rubberband Check
    rb = shutil.which("rubberband") or os.path.exists("rubberband.exe")
    if rb:
        print("✅ Rubberband CLI      -> FOUND (Pitching enabled)")
    else:
        print("❌ Rubberband CLI      -> MISSING (Crucial for tempo adjustment!)")

    print("\n" + "=" * 50)
    
    # 5. Conclusion
    if all(results.values()) and rb:
        print("STATUS: ALL SYSTEMS GO!")
        print("   Start now: streamlit run dashboard.py")
    else:
        print("ACTION REQUIRED:")
        if not results["Music Database"]:
            print("   -> Run: python scripts/library_processor.py to build your database.")
        if not results["Demo Data"]:
            print("   -> Run: python scripts/generate_demo.py")
        if not rb:
            print("   -> Download 'rubberband.exe' and place it in the project root or system path.")

if __name__ == "__main__":
    run_diagnostics()