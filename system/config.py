"""
Configuration module for the BPM-Buddy system.
Defines central file paths, network endpoints, DSP parameters, and system thresholds.
"""
import os

# --- PATHS ---
BASE_DIR = os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "_data")

LIBRARY_FILE = os.path.join(DATA_DIR, "library_full.json")
AUDIO_ROOT = os.path.join(DATA_DIR, "cache_audio")
AUDIO_CACHE_DIR = os.path.join(DATA_DIR, "cache_stretch")

SIM_DATA_FILE = os.path.join(OUT_DIR, "demo_session.csv")
DB_PATH = os.path.join(OUT_DIR, "live_session.db")
CONTROL_JSON = os.path.join(OUT_DIR, "control.json")
STATUS_FILE = os.path.join(OUT_DIR, "status.txt")

MODEL_CNN_PATH = "_modules/activity_cnn.pth"
MODEL_RL_PATH = "_modules/rl_agent_dqn.pth"

# --- NETWORK ---
DEFAULT_PHYPHOX_URL = "http://10.9.4.80:8080" 

# --- DSP & SENSORS ---
TARGET_HZ = 20.0
CNN_WINDOW = 60
CAD_WINDOW = 120
POLL_INTERVAL = 0.03

# --- LOGIC & THRESHOLDS ---
DECISION_EVERY_S = 2.5
CONF_DECISION_TH = 0.35 
CONF_UPDATE_TH = 0.25    

SWITCH_COOLDOWN_S = 12.0 
PENDING_CONFIRM_N = 2
IMPROVE_MARGIN_SCORE = 5.0 

TOP_K = 8
IDLE_TARGET_BPM = 80.0

# --- FEATURE FLAGS ---
USE_RL = True
ENABLE_TIME_STRETCH = True
ENABLE_HARMONIC_MIX = True  
ENABLE_BEAT_ALIGN = True