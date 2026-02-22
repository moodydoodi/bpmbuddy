"""
Optimized Streamlit Dashboard for BPM-Buddy.
Features: Balanced real-time filtering, dynamic pace tracking, 
and professional chart aesthetics suitable for live use and presentation.
"""
import sys
from pathlib import Path
import os
import sqlite3
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import time
import subprocess
import json

# --- PATH SETUP ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system import config as cfg

# --- PAGE CONFIG ---
st.set_page_config(page_title="BPM-Buddy Control Room", layout="wide", page_icon="🏃‍♂️")

# --- UI THEME INJECTION ---
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #083634; }
    [data-testid="stSidebar"] { background-color: #DFF5F4; }
    .now-playing-box { 
        background-color: #DFF5F4; 
        border-left: 10px solid #0B6F6B; 
        padding: 25px; 
        border-radius: 12px; 
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .kpi-card { 
        background-color: #FFFFFF; 
        border: 1px solid #DFF5F4; 
        border-radius: 12px; 
        padding: 20px; 
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    h3 { color: #0B6F6B !important; font-weight: 700 !important; }
    .stMetric { background-color: #f8fcfc; padding: 10px; border-radius: 8px; border: 1px solid #e0f0f0; }
</style>
""", unsafe_allow_html=True)

# --- PROCESS MANAGEMENT ---
def is_running() -> bool:
    pid_file = os.path.join(cfg.OUT_DIR, "backend.pid")
    try:
        if not os.path.exists(pid_file): return False
        with open(pid_file, "r") as f: pid = int(f.read().strip())
        # Standard Windows tasklist check
        return str(pid) in subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True).decode()
    except Exception: return False

def start_system(mode="live"):
    stop_system()
    with open(cfg.CONTROL_JSON, "w") as f:
        json.dump({"running": True, "mode": mode}, f)
    main_script = os.path.join(ROOT, "main.py")
    proc = subprocess.Popen([sys.executable, main_script])
    with open(os.path.join(cfg.OUT_DIR, "backend.pid"), "w") as f: 
        f.write(str(proc.pid))
    time.sleep(1.5)

def stop_system():
    pid_file = os.path.join(cfg.OUT_DIR, "backend.pid")
    if os.path.exists(pid_file):
        try:
            with open(pid_file) as f: pid = int(f.read().strip())
            subprocess.call(f"taskkill /F /T /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception: pass
        os.remove(pid_file)

def delete_db_safely():
    """Attempts to remove the SQLite DB and its WAL sidecars handling Windows file locks."""
    if is_running():
        st.sidebar.error("Stop backend before resetting!")
        return
    for f_path in [cfg.DB_PATH, cfg.DB_PATH + "-wal", cfg.DB_PATH + "-shm"]:
        if os.path.exists(f_path):
            try: os.remove(f_path)
            except: pass
    st.sidebar.success("Database reset!")

def get_telemetry_data() -> pd.DataFrame:
    if not os.path.exists(cfg.DB_PATH): return pd.DataFrame()
    try:
        conn = sqlite3.connect(cfg.DB_PATH)
        # Select context window (300 samples for better responsiveness)
        df = pd.read_sql_query("SELECT * FROM session_log ORDER BY ts DESC LIMIT 300", conn)
        conn.close()
        df = df.sort_values(by="ts").reset_index(drop=True)
        
        if not df.empty:
            # --- CLEAN LIVE FILTERING ---
            # 1. Fill gaps for visual continuity
            df['cadence_smooth'] = df['cadence_smooth'].ffill().fillna(80.0)
            df['effective_bpm'] = df['effective_bpm'].ffill().fillna(80.0)
            
            # 2. Balanced Rolling Window
            # window=8 provides stability without noticeably lagging behind reality
            df['cadence_plot'] = df['cadence_smooth'].rolling(window=8, min_periods=1, center=True).mean()
            df['music_plot'] = df['effective_bpm'].rolling(window=5, min_periods=1, center=True).mean()
            
        return df
    except Exception: return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/running.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    app_mode = st.radio("Operating Mode", ["Simulation (Demo)", "Live Sensor (Phyphox)"])
    mode_arg = "replay" if "Sim" in app_mode else "live"
    
    if is_running():
        st.success("BACKEND RUNNING")
        if st.button("⏹ STOP SYSTEM", use_container_width=True): 
            stop_system(); st.rerun()
    else:
        st.error("BACKEND STOPPED")
        if st.button("▶ START SYSTEM", use_container_width=True): 
            start_system(mode_arg); st.rerun()
    
    st.markdown("---")
    if st.button("Reset Database", use_container_width=True):
        delete_db_safely(); st.rerun()

# --- MAIN UI ---
st.title("BPM-Buddy Dashboard")
df = get_telemetry_data()

if df.empty:
    st.info("Waiting for data stream... Click 'START SYSTEM' in the sidebar.")
else:
    last = df.iloc[-1]
    
    # 1. NOW PLAYING BANNER
    title = str(last.get("selected_title", "Analyzing..."))
    if title in ["None", ""]: title = "Synchronizing DJ Engine..."
    
    st.markdown(f"""
    <div class="now-playing-box">
        <p style='margin:0; font-size: 14px; opacity:0.7; font-weight:bold; letter-spacing:1px;'>🎵 NOW PLAYING</p>
        <h2 style='color:#0B6F6B; margin:0; font-size: 36px; font-weight:800;'>{title}</h2>
        <p style='margin:0; font-size: 16px; opacity:0.8; margin-top:5px;'>
            <b>{last.get('effective_bpm', 0.0):.1f} BPM</b> • 
            Activity: <span style='color:#0B6F6B; font-weight:bold;'>{last.get('activity', 'Active')}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. KPI CARDS
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("State", str(last.get("activity", "Idle")))
    with c2: st.metric("Cadence", f"{last.get('cadence_smooth', 0.0):.0f} SPM")
    with c3: st.metric("Energy", f"{last.get('energy', 0.0):.2f}")
    with c4: st.metric("Sync Score", f"{last.get('score', 0.0):.1f}")

    # 3. CHART & HISTORY
    col_chart, col_hist = st.columns([2.2, 1])
    
    with col_chart:
        st.markdown("### Real-time Synchronization")
        
        # Relative time calculation
        df['Time'] = df['ts'] - df['ts'].iloc[0]
        
        plot_df = df[['Time', 'cadence_plot', 'music_plot']].copy()
        melted = plot_df.melt('Time', var_name='Metric', value_name='Val')
        melted['Metric'] = melted['Metric'].replace({'cadence_plot': 'Runner (SPM)', 'music_plot': 'Music (BPM)'})

        # Song change markers
        df['title_shifted'] = df['selected_title'].shift(1)
        markers = df[(df['selected_title'] != df['title_shifted']) & (df['selected_title'].notna()) & (df['selected_title'] != "")]

        # Line Chart
        lines = alt.Chart(melted).mark_line(strokeWidth=4, interpolate='monotone').encode(
            x=alt.X('Time:Q', title="Time (Seconds)"),
            y=alt.Y('Val:Q', scale=alt.Scale(domain=[50, 200]), title="Frequency (BPM / SPM)"),
            color=alt.Color('Metric:N', scale=alt.Scale(domain=['Runner (SPM)', 'Music (BPM)'], range=['#0B6F6B', '#f4a261']), title=None)
        )
        
        # High-visibility dots for transitions
        dots = alt.Chart(markers).mark_circle(size=150, color='#f4a261', stroke='white', strokeWidth=3).encode(
            x='Time:Q', y='effective_bpm:Q', tooltip=['selected_title']
        )
        
        st.altair_chart((lines + dots).properties(height=420), use_container_width=True)

    with col_hist:
        st.markdown("### Session History")
        hist_df = df.dropna(subset=['selected_title']).copy()
        hist_df = hist_df[hist_df['selected_title'].str.strip() != ""]
        if not hist_df.empty:
            hist_df['title_shifted'] = hist_df['selected_title'].shift(1)
            changes = hist_df[hist_df['selected_title'] != hist_df['title_shifted']].tail(5).iloc[::-1]
            for _, row in changes.iterrows():
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-left: 5px solid #DFF5F4; padding: 12px; margin-bottom: 12px; border-radius: 8px; border: 1px solid #eee;">
                    <p style="margin:0; font-weight: 800; color: #083634;">{row['selected_title']}</p>
                    <p style="margin:0; font-size: 13px; color: #666;">{row['effective_bpm']:.1f} BPM • {row.get('selected_genre', 'General')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("Listening for the first track...")

# Automatic fast refresh
if is_running():
    time.sleep(0.4)
    st.rerun()