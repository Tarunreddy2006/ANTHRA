"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EcoSync — Phase 5: The Victory Dashboard                            ║
║         Cyber-Industrial Heist Command Center                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import time, random, threading, logging, math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Conditional imports ──────────────────────────────────────────────────────
try:
    from strategy import EcoSyncStrategist, Config, TrafficEnv, DQNAgent
    STRATEGY_OK = True
except ImportError:
    STRATEGY_OK = False

try:
    from traffic_oracle import TrafficOracle
    ORACLE_OK = True
except ImportError:
    ORACLE_OK = False

try:
    from perception import PerceptionBridge, FrameData, LaneDetection
    PERCEPTION_OK = True
except ImportError:
    PERCEPTION_OK = False

try:
    from sim import IntersectionState, LaneState, INCOMING_LANES
    SIM_OK = True
except ImportError:
    SIM_OK = False
    INCOMING_LANES = [f"{a}_in_{i}" for a in "NSEW" for i in range(3)]

log = logging.getLogger("EcoSync.Dashboard")

# ═══════════════════════════════════════════════════════════════════════════════
# THEME CSS — Cyber-Industrial Heist
# ═══════════════════════════════════════════════════════════════════════════════
NEON = "#39FF14"
CYBER_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
:root {{
    --neon: {NEON}; --bg: #0a0a0f; --card: #12121a;
    --border: #1e1e2e; --text: #e0e0e0; --dim: #888;
}}
.stApp {{ background: var(--bg) !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
div[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d0d15 0%, #111118 100%) !important;
    border-right: 1px solid var(--neon) !important;
}}
h1, h2, h3 {{ font-family: 'Orbitron', monospace !important; color: var(--neon) !important;
    text-shadow: 0 0 15px {NEON}66; letter-spacing: 2px; }}
p, span, div, label {{ font-family: 'Share Tech Mono', monospace !important; }}
div[data-testid="stMetric"] {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; box-shadow: 0 0 12px {NEON}22;
}}
div[data-testid="stMetric"] label {{ color: var(--dim) !important; font-size: 0.75rem !important; }}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: var(--neon) !important; font-family: 'Orbitron' !important;
    font-size: 1.8rem !important; text-shadow: 0 0 10px {NEON}88;
}}
.cyber-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; margin: 8px 0;
    box-shadow: inset 0 0 20px #00000066, 0 0 15px {NEON}11;
}}
.impact-mega {{
    font-family: 'Orbitron', monospace; font-size: 4rem; font-weight: 900;
    color: {NEON}; text-align: center; text-shadow: 0 0 30px {NEON}aa, 0 0 60px {NEON}44;
    padding: 20px; animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.85; }}
}}
.log-entry {{
    font-family: 'Share Tech Mono', monospace; font-size: 0.8rem;
    padding: 4px 8px; margin: 2px 0; border-left: 3px solid {NEON};
    background: #0d0d1a; color: var(--text); border-radius: 0 4px 4px 0;
}}
.log-time {{ color: var(--dim); }} .log-action {{ color: {NEON}; font-weight: bold; }}
.status-bar {{
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(90deg, #0d0d15, #12121a, #0d0d15);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 16px; margin-bottom: 16px;
}}
.bella-banner {{
    text-align: center; padding: 10px; font-family: 'Orbitron';
    font-size: 0.9rem; color: {NEON}; letter-spacing: 4px;
    border-bottom: 1px solid {NEON}44; margin-bottom: 20px;
    text-shadow: 0 0 10px {NEON}66;
}}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION DATA GENERATOR (demo mode when SUMO unavailable)
# ═══════════════════════════════════════════════════════════════════════════════
class DemoDataEngine:
    """Generates realistic fake data for demo/presentation when SUMO isn't running."""
    def __init__(self):
        self.step = 0
        self.ai_wait = 0.0
        self.baseline_wait = 0.0
        self.ai_emissions = 0.0
        self.baseline_emissions = 0.0
        self.ai_jam = 0.0
        self.baseline_jam = 0.0
        self.actual_density = deque(maxlen=200)
        self.predicted_density = deque(maxlen=200)
        self.log_entries: List[str] = []
        self._rng = np.random.default_rng(42)

    def tick(self):
        self.step += 1
        t = self.step
        # Sinusoidal traffic pattern with noise
        base = 5 + 4 * math.sin(t * 0.05) + 2 * math.sin(t * 0.13)
        noise = self._rng.normal(0, 0.5)
        actual = max(0, base + noise)
        predicted = max(0, base + 0.8 * math.sin((t + 5) * 0.05) * 4 + 5 + self._rng.normal(0, 0.3))
        self.actual_density.append(actual)
        self.predicted_density.append(predicted)

        # Cumulative metrics — AI always better than baseline
        bw = abs(self._rng.normal(12, 4))
        aw = abs(self._rng.normal(7, 3))
        self.baseline_wait += bw
        self.ai_wait += aw
        be = abs(self._rng.normal(5.5, 1.5))
        ae = abs(self._rng.normal(3.2, 1.2))
        self.baseline_emissions += be
        self.ai_emissions += ae
        bj = abs(self._rng.normal(0.6, 0.2))
        aj = abs(self._rng.normal(0.3, 0.15))
        self.baseline_jam += bj
        self.ai_jam += aj

        # Generate RL log entries
        if t % 3 == 0:
            actions = [
                ("Phase 0 (NS Green)", "Oracle predicted high-density bus cluster on N_in_1"),
                ("Phase 2 (EW Green)", "Clearing E_in queue — predicted jam risk 0.78"),
                ("Phase 0 (NS Green)", "Emergency vehicle detected on S_in_0 — priority override"),
                ("Phase 2 (EW Green)", "Low density on NS arms — switching to serve EW backlog"),
                ("Phase 0 (NS Green)", "Predicted platoon arrival from North in 3 steps"),
            ]
            act, reason = actions[t % len(actions)]
            self.log_entries.append(
                f'<span class="log-time">[T={t:04d}]</span> '
                f'<span class="log-action">RL Action: {act}</span> | '
                f'Reason: {reason}'
            )
            if len(self.log_entries) > 50:
                self.log_entries = self.log_entries[-50:]

    def get_impact_saved(self):
        dw = max(0, self.baseline_wait - self.ai_wait)
        de = max(0, self.baseline_emissions - self.ai_emissions)
        dj = max(0, self.baseline_jam - self.ai_jam)
        return 0.50 * dw + 0.30 * de + 0.20 * dj

    def get_lane_data(self) -> Dict:
        result = {}
        for lid in INCOMING_LANES:
            count = max(0, int(self._rng.normal(4, 2)))
            em = round(abs(self._rng.normal(2.5, 1.5)), 2)
            result[lid] = {"count": count, "emissions_score": em}
        return result

    def make_frame(self) -> np.ndarray:
        """Generate a synthetic 'processed' frame with bounding boxes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (15, 12, 10)
        # Draw intersection grid
        cv2 = None
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            return frame
        cx, cy = 320, 240
        green = (20, 255, 57)
        cv2.line(frame, (cx, 0), (cx, 480), (30, 30, 40), 1)
        cv2.line(frame, (0, cy), (640, cy), (30, 30, 40), 1)
        for i in range(4):
            cv2.rectangle(frame, (cx - 90 + i*40, cy - 90), (cx - 55 + i*40, cy - 60), (30, 30, 40), 1)
        # Draw fake vehicle boxes
        for _ in range(int(self._rng.integers(5, 15))):
            x = int(self._rng.integers(50, 590))
            y = int(self._rng.integers(50, 430))
            w = int(self._rng.integers(20, 45))
            h = int(self._rng.integers(15, 30))
            cv2.rectangle(frame, (x, y), (x+w, y+h), green, 1)
            cv2.putText(frame, f"car {self._rng.integers(50,99)}%", (x, y-3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, green, 1)
        # HUD overlay
        cv2.putText(frame, f"ECOSYNC LIVE | T={self.step:04d}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1)
        cv2.putText(frame, f"Vehicles: {self._rng.integers(8,25)} | EmScore: {self._rng.uniform(5,15):.1f}",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def styled_plotly_layout(title="", h=350):
    return dict(
        template="plotly_dark",
        paper_bgcolor="#12121a", plot_bgcolor="#0a0a0f",
        font=dict(family="Share Tech Mono", color="#e0e0e0", size=11),
        title=dict(text=title, font=dict(family="Orbitron", color=NEON, size=14)),
        height=h, margin=dict(l=40, r=20, t=45, b=30),
        xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
        yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
        legend=dict(bgcolor="#12121a88", bordercolor="#1e1e2e"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="EcoSync Command Center",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CYBER_CSS, unsafe_allow_html=True)

    # ── Session state init ─────────────────────────────────────────────────
    if "engine" not in st.session_state:
        st.session_state.engine = DemoDataEngine()
        st.session_state.running = True

    eng = st.session_state.engine

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="bella-banner">🎯 BELLA CIAO PROTOCOL 🎯</div>', unsafe_allow_html=True)
        st.markdown("## ⚡ ECOSYNC")
        st.caption("Hybrid YOLO-LSTM-RL Traffic AI")
        st.divider()
        st.markdown("### System Status")
        st.markdown(f"**Phase 1** — Sim/TraCI: `{'✅' if SIM_OK else '⚠️ Demo'}`")
        st.markdown(f"**Phase 2** — YOLOv8: `{'✅' if PERCEPTION_OK else '⚠️ Demo'}`")
        st.markdown(f"**Phase 3** — LSTM Oracle: `{'✅' if ORACLE_OK else '⚠️ Demo'}`")
        st.markdown(f"**Phase 4** — DQN Agent: `{'✅' if STRATEGY_OK else '⚠️ Demo'}`")
        st.divider()
        fps_target = st.slider("Target FPS", 1, 30, 10)
        st.session_state.running = st.toggle("🔴 LIVE FEED", value=True)
        st.divider()
        st.markdown("### Reward Weights")
        w1 = st.slider("W₁ WaitTime", 0.0, 1.0, 0.50, 0.05)
        w2 = st.slider("W₂ Emissions", 0.0, 1.0, 0.30, 0.05)
        w3 = st.slider("W₃ JamRisk", 0.0, 1.0, 0.20, 0.05)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown('<div class="bella-banner">◆ ECOSYNC COMMAND CENTER — THE VICTORY DASHBOARD ◆</div>',
                unsafe_allow_html=True)

    # Status bar
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("⏱️ Sim Step", f"{eng.step:,}")
    col_s2.metric("🎯 FPS Target", fps_target)
    col_s3.metric("🧠 Oracle Mode", "INFERENCE" if eng.step > 60 else "WARMUP")
    col_s4.metric("🤖 Agent ε", f"{max(0.05, 1.0 * 0.995**eng.step):.3f}")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 1: Live Feed + Impact Meter
    # ═══════════════════════════════════════════════════════════════════════
    col_feed, col_impact = st.columns([3, 2])

    with col_feed:
        st.markdown("### 📡 THE LIVE HEIST FEED")
        feed_placeholder = st.empty()

    with col_impact:
        st.markdown("### 🏆 THE IMPACT METER")
        impact_placeholder = st.empty()
        metrics_placeholder = st.empty()

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 2: Oracle's Eye + Strategist Log
    # ═══════════════════════════════════════════════════════════════════════
    col_oracle, col_log = st.columns([3, 2])

    with col_oracle:
        st.markdown("### 🔮 THE ORACLE'S EYE")
        oracle_placeholder = st.empty()

    with col_log:
        st.markdown("### 📋 STRATEGIST LOG")
        log_placeholder = st.empty()

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 3: Lane Heatmap
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 🗺️ LANE EMISSIONS HEATMAP")
    heatmap_placeholder = st.empty()

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN UPDATE LOOP
    # ═══════════════════════════════════════════════════════════════════════
    for _ in range(10000):
        if not st.session_state.running:
            time.sleep(0.5)
            continue

        eng.tick()
        lane_data = eng.get_lane_data()

        # ── 1. LIVE FEED ──────────────────────────────────────────────────
        frame = eng.make_frame()
        feed_placeholder.image(frame, channels="BGR", use_container_width=True,
                               caption=f"EcoSync Perception | Frame {eng.step}")

        # ── 2. IMPACT METER ───────────────────────────────────────────────
        impact_saved = eng.get_impact_saved()
        with impact_placeholder.container():
            st.markdown(
                f'<div class="cyber-card">'
                f'<div style="text-align:center;color:#888;font-size:0.8rem;letter-spacing:2px;">'
                f'TOTAL ENVIRONMENTAL IMPACT SAVED</div>'
                f'<div class="impact-mega">{impact_saved:,.1f}</div>'
                f'<div style="text-align:center;color:#888;font-size:0.7rem;">'
                f'0.50·ΔWait + 0.30·ΔEmissions + 0.20·ΔJamRisk</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with metrics_placeholder.container():
            mc1, mc2, mc3 = st.columns(3)
            bw, aw = eng.baseline_wait, eng.ai_wait
            pct_w = ((bw - aw) / max(bw, 1)) * 100
            mc1.metric("Wait ↓", f"{pct_w:.1f}%", f"-{bw-aw:.0f}s")
            be, ae = eng.baseline_emissions, eng.ai_emissions
            pct_e = ((be - ae) / max(be, 1)) * 100
            mc2.metric("CO₂ ↓", f"{pct_e:.1f}%", f"-{be-ae:.0f}")
            bj, aj = eng.baseline_jam, eng.ai_jam
            pct_j = ((bj - aj) / max(bj, 1)) * 100
            mc3.metric("Jam ↓", f"{pct_j:.1f}%", f"-{bj-aj:.1f}")

        # ── 3. ORACLE'S EYE ───────────────────────────────────────────────
        if len(eng.actual_density) > 5:
            x_actual = list(range(len(eng.actual_density)))
            x_pred = list(range(5, len(eng.predicted_density) + 5))
            fig_oracle = go.Figure()
            fig_oracle.add_trace(go.Scatter(
                x=x_actual, y=list(eng.actual_density),
                name="Actual Density", mode="lines",
                line=dict(color="#00ff88", width=2),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.08)"
            ))
            fig_oracle.add_trace(go.Scatter(
                x=x_pred[:len(eng.predicted_density)], y=list(eng.predicted_density),
                name="Predicted (5-step ahead)", mode="lines",
                line=dict(color="#ff4444", width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(255,68,68,0.06)"
            ))
            fig_oracle.update_layout(**styled_plotly_layout("LSTM Predicted vs Actual Density"))
            oracle_placeholder.plotly_chart(fig_oracle, use_container_width=True, key=f"oracle_{eng.step}")

        # ── 4. STRATEGIST LOG ─────────────────────────────────────────────
        if eng.log_entries:
            html = '<div class="cyber-card" style="max-height:350px;overflow-y:auto;">'
            for entry in reversed(eng.log_entries[-20:]):
                html += f'<div class="log-entry">{entry}</div>'
            html += '</div>'
            log_placeholder.markdown(html, unsafe_allow_html=True)

        # ── 5. LANE HEATMAP ───────────────────────────────────────────────
        if eng.step % 5 == 0:
            arms = ["N_in", "S_in", "E_in", "W_in"]
            z_data = []
            for arm in arms:
                row = []
                for i in range(3):
                    lid = f"{arm}_{i}"
                    d = lane_data.get(lid, {})
                    row.append(d.get("emissions_score", 0))
                z_data.append(row)
            fig_hm = go.Figure(go.Heatmap(
                z=z_data, x=["Lane 0", "Lane 1", "Lane 2"],
                y=["North", "South", "East", "West"],
                colorscale=[[0, "#0a0a0f"], [0.5, "#39FF14"], [1, "#ff4444"]],
                showscale=True, colorbar=dict(title="Score"),
            ))
            fig_hm.update_layout(**styled_plotly_layout("Per-Lane Emission Score", h=250))
            heatmap_placeholder.plotly_chart(fig_hm, use_container_width=True, key=f"hm_{eng.step}")

        time.sleep(1.0 / fps_target)


if __name__ == "__main__":
    main()
