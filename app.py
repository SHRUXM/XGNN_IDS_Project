# ============================================================
# XGNN-Based Intrusion Detection System
# File: app.py - Cyberpunk Streamlit Web Application
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="XGNN Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Cyberpunk CSS + Animations
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── MATRIX BACKGROUND ── */
.stApp {
    background-color: #020b14 !important;
    background-image:
        radial-gradient(ellipse at 20% 50%,
            rgba(0,255,136,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%,
            rgba(180,0,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse at 60% 80%,
            rgba(0,200,255,0.04) 0%, transparent 60%);
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #020b14; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00ff88, #b400ff);
    border-radius: 2px;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        #030d1a 0%,
        #050f20 50%,
        #030d1a 100%) !important;
    border-right: 1px solid rgba(0,255,136,0.2) !important;
    box-shadow: 4px 0 30px rgba(0,255,136,0.05) !important;
}

section[data-testid="stSidebar"] * {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ff88 !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #00cc6a !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    padding: 4px 0 !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
    text-shadow: 0 0 10px #00ff88 !important;
}

/* ── MAIN HEADER ── */
.cyber-header {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}

.cyber-title {
    font-family: 'Orbitron', monospace !important;
    font-size: clamp(1.8rem, 4vw, 3.2rem) !important;
    font-weight: 900 !important;
    background: linear-gradient(
        135deg,
        #00ff88 0%,
        #00ccff 40%,
        #b400ff 80%,
        #00ff88 100%
    );
    background-size: 200% auto;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: shimmer 4s linear infinite,
               glow-text 2s ease-in-out infinite alternate;
    letter-spacing: 3px;
    line-height: 1.3;
    margin-bottom: 0.5rem;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

@keyframes glow-text {
    from { filter: drop-shadow(0 0 8px rgba(0,255,136,0.4)); }
    to   { filter: drop-shadow(0 0 20px rgba(0,200,255,0.8)); }
}

.cyber-subtitle {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    color: rgba(0,255,136,0.6) !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    margin-top: 0.5rem;
}

/* ── SCANLINE EFFECT ── */
.cyber-header::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,136,0.015) 2px,
        rgba(0,255,136,0.015) 4px
    );
    pointer-events: none;
    animation: scanlines 8s linear infinite;
}

@keyframes scanlines {
    0% { background-position: 0 0; }
    100% { background-position: 0 100px; }
}

/* ── METRIC CARDS ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: linear-gradient(135deg,
        rgba(0,255,136,0.04) 0%,
        rgba(0,30,60,0.8) 100%);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.4s ease;
    animation: card-appear 0.6s ease forwards;
}

.metric-card:nth-child(2) { animation-delay: 0.1s; }
.metric-card:nth-child(3) { animation-delay: 0.2s; }
.metric-card:nth-child(4) { animation-delay: 0.3s; }

@keyframes card-appear {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 2px;
    background: linear-gradient(90deg,
        transparent, #00ff88, transparent);
    animation: scan-line 3s linear infinite;
}

.metric-card:nth-child(2)::before { animation-delay: 0.75s; }
.metric-card:nth-child(3)::before { animation-delay: 1.5s; }
.metric-card:nth-child(4)::before { animation-delay: 2.25s; }

@keyframes scan-line {
    0%   { left: -100%; }
    100% { left: 200%; }
}

.metric-card:hover {
    border-color: rgba(0,255,136,0.7);
    box-shadow: 0 0 25px rgba(0,255,136,0.15),
                inset 0 0 25px rgba(0,255,136,0.03);
    transform: translateY(-3px);
}

.metric-value {
    font-family: 'Orbitron', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #00ff88 !important;
    text-shadow: 0 0 15px rgba(0,255,136,0.5);
    margin: 0.3rem 0;
}

.metric-label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    color: rgba(0,200,255,0.7) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

.metric-delta {
    font-size: 0.7rem !important;
    color: rgba(0,255,136,0.5) !important;
    margin-top: 0.3rem;
}

/* ── SECTION HEADERS ── */
.cyber-section {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #00ff88 !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
    padding: 0.7rem 1rem !important;
    border-left: 3px solid #00ff88 !important;
    background: rgba(0,255,136,0.04) !important;
    margin: 1.5rem 0 1rem 0 !important;
    text-shadow: 0 0 10px rgba(0,255,136,0.4);
}

/* ── PROGRESS BARS ── */
.progress-container {
    margin: 0.6rem 0;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: rgba(0,200,255,0.8);
    margin-bottom: 4px;
    letter-spacing: 1px;
}

.progress-bar-bg {
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.15);
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 2px;
    position: relative;
    animation: fill-bar 1.5s ease forwards;
    transform-origin: left;
}

@keyframes fill-bar {
    from { width: 0% !important; }
}

.progress-gcn {
    background: linear-gradient(90deg, #00ff88, #00ccff);
    box-shadow: 0 0 8px rgba(0,255,136,0.4);
}
.progress-gat {
    background: linear-gradient(90deg, #b400ff, #00ccff);
    box-shadow: 0 0 8px rgba(180,0,255,0.4);
}
.progress-rf {
    background: linear-gradient(90deg, #ff6b00, #ffcc00);
    box-shadow: 0 0 8px rgba(255,107,0,0.3);
}
.progress-mlp {
    background: linear-gradient(90deg, #ff0055, #ff6b00);
    box-shadow: 0 0 8px rgba(255,0,85,0.3);
}

/* ── DATA TABLE ── */
.stDataFrame {
    border: 1px solid rgba(0,255,136,0.2) !important;
    border-radius: 6px !important;
}

.stDataFrame th {
    background: rgba(0,255,136,0.1) !important;
    color: #00ff88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px !important;
}

/* ── IMAGES ── */
.stImage img {
    border: 1px solid rgba(0,255,136,0.2) !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

.stImage img:hover {
    border-color: rgba(0,255,136,0.6) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.15) !important;
}

/* ── BUTTONS ── */
.stButton button {
    background: linear-gradient(135deg,
        rgba(0,255,136,0.1), rgba(0,200,255,0.1)) !important;
    border: 1px solid #00ff88 !important;
    color: #00ff88 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    border-radius: 4px !important;
}

.stButton button:hover {
    background: rgba(0,255,136,0.2) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
    transform: translateY(-2px) !important;
}

/* ── ALERTS ── */
.stSuccess {
    background: rgba(0,255,136,0.07) !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
    border-radius: 6px !important;
}

.stError {
    background: rgba(255,0,85,0.07) !important;
    border: 1px solid rgba(255,0,85,0.3) !important;
    border-radius: 6px !important;
}

.stInfo {
    background: rgba(0,200,255,0.07) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    border-radius: 6px !important;
}

.stWarning {
    background: rgba(255,180,0,0.07) !important;
    border: 1px solid rgba(255,180,0,0.3) !important;
    border-radius: 6px !important;
}

/* ── SLIDERS ── */
.stSlider .stSlider > div {
    color: #00ff88 !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0,255,136,0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── CANVAS MATRIX ── */
#matrix-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: -1;
    opacity: 0.03;
    pointer-events: none;
}

/* ── SIDEBAR LOGO ── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0;
}

.sidebar-logo-text {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 900;
    color: #00ff88;
    text-shadow: 0 0 15px rgba(0,255,136,0.5);
    letter-spacing: 3px;
}

.sidebar-info {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: rgba(0,255,136,0.5);
    letter-spacing: 1px;
    line-height: 1.8;
    margin-top: 0.5rem;
}

/* ── GLITCH ANIMATION ── */
.glitch {
    position: relative;
    animation: glitch 5s infinite;
}

@keyframes glitch {
    0%, 90%, 100% { transform: translate(0); }
    92% { transform: translate(-2px, 1px); filter: hue-rotate(20deg); }
    94% { transform: translate(2px, -1px); filter: hue-rotate(-20deg); }
    96% { transform: translate(0); }
}

/* ── STATUS BADGE ── */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #00ff88;
    letter-spacing: 2px;
}

.status-dot {
    width: 8px; height: 8px;
    background: #00ff88;
    border-radius: 50%;
    box-shadow: 0 0 6px #00ff88;
    animation: pulse-dot 1.5s ease infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}
/* ── HIDE SIDEBAR TOGGLE ── */
button[data-testid="collapsedControl"] {
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* ── CUSTOM SIDEBAR ARROW ── */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}
</style>

<!-- Matrix Rain Canvas -->
<canvas id="matrix-canvas"></canvas>
<script>
const canvas = document.getElementById('matrix-canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノ';
const fontSize = 14;
const cols = Math.floor(canvas.width / fontSize);
const drops = Array(cols).fill(1);
function drawMatrix() {
    ctx.fillStyle = 'rgba(2,11,20,0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#00ff88';
    ctx.font = fontSize + 'px monospace';
    drops.forEach((y, i) => {
        const char = chars[Math.floor(Math.random()*chars.length)];
        ctx.fillText(char, i*fontSize, y*fontSize);
        if (y*fontSize > canvas.height && Math.random() > 0.975)
            drops[i] = 0;
        drops[i]++;
    });
}
setInterval(drawMatrix, 50);
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
</script>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text glitch">⬡ XGNN IDS</div>
        <div class="sidebar-info">
            INTRUSION DETECTION SYSTEM<br>
            NEURAL NETWORK v1.0<br>
            ─────────────────────────<br>
            AUTHOR:<br>
            SHREYAS SANTOSH SHINDE<br>
            ─────────────────────────<br>
            GUIDE:<br>
            DR. PRABHA KADAM<br>
            ─────────────────────────<br>
            KIRTI COLLEGE OF ARTS,<br>
            SCIENCE & COMMERCE<br>
            MUMBAI · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="status-online">
        <div class="status-dot"></div>
        SYSTEM ONLINE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        [
            "⬡ HOME",
            "◈ MODEL RESULTS",
            "◉ EXPLAINABILITY",
            "▶ LIVE PREDICTION",
            "◆ MODEL COMPARISON",
            "📂 CUSTOM DATASET"
        ],
        label_visibility="visible"
    )


# ============================================================
# Helper
# ============================================================

def show_image(path, caption=""):
    if os.path.exists(path):
        st.image(path, caption=caption,
                 width="stretch")
    else:
        st.warning(f"⚠ Image not found: {path}")


def cyber_header(text):
    st.markdown(
        f'<div class="cyber-section">{text}</div>',
        unsafe_allow_html=True
    )


def progress_bar(label, value, css_class,
                 show_pct=True):
    pct = f"{value*100:.2f}%" if show_pct else ""
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-label">
            <span>{label}</span>
            <span>{pct}</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill {css_class}"
                 style="width:{value*100}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1 — HOME
# ============================================================

if page == "⬡ HOME":

    st.markdown("""
    <div class="cyber-header">
        <div class="cyber-title glitch">
            EXPLAINABLE GRAPH<br>NEURAL NETWORK IDS
        </div>
        <div class="cyber-subtitle">
            ── MSc Computer Science · Kirti College Mumbai ──
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">GCN ACCURACY</div>
            <div class="metric-value">93.61%</div>
            <div class="metric-delta">✦ EXPLAINABLE</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">GAT ACCURACY</div>
            <div class="metric-value">92.72%</div>
            <div class="metric-delta">✦ EXPLAINABLE</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">GRAPH NODES</div>
            <div class="metric-value">25,192</div>
            <div class="metric-delta">✦ CONNECTIONS</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">FEATURES</div>
            <div class="metric-value">41</div>
            <div class="metric-delta">✦ TRAFFIC FEATURES</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        cyber_header("◈ PROJECT OVERVIEW")
        st.markdown("""
        <div style="font-family:'Rajdhani',sans-serif;
                    color:rgba(200,230,255,0.85);
                    line-height:1.8; font-size:1rem;">
        This project proposes an <b style="color:#00ff88">
        Explainable Graph Neural Network (XGNN)</b> based
        <b style="color:#00ccff">Network Intrusion Detection
        System</b> that detects cyberattacks with high accuracy
        while providing <b style="color:#b400ff">
        human-understandable explanations</b> for every
        prediction.<br><br>
        Traditional ML models function as <b style="color:
        #ff6b00">black boxes</b> — they cannot explain WHY a
        connection is flagged. In cybersecurity,
        <b style="color:#00ff88">explainability is as critical
        as accuracy.</b>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        cyber_header("◉ METHODOLOGY")
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;
                    font-size:0.82rem; line-height:2;
                    color:rgba(0,255,136,0.8);">
        ► DATA PREPROCESSING<br>
        &nbsp;&nbsp;KDD Cup Dataset · 25,192 connections<br>
        ► GRAPH CONSTRUCTION<br>
        &nbsp;&nbsp;Nodes=Connections · Edges=Similar pairs<br>
        ► GNN MODELS<br>
        &nbsp;&nbsp;Graph Convolutional Network (GCN)<br>
        &nbsp;&nbsp;Graph Attention Network (GAT)<br>
        ► EXPLAINABILITY<br>
        &nbsp;&nbsp;Feature importance · Subgraph viz<br>
        &nbsp;&nbsp;Attention weight analysis<br>
        ► EVALUATION<br>
        &nbsp;&nbsp;vs Random Forest · MLP baseline
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    cyber_header("🌐 NETWORK TRAFFIC GRAPH")
    show_image("outputs/network_graph.png",
               "Network traffic as graph — "
               "Red=Attack · Blue=Normal")


# ============================================================
# PAGE 2 — MODEL RESULTS
# ============================================================

elif page == "◈ MODEL RESULTS":

    st.markdown(
        '<div class="cyber-title" '
        'style="font-size:1.8rem;text-align:left;'
        'padding:1rem 0;">◈ MODEL RESULTS</div>',
        unsafe_allow_html=True
    )

    cyber_header("▸ PERFORMANCE METRICS")

    # Progress bars
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\','
            'monospace;color:#00ccff;font-size:0.8rem;'
            'letter-spacing:2px;margin-bottom:0.8rem;">'
            '── GNN MODELS ──</div>',
            unsafe_allow_html=True
        )
        progress_bar("GCN  ACCURACY", 0.9361,
                     "progress-gcn")
        progress_bar("GCN  F1-SCORE", 0.9302,
                     "progress-gcn")
        progress_bar("GCN  ROC-AUC", 0.9856,
                     "progress-gcn")
        st.markdown("<br>", unsafe_allow_html=True)
        progress_bar("GAT  ACCURACY", 0.9272,
                     "progress-gat")
        progress_bar("GAT  F1-SCORE", 0.9199,
                     "progress-gat")
        progress_bar("GAT  ROC-AUC", 0.9732,
                     "progress-gat")

    with col2:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\','
            'monospace;color:#ff6b00;font-size:0.8rem;'
            'letter-spacing:2px;margin-bottom:0.8rem;">'
            '── BASELINE MODELS ──</div>',
            unsafe_allow_html=True
        )
        progress_bar("RF   ACCURACY", 0.9962,
                     "progress-rf")
        progress_bar("RF   F1-SCORE", 0.9959,
                     "progress-rf")
        progress_bar("RF   ROC-AUC", 0.9999,
                     "progress-rf")
        st.markdown("<br>", unsafe_allow_html=True)
        progress_bar("MLP  ACCURACY", 0.9940,
                     "progress-mlp")
        progress_bar("MLP  F1-SCORE", 0.9936,
                     "progress-mlp")
        progress_bar("MLP  ROC-AUC", 0.9988,
                     "progress-mlp")

    st.markdown("---")

    # Results table
    cyber_header("▸ FULL COMPARISON TABLE")
    results_df = pd.DataFrame({
        'MODEL': ['GCN', 'GAT',
                  'Random Forest', 'MLP'],
        'ACCURACY': ['93.61%', '92.72%',
                     '99.62%', '99.40%'],
        'PRECISION': ['0.9471', '0.9436',
                      '0.9979', '0.9936'],
        'RECALL': ['0.9140', '0.8974',
                   '0.9940', '0.9936'],
        'F1': ['0.9302', '0.9199',
               '0.9959', '0.9936'],
        'ROC-AUC': ['0.9856', '0.9732',
                    '0.9999', '0.9988'],
        'EXPLAINABLE': ['✅ YES', '✅ YES',
                        '❌ NO', '❌ NO']
    })
    st.dataframe(results_df,
                 use_container_width=True,
                 hide_index=True)

    st.info(
        "💡 Random Forest achieves higher raw accuracy "
        "but GCN/GAT provide explainable decisions — "
        "critical for real-world cybersecurity deployment."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        cyber_header("▸ GCN TRAINING CURVES")
        show_image("outputs/gcn_training_results.png")
    with col2:
        cyber_header("▸ GAT TRAINING CURVES")
        show_image("outputs/gat_training_results.png")

    st.markdown("---")
    cyber_header("▸ CONFUSION MATRICES")
    show_image("outputs/confusion_matrices.png")

    st.markdown("---")
    cyber_header("▸ ROC CURVES")
    show_image("outputs/roc_curves.png")


# ============================================================
# PAGE 3 — EXPLAINABILITY
# ============================================================

elif page == "◉ EXPLAINABILITY":

    st.markdown(
        '<div class="cyber-title" '
        'style="font-size:1.8rem;text-align:left;'
        'padding:1rem 0;">◉ EXPLAINABILITY ANALYSIS</div>',
        unsafe_allow_html=True
    )

    cyber_header("▸ FEATURE IMPORTANCE")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\','
        'monospace;color:rgba(0,200,255,0.7);'
        'font-size:0.8rem;letter-spacing:1px;">'
        'Most critical network features '
        'for intrusion detection:</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\','
            'monospace;color:#00ff88;font-size:0.75rem;'
            'letter-spacing:2px;margin:0.5rem 0;">'
            '── GCN FEATURES ──</div>',
            unsafe_allow_html=True
        )
        show_image("outputs/gcn_feature_importance.png")
    with col2:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\','
            'monospace;color:#b400ff;font-size:0.75rem;'
            'letter-spacing:2px;margin:0.5rem 0;">'
            '── GAT FEATURES ──</div>',
            unsafe_allow_html=True
        )
        show_image("outputs/gat_feature_importance.png")

    st.success(
        "🔑 KEY FINDING: `wrong_fragment` is the #1 "
        "feature in BOTH models — a classic network "
        "attack indicator confirmed by both GCN and GAT!"
    )

    st.markdown("---")
    cyber_header("▸ ATTACK SUBGRAPH EXPLANATION")
    show_image("outputs/gcn_subgraph_node_2.png",
               "Yellow=Target · Red=Attack · Blue=Normal")

    st.markdown("---")
    cyber_header("▸ GAT ATTENTION WEIGHTS")

    col1, col2 = st.columns(2)
    with col1:
        show_image("outputs/gat_attention_weights.png")
    with col2:
        show_image("outputs/attention_per_head.png")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        show_image("outputs/top_attended_nodes.png")
    with col2:
        show_image(
            "outputs/attack_vs_normal_attention.png"
        )


# ============================================================
# PAGE 4 — LIVE PREDICTION
# ============================================================

elif page == "▶ LIVE PREDICTION":

    st.markdown(
        '<div class="cyber-title" '
        'style="font-size:1.8rem;text-align:left;'
        'padding:1rem 0;">▶ LIVE INTRUSION DETECTION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-family:\'Share Tech Mono\','
        'monospace;color:rgba(0,200,255,0.7);'
        'font-size:0.82rem;letter-spacing:1px;">'
        'Adjust network parameters to simulate '
        'connections and detect intrusions in real time.'
        '</div><br>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        cyber_header("CONNECTION")
        duration = st.slider("Duration", 0, 100, 0)
        protocol = st.selectbox(
            "Protocol", ["tcp", "udp", "icmp"]
        )
        wrong_fragment = st.slider(
            "Wrong Fragments", 0, 10, 0
        )
        hot = st.slider("Hot Indicators", 0, 30, 0)
        logged_in = st.selectbox("Logged In", [0, 1])

    with col2:
        cyber_header("TRAFFIC STATS")
        count = st.slider("Count", 0, 512, 1)
        srv_count = st.slider(
            "Service Count", 0, 512, 1
        )
        same_srv_rate = st.slider(
            "Same Srv Rate", 0.0, 1.0, 1.0
        )
        serror_rate = st.slider(
            "SYN Error Rate", 0.0, 1.0, 0.0
        )
        rerror_rate = st.slider(
            "REJ Error Rate", 0.0, 1.0, 0.0
        )

    with col3:
        cyber_header("HOST STATS")
        dst_host_count = st.slider(
            "Dst Host Count", 0, 255, 100
        )
        dst_host_srv_count = st.slider(
            "Dst Host Srv Count", 0, 255, 100
        )
        dst_host_same_srv_rate = st.slider(
            "Dst Host Srv Rate", 0.0, 1.0, 1.0
        )
        dst_host_serror_rate = st.slider(
            "Dst SYN Error", 0.0, 1.0, 0.0
        )
        dst_host_rerror_rate = st.slider(
            "Dst REJ Error", 0.0, 1.0, 0.0
        )

    st.markdown("---")

    if st.button("⚡ ANALYZE CONNECTION",
                 use_container_width=True):

        attack_score = 0
        reasons = []

        if wrong_fragment > 0:
            attack_score += 3
            reasons.append(
                f"⚠ Wrong fragments: {wrong_fragment} "
                f"— strong attack indicator"
            )
        if serror_rate > 0.5:
            attack_score += 2
            reasons.append(
                f"⚠ SYN error rate: {serror_rate:.2f} "
                f"— possible SYN flood attack"
            )
        if rerror_rate > 0.5:
            attack_score += 2
            reasons.append(
                f"⚠ REJ error rate: {rerror_rate:.2f} "
                f"— port scanning detected"
            )
        if dst_host_serror_rate > 0.5:
            attack_score += 2
            reasons.append(
                f"⚠ Host SYN error: "
                f"{dst_host_serror_rate:.2f} "
                f"— host under attack"
            )
        if dst_host_rerror_rate > 0.5:
            attack_score += 1
            reasons.append(
                f"⚠ Host REJ error: "
                f"{dst_host_rerror_rate:.2f}"
            )
        if hot > 5:
            attack_score += 1
            reasons.append(
                f"⚠ Hot indicators: {hot} — "
                f"suspicious activity"
            )
        if count > 200:
            attack_score += 1
            reasons.append(
                f"⚠ High count: {count} — "
                f"possible flood"
            )

        confidence = min(
            0.5 + (attack_score * 0.06), 0.99
        )

        if attack_score >= 3:
            st.markdown(f"""
            <div style="background:rgba(255,0,85,0.08);
                        border:1px solid rgba(255,0,85,0.4);
                        border-radius:8px;padding:1.5rem;
                        margin:1rem 0;
                        font-family:'Orbitron',monospace;">
                <div style="color:#ff0055;font-size:1.4rem;
                            font-weight:700;
                            text-shadow:0 0 15px #ff0055;
                            letter-spacing:3px;">
                    🚨 INTRUSION DETECTED
                </div>
                <div style="color:rgba(255,100,100,0.8);
                            font-size:0.85rem;margin-top:0.5rem;
                            font-family:'Share Tech Mono',
                            monospace;letter-spacing:2px;">
                    CONFIDENCE: {confidence*100:.1f}% ·
                    THREAT LEVEL: {'CRITICAL' if attack_score >= 6 else 'HIGH'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            cyber_header("▸ THREAT INDICATORS")
            for r in reasons:
                st.markdown(
                    f'<div style="font-family:\'Share Tech '
                    f'Mono\',monospace;color:#ff6b6b;'
                    f'font-size:0.82rem;padding:4px 0;'
                    f'letter-spacing:1px;">{r}</div>',
                    unsafe_allow_html=True
                )
        else:
            normal_conf = min(0.9 - attack_score*0.05,
                              0.99)
            st.markdown(f"""
            <div style="background:rgba(0,255,136,0.06);
                        border:1px solid rgba(0,255,136,0.3);
                        border-radius:8px;padding:1.5rem;
                        margin:1rem 0;
                        font-family:'Orbitron',monospace;">
                <div style="color:#00ff88;font-size:1.4rem;
                            font-weight:700;
                            text-shadow:0 0 15px #00ff88;
                            letter-spacing:3px;">
                    ✅ NORMAL TRAFFIC
                </div>
                <div style="color:rgba(0,255,136,0.7);
                            font-size:0.85rem;margin-top:0.5rem;
                            font-family:'Share Tech Mono',
                            monospace;letter-spacing:2px;">
                    CONFIDENCE: {normal_conf*100:.1f}% ·
                    STATUS: SAFE
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Feature risk table
        st.markdown("---")
        cyber_header("▸ FEATURE RISK ANALYSIS")
        risk_df = pd.DataFrame({
            'FEATURE': [
                'Wrong Fragments', 'SYN Error Rate',
                'REJ Error Rate', 'Hot Indicators',
                'Connection Count', 'Logged In'
            ],
            'VALUE': [
                wrong_fragment, serror_rate,
                rerror_rate, hot, count, logged_in
            ],
            'RISK': [
                '🔴 HIGH' if wrong_fragment > 0
                else '🟢 LOW',
                '🔴 HIGH' if serror_rate > 0.5
                else '🟢 LOW',
                '🔴 HIGH' if rerror_rate > 0.5
                else '🟢 LOW',
                '🟡 MED' if hot > 5 else '🟢 LOW',
                '🟡 MED' if count > 200 else '🟢 LOW',
                '🟢 SAFE' if logged_in == 1
                else '🟡 UNKNOWN'
            ]
        })
        st.dataframe(risk_df,
                     use_container_width=True,
                     hide_index=True)


# ============================================================
# PAGE 5 — MODEL COMPARISON
# ============================================================

elif page == "◆ MODEL COMPARISON":

    st.markdown(
        '<div class="cyber-title" '
        'style="font-size:1.8rem;text-align:left;'
        'padding:1rem 0;">◆ MODEL COMPARISON</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        cyber_header("▸ RADAR CHART")
        show_image("outputs/radar_chart.png")
    with col2:
        cyber_header("▸ METRICS HEATMAP")
        show_image("outputs/metrics_heatmap.png")

    st.markdown("---")
    cyber_header("▸ ACCURACY vs EXPLAINABILITY")
    show_image("outputs/accuracy_vs_explainability.png",
               "Key research argument — "
               "XGNN achieves high explainability "
               "with competitive accuracy")

    st.markdown("---")
    cyber_header("▸ FULL MODEL COMPARISON")
    show_image("outputs/model_comparison.png")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        cyber_header("▸ WHY XGNN WINS")
        st.success("""
        ✅ Explainable decisions
        ✅ Shows WHICH features triggered alert
        ✅ Visualizes attack propagation paths
        ✅ GAT attention maps critical connections
        ✅ Suitable for real-world deployment
        ✅ Graph-aware — understands relationships
        """)
    with col2:
        cyber_header("▸ BASELINE LIMITATIONS")
        st.warning("""
        ❌ Random Forest — black box
        ❌ MLP — no feature attribution
        ❌ Cannot justify decisions to analysts
        ❌ Difficult to audit in production
        ❌ No graph structure awareness
        ❌ Cannot explain attack pathways
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;
                font-family:'Share Tech Mono',monospace;
                color:rgba(0,255,136,0.4);
                font-size:0.75rem;letter-spacing:3px;
                padding:1rem;">
        XGNN IDS · SHREYAS SANTOSH SHINDE ·
        KIRTI COLLEGE MUMBAI · 2026
    </div>
    """, unsafe_allow_html=True)
# ============================================================
# PAGE 6 — CUSTOM DATASET UPLOAD
# ============================================================

elif page == "📂 CUSTOM DATASET":

    st.markdown(
        '<div class="cyber-title" '
        'style="font-size:1.8rem;text-align:left;'
        'padding:1rem 0;">📂 CUSTOM DATASET ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;
                color:rgba(0,200,255,0.7);font-size:0.82rem;
                letter-spacing:1px;margin-bottom:1rem;">
        Upload any network traffic CSV file and get
        instant intrusion detection predictions —
        no configuration needed!
    </div>
    """, unsafe_allow_html=True)

    # ── Simple Upload ──
    cyber_header("▸ UPLOAD YOUR DATASET")

    uploaded_file = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="Supports: KDD Cup, NSL-KDD, "
             "CIC-IDS-2017, UNSW-NB15, "
             "or any network traffic CSV"
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            st.stop()

        # ── Cybersecurity Dataset Validator ──
        REQUIRED_KEYWORDS = [
            'duration', 'protocol', 'service', 'flag',
            'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'count', 'serror', 'rerror',
            'attack', 'label', 'class', 'category',
            'packet', 'port', 'ip', 'tcp', 'udp', 'icmp',
            'flow', 'bytes', 'traffic', 'connection',
            'syn', 'ack', 'fin', 'rst', 'intrusion'
        ]

        col_lower = [c.lower() for c in df.columns]
        matched = [
            kw for kw in REQUIRED_KEYWORDS
            if any(kw in col for col in col_lower)
        ]

        if len(matched) < 3:
            st.error("❌ INVALID DATASET — This does not appear to be a cybersecurity/network traffic dataset.")
            st.markdown("""
            <div style="background:rgba(255,0,85,0.06);
                        border:1px solid rgba(255,0,85,0.3);
                        border-radius:8px;padding:1.5rem;
                        font-family:'Share Tech Mono',monospace;
                        font-size:0.82rem;
                        color:rgba(255,100,100,0.9);
                        letter-spacing:1px;line-height:2;">
                ⚠ SYSTEM ONLY ACCEPTS NETWORK TRAFFIC DATASETS<br>
                ────────────────────────────────────────<br>
                ✅ ALLOWED: KDD Cup 1999, NSL-KDD, CIC-IDS-2017, UNSW-NB15<br>
                ❌ REJECTED: Sales data, medical data, general CSV files<br>
                ────────────────────────────────────────<br>
                YOUR FILE COLUMNS DETECTED:<br>
            """ + "<br>".join([f"&nbsp;&nbsp;• {c}" for c in df.columns[:10]]) + """
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        st.success(
            f"✅ Valid cybersecurity dataset detected! "
            f"{df.shape[0]} rows · "
            f"{df.shape[1]} columns · "
            f"{len(matched)} network features matched"
        )

        # ── Preview ──
        cyber_header("▸ DATASET PREVIEW")
        st.dataframe(
            df.head(5),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        if st.button("⚡ ANALYZE DATASET",
                     use_container_width=True):

            with st.spinner(
                "🔍 Analyzing... please wait"
            ):
                try:
                    import torch
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import networkx as nx
                    from sklearn.preprocessing import (
                        MinMaxScaler, LabelEncoder
                    )
                    from sklearn.ensemble import (
                        IsolationForest,
                        RandomForestClassifier
                    )

                    # ── Auto Preprocessing ──
                    analysis_df = df.copy()

                    # Auto detect and drop
                    # label-like columns
                    label_keywords = [
                        'class', 'label', 'attack',
                        'target', 'category',
                        'type', 'outcome'
                    ]
                    label_col_found = None
                    for col in analysis_df.columns:
                        if any(
                            kw in col.lower()
                            for kw in label_keywords
                        ):
                            label_col_found = col
                            break

                    if label_col_found:
                        true_labels = analysis_df[
                            label_col_found
                        ].copy()
                        analysis_df.drop(
                            columns=[label_col_found],
                            inplace=True
                        )
                        st.info(
                            f"🔍 Auto-detected label "
                            f"column: `{label_col_found}`"
                        )
                    else:
                        true_labels = None

                    # Auto encode categorical columns
                    le = LabelEncoder()
                    for col in analysis_df.select_dtypes(
                        include=['object']
                    ).columns:
                        try:
                            analysis_df[col] = \
                                le.fit_transform(
                                    analysis_df[col]
                                    .astype(str)
                                )
                        except:
                            analysis_df.drop(
                                columns=[col],
                                inplace=True,
                                errors='ignore'
                            )

                    # Fill missing/infinite values
                    analysis_df.fillna(0, inplace=True)
                    analysis_df.replace(
                        [np.inf, -np.inf],
                        0, inplace=True
                    )

                    feature_cols_up = list(
                        analysis_df.columns
                    )

                    # Normalize
                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(analysis_df)

                    # ── Isolation Forest ──
                    iso = IsolationForest(
                        contamination=0.1,
                        random_state=42,
                        n_estimators=100
                    )
                    iso.fit(X)
                    scores = iso.decision_function(X)
                    predictions = iso.predict(X)

                    pred_labels = np.where(
                        predictions == -1,
                        'ATTACK', 'NORMAL'
                    )
                    confidence_scores = np.abs(scores)
                    if confidence_scores.max() > 0:
                        confidence_scores = (
                            confidence_scores /
                            confidence_scores.max()
                        ) * 100

                    # ── Summary Metrics ──
                    cyber_header("▸ DETECTION RESULTS")

                    attack_count = (
                        pred_labels == 'ATTACK'
                    ).sum()
                    normal_count = (
                        pred_labels == 'NORMAL'
                    ).sum()
                    total = len(pred_labels)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                TOTAL CONNECTIONS
                            </div>
                            <div class="metric-value"
                                 style="font-size:1.5rem">
                                {total:,}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card"
                             style="border-color:
                             rgba(255,0,85,0.5);">
                            <div class="metric-label"
                                 style="color:#ff6b6b">
                                ATTACKS DETECTED
                            </div>
                            <div class="metric-value"
                                 style="color:#ff0055;
                                 font-size:1.5rem;
                                 text-shadow:
                                 0 0 10px #ff0055">
                                {attack_count:,}
                            </div>
                            <div class="metric-delta"
                                 style="color:
                                 rgba(255,0,85,0.6)">
                                {attack_count/total*100:.1f}%
                                OF TRAFFIC
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card"
                             style="border-color:
                             rgba(0,255,136,0.5);">
                            <div class="metric-label">
                                NORMAL TRAFFIC
                            </div>
                            <div class="metric-value"
                                 style="font-size:1.5rem">
                                {normal_count:,}
                            </div>
                            <div class="metric-delta">
                                {normal_count/total*100:.1f}%
                                OF TRAFFIC
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(
                        "<br>", unsafe_allow_html=True
                    )

                    # ── Results Table ──
                    cyber_header(
                        "▸ PER-ROW PREDICTIONS"
                    )

                    results_df = df.copy()
                    results_df['🔍 PREDICTION'] = \
                        pred_labels
                    results_df['📊 CONFIDENCE'] = \
                        confidence_scores.round(1)\
                        .astype(str) + '%'

                    # Move prediction cols to front
                    front_cols = [
                        '🔍 PREDICTION',
                        '📊 CONFIDENCE'
                    ]
                    other_cols = [
                        c for c in results_df.columns
                        if c not in front_cols
                    ]
                    results_df = results_df[
                        front_cols + other_cols
                    ]

                    st.dataframe(
                        results_df.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                    if len(results_df) > 100:
                        st.caption(
                            f"Showing first 100 of "
                            f"{len(results_df):,} rows. "
                            f"Download CSV for full results."
                        )

                    st.markdown("---")

                    # ── Feature Importance ──
                    cyber_header(
                        "▸ FEATURE IMPORTANCE"
                    )

                    rf = RandomForestClassifier(
                        n_estimators=50,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf.fit(X, predictions)
                    importances = rf.feature_importances_
                    top_n = min(15, len(feature_cols_up))
                    indices = np.argsort(
                        importances
                    )[::-1][:top_n]

                    top_features = [
                        feature_cols_up[i]
                        for i in indices
                    ]
                    top_importance = importances[indices]

                    fig, ax = plt.subplots(
                        figsize=(10, 6),
                        facecolor='#020b14'
                    )
                    ax.set_facecolor('#020b14')

                    colors_bar = plt.cm.plasma(
                        np.linspace(
                            0.2, 0.9,
                            len(top_features)
                        )
                    )
                    ax.barh(
                        range(len(top_features)),
                        top_importance[::-1],
                        color=colors_bar[::-1],
                        edgecolor='none',
                        height=0.7
                    )
                    ax.set_yticks(
                        range(len(top_features))
                    )
                    ax.set_yticklabels(
                        top_features[::-1],
                        color='#00ff88', fontsize=9
                    )
                    ax.set_xlabel(
                        'Importance Score',
                        color='#00ccff'
                    )
                    ax.set_title(
                        f'Top {top_n} Important Features',
                        color='#00ff88',
                        fontsize=13,
                        fontweight='bold'
                    )
                    ax.tick_params(colors='#00ccff')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#00ff88')
                    ax.grid(
                        True, alpha=0.1,
                        color='#00ff88', axis='x'
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.markdown("---")

                    # ── Graph Visualization ──
                    cyber_header(
                        "▸ NETWORK GRAPH"
                    )

                    sample_size = min(150, len(X))
                    sample_idx = np.random.choice(
                        len(X), sample_size,
                        replace=False
                    )

                    G_up = nx.DiGraph()
                    for idx in sample_idx:
                        G_up.add_node(
                            int(idx),
                            label=pred_labels[idx]
                        )
                    for i in range(
                        len(sample_idx) - 1
                    ):
                        G_up.add_edge(
                            int(sample_idx[i]),
                            int(sample_idx[i+1])
                        )

                    fig2, ax2 = plt.subplots(
                        figsize=(10, 7),
                        facecolor='#020b14'
                    )
                    ax2.set_facecolor('#020b14')

                    pos = nx.spring_layout(
                        G_up, seed=42, k=1.5
                    )
                    node_colors = [
                        '#ff0055'
                        if pred_labels[n] == 'ATTACK'
                        else '#00ff88'
                        for n in G_up.nodes()
                    ]
                    nx.draw_networkx_nodes(
                        G_up, pos,
                        node_color=node_colors,
                        node_size=80,
                        alpha=0.9, ax=ax2
                    )
                    nx.draw_networkx_edges(
                        G_up, pos,
                        edge_color='#00ccff',
                        width=0.5,
                        arrows=False, ax=ax2
                    )
                    ax2.set_title(
                        'Network Traffic Graph — '
                        'Red=Attack · Green=Normal',
                        color='#00ff88',
                        fontsize=12,
                        fontweight='bold'
                    )
                    ax2.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                    st.markdown("---")

                    # ── Download ──
                    cyber_header(
                        "▸ DOWNLOAD RESULTS"
                    )

                    csv_out = results_df.to_csv(
                        index=False
                    ).encode('utf-8')

                    st.download_button(
                        label="⬇ DOWNLOAD FULL "
                              "PREDICTIONS AS CSV",
                        data=csv_out,
                        file_name=(
                            "xgnn_predictions.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.success(
                        f"✅ Analysis complete! "
                        f"Detected {attack_count:,} "
                        f"attacks out of "
                        f"{total:,} connections."
                    )

                except Exception as e:
                    st.error(
                        f"❌ Analysis error: {str(e)}"
                    )
                    st.info(
                        "💡 Make sure your CSV contains "
                        "numerical network traffic features."
                    )

    else:
        # ── Empty State ──
        st.markdown("""
        <div style="background:rgba(0,255,136,0.03);
                    border:1px dashed
                    rgba(0,255,136,0.2);
                    border-radius:8px;
                    padding:3rem;text-align:center;
                    margin:2rem 0;">
            <div style="font-family:'Orbitron',monospace;
                        font-size:1.5rem;color:#00ff88;
                        text-shadow:0 0 15px
                        rgba(0,255,136,0.4);
                        margin-bottom:1.5rem;">
                📂 DROP YOUR CSV HERE
            </div>
            <div style="font-family:'Share Tech Mono',
                        monospace;font-size:0.8rem;
                        color:rgba(0,200,255,0.6);
                        line-height:2.2;
                        letter-spacing:1px;">
                JUST UPLOAD — NO SETUP NEEDED<br>
                THE SYSTEM DOES EVERYTHING AUTOMATICALLY<br>
                ────────────────────────────<br>
                ► KDD CUP 1999 &nbsp;✅ &nbsp;<a href="https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data" target="_blank" style="color:#00ff88;text-decoration:none;border-bottom:1px solid rgba(0,255,136,0.4);">⬇ DOWNLOAD</a><br>
                ► NSL-KDD &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ &nbsp;<a href="https://www.unb.ca/cic/datasets/nsl.html" target="_blank" style="color:#00ff88;text-decoration:none;border-bottom:1px solid rgba(0,255,136,0.4);">⬇ DOWNLOAD</a><br>
                ► CIC-IDS-2017 &nbsp;✅ &nbsp;<a href="https://www.unb.ca/cic/datasets/ids-2017.html" target="_blank" style="color:#00ff88;text-decoration:none;border-bottom:1px solid rgba(0,255,136,0.4);">⬇ DOWNLOAD</a><br>
                ► UNSW-NB15 &nbsp;&nbsp;&nbsp;✅ &nbsp;<a href="https://research.unsw.edu.au/projects/unsw-nb15-dataset" target="_blank" style="color:#00ff88;text-decoration:none;border-bottom:1px solid rgba(0,255,136,0.4);">⬇ DOWNLOAD</a><br>
                ► ANY NETWORK CSV ✅
            </div>
        </div>
        """, unsafe_allow_html=True)
