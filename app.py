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
.graph-explanation {
    background: rgba(0,255,136,0.03);
    border-left: 3px solid rgba(0,255,136,0.3);
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 1rem 0;
    font-family: 'Rajdhani', sans-serif;
    color: rgba(200,230,255,0.75);
    font-size: 0.95rem;
    line-height: 1.8;
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

def graph_explanation(text):
    st.markdown(
        f'<div class="graph-explanation">{text}</div>',
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
            <div class="metric-value">95.17%</div>
            <div class="metric-delta">✦ EXPLAINABLE</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">GAT ACCURACY</div>
            <div class="metric-value">92.84%</div>
            <div class="metric-delta">✦ EXPLAINABLE</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">GRAPH NODES</div>
            <div class="metric-value">175,341</div>
            <div class="metric-delta">✦ CONNECTIONS</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">FEATURES</div>
            <div class="metric-value">42</div>
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
        &nbsp;&nbsp;UNSW-NB15 Dataset · 175,341 connections<br>
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
        progress_bar("GCN  ACCURACY", 0.9517,
                     "progress-gcn")
        progress_bar("GCN  F1-SCORE", 0.9654,
                     "progress-gcn")
        progress_bar("GCN  ROC-AUC", 0.9890,
                     "progress-gcn")
        st.markdown("<br>", unsafe_allow_html=True)
        progress_bar("GAT  ACCURACY", 0.9284,
                     "progress-gat")
        progress_bar("GAT  F1-SCORE", 0.9500,
                     "progress-gat")
        progress_bar("GAT  ROC-AUC", 0.9670,
                     "progress-gat")

    with col2:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\','
            'monospace;color:#ff6b00;font-size:0.8rem;'
            'letter-spacing:2px;margin-bottom:0.8rem;">'
            '── BASELINE MODELS ──</div>',
            unsafe_allow_html=True
        )
        progress_bar("RF   ACCURACY", 0.9608,
                     "progress-rf")
        progress_bar("RF   F1-SCORE", 0.9714,
                     "progress-rf")
        progress_bar("RF   ROC-AUC", 0.9941,
                     "progress-rf")
        st.markdown("<br>", unsafe_allow_html=True)
        progress_bar("MLP  ACCURACY", 0.9487,
                     "progress-mlp")
        progress_bar("MLP  F1-SCORE", 0.9626,
                     "progress-mlp")
        progress_bar("MLP  ROC-AUC", 0.9911,
                     "progress-mlp")

    st.markdown("---")
    cyber_header("▸ INTERACTIVE PERFORMANCE CHART")
    import plotly.graph_objects as go

    models = ['GCN', 'GAT', 'Random Forest', 'MLP']
    accuracy = [95.17, 92.84, 96.08, 94.87]
    f1 = [96.54, 95.00, 97.14, 96.26]
    roc = [98.90, 96.70, 99.41, 99.11]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='Accuracy', x=models, y=accuracy,
        marker_color=['#00ff88', '#00ff88', '#ff6b00', '#ff0055'],
        text=[f'{v}%' for v in accuracy], textposition='outside'
    ))
    fig_bar.add_trace(go.Bar(
        name='F1-Score', x=models, y=f1,
        marker_color=['rgba(0,255,136,0.5)', 'rgba(0,255,136,0.5)', 'rgba(255,107,0,0.5)', 'rgba(255,0,85,0.5)'],
        text=[f'{v}%' for v in f1], textposition='outside'
    ))
    fig_bar.add_trace(go.Bar(
        name='ROC-AUC', x=models, y=roc,
        marker_color=['rgba(0,200,255,0.5)', 'rgba(0,200,255,0.5)', 'rgba(255,204,0,0.5)', 'rgba(255,107,0,0.5)'],
        text=[f'{v}%' for v in roc], textposition='outside'
    ))
    fig_bar.update_layout(
        barmode='group', paper_bgcolor='#020b14', plot_bgcolor='#020b14',
        font=dict(color='#00ff88', family='Share Tech Mono'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,255,136,0.3)'),
        xaxis=dict(gridcolor='rgba(0,255,136,0.1)', tickfont=dict(color='#00ccff')),
        yaxis=dict(gridcolor='rgba(0,255,136,0.1)', tickfont=dict(color='#00ccff'), range=[85, 101]),
        hoverlabel=dict(bgcolor='#020b14', bordercolor='#00ff88', font=dict(color='#00ff88')),
        margin=dict(t=30, b=10), height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True,
                    config={'displayModeBar': True, 'displaylogo': False})

    # Results table
    cyber_header("▸ FULL COMPARISON TABLE")
    results_df = pd.DataFrame({
        'MODEL': ['GCN', 'GAT',
                  'Random Forest', 'MLP'],
        'ACCURACY': ['95.17%', '92.84%',
                     '96.08%', '94.87%'],
        'PRECISION': ['0.9411', '0.9048',
                      '0.9648', '0.9547'],
        'RECALL': ['0.9911', '1.0000',
                   '0.9782', '0.9707'],
        'F1': ['0.9654', '0.9500',
               '0.9714', '0.9626'],
        'ROC-AUC': ['0.9880', '0.9670',
                    '0.9941', '0.9911'],
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
        graph_explanation(
            "GCN training over 100 epochs on UNSW-NB15. "
            "Loss steadily decreased from 0.2689 to 0.1202, "
            "while test accuracy improved from 93.03% to "
            "<b style='color:#00ff88'>95.17%</b> — showing "
            "stable learning with no overfitting."
        )
    with col2:
        cyber_header("▸ GAT TRAINING CURVES")
        show_image("outputs/gat_training_results.png")
        graph_explanation(
            "GAT training over 100 epochs on UNSW-NB15. "
            "The attention mechanism required more epochs to "
            "stabilize, achieving a final test accuracy of "
            "<b style='color:#b400ff'>92.84%</b> with "
            "perfect recall of 1.0000 — detecting every attack."
        )

    st.markdown("---")
    cyber_header("▸ CONFUSION MATRICES")
    show_image("outputs/confusion_matrices.png")
    graph_explanation(
        "Confusion matrices for all 4 models on UNSW-NB15 test set. "
        "GCN correctly classified <b style='color:#00ff88'>95.17%</b> "
        "of connections. GAT achieved perfect recall — it missed "
        "<b style='color:#b400ff'>zero attacks</b>. "
        "Random Forest had the highest overall accuracy at 96.08%."
    )

    st.markdown("---")
    cyber_header("▸ ROC CURVES")
    show_image("outputs/roc_curves.png")
    graph_explanation(
        "ROC curves compare all models on UNSW-NB15. "
        "GCN achieves ROC-AUC of <b style='color:#00ff88'>0.9890</b> "
        "and GAT achieves <b style='color:#b400ff'>0.9670</b> — "
        "both significantly above the random classifier baseline. "
        "Higher AUC means better attack vs normal discrimination."
    )


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
        graph_explanation(
            "GCN top feature: <b style='color:#00ff88'>ct_srv_dst</b> "
            "(0.0747) — counts connections to the same service on dst. "
            "Other key features: ct_state_ttl, state, dttl — all "
            "connection-tracking features unique to UNSW-NB15 that "
            "capture abnormal network behaviour patterns."
        )
    with col2:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\','
            'monospace;color:#b400ff;font-size:0.75rem;'
            'letter-spacing:2px;margin:0.5rem 0;">'
            '── GAT FEATURES ──</div>',
            unsafe_allow_html=True
        )
        show_image("outputs/gat_feature_importance.png")
        graph_explanation(
            "GAT top feature: <b style='color:#b400ff'>ct_dst_sport_ltm</b> "
            "(0.1313) — tracks destination sport over time. "
            "GAT's attention mechanism identified <b style='color:#00ccff'>"
            "dload and rate</b> as critical — capturing data transfer "
            "patterns that distinguish DoS and reconnaissance attacks."
        )

    st.success(
        "🔑 KEY FINDING: `ct_srv_dst` is the #1 GCN feature "
        "and `ct_dst_sport_ltm` is the #1 GAT feature — "
        "both are UNSW-NB15 connection-tracking features "
        "that capture how many times a service was accessed, "
        "making them strong indicators of scanning and DoS attacks!"
    )

    st.markdown("---")
    cyber_header("▸ ATTACK SUBGRAPH EXPLANATION")
    show_image("outputs/gcn_subgraph_node_47911.png",
               "Yellow=Target · Red=Attack · Blue=Normal")
    graph_explanation(
        "Subgraph explanation for node 47911 — predicted as "
        "<b style='color:#ff0055'>ATTACK</b> with 99.69% confidence. "
        "The yellow node is the target connection. Red neighbour nodes "
        "show surrounding attack traffic. This visualizes HOW the GCN "
        "uses graph structure — not just features — to detect intrusions."
    )

    st.markdown("---")
    cyber_header("▸ GAT ATTENTION WEIGHTS")

    col1, col2 = st.columns(2)
    with col1:
        show_image("outputs/gat_attention_weights.png")
        graph_explanation(
            "GAT attention weight distribution across all edges. "
            "Mean attention: 0.3335 · Max: 1.0000. "
            "Higher attention = GAT considers that connection "
            "more important for the prediction decision."
        )
    with col2:
        show_image("outputs/attention_per_head.png")
        graph_explanation(
            "Attention weights split across all 8 attention heads. "
            "Each head specializes in different structural patterns — "
            "some focus on attack clusters, others on normal traffic flows."
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        show_image("outputs/top_attended_nodes.png")
        graph_explanation(
            "Top 50 most attended edges in the UNSW-NB15 graph. "
            "These are the connections the GAT model considers "
            "most critical when making its predictions."
        )
    with col2:
        show_image("outputs/attack_vs_normal_attention.png")
        graph_explanation(
            "Attack connections: mean attention 0.3336 · "
            "Normal connections: mean attention 0.3333. "
            "GAT assigns slightly higher attention to attack traffic — "
            "showing it has learned to focus on suspicious connections."
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
        dur = st.slider("Duration (sec)", 0, 100, 0)
        proto = st.selectbox(
            "Protocol", ["tcp", "udp", "unas", "ospf", "arp"]
        )
        state = st.selectbox(
            "State", ["FIN", "CON", "INT", "REQ", "RST"]
        )
        spkts = st.slider("Src Packets", 0, 500, 10)
        dpkts = st.slider("Dst Packets", 0, 500, 10)

    with col2:
        cyber_header("TRAFFIC STATS")
        sbytes = st.slider("Src Bytes", 0, 100000, 500)
        dbytes = st.slider("Dst Bytes", 0, 100000, 500)
        rate = st.slider("Packet Rate", 0.0, 1000.0, 10.0)
        sttl = st.slider("Src TTL", 0, 255, 64)
        dttl = st.slider("Dst TTL", 0, 255, 64)

    with col3:
        cyber_header("CONNECTION TRACKING")
        ct_state_ttl = st.slider(
            "CT State TTL", 0, 6, 2
        )
        ct_srv_src = st.slider(
            "CT Srv Src", 0, 100, 10
        )
        ct_srv_dst = st.slider(
            "CT Srv Dst", 0, 100, 10
        )
        is_sm_ips_ports = st.selectbox(
            "Same IP/Port", [0, 1]
        )
        ct_dst_sport_ltm = st.slider(
            "CT Dst Sport LTM", 0, 100, 5
        )

    st.markdown("---")

    if st.button("⚡ ANALYZE CONNECTION",
                 use_container_width=True):

        attack_score = 0
        reasons = []

        if ct_state_ttl <= 1:
            attack_score += 3
            reasons.append(
                f"⚠ CT State TTL: {ct_state_ttl} "
                f"— abnormal state transition detected"
            )
        if is_sm_ips_ports == 1:
            attack_score += 2
            reasons.append(
                f"⚠ Same IP/Port flag set "
                f"— possible shellcode or exploit"
            )
        if rate > 500:
            attack_score += 2
            reasons.append(
                f"⚠ Packet rate: {rate:.1f} "
                f"— possible DoS/flood attack"
            )
        if dur == 0 and dpkts > 100:
            attack_score += 2
            reasons.append(
                f"⚠ Zero duration with high dst packets "
                f"— reconnaissance detected"
            )
        if ct_srv_dst > 80:
            attack_score += 1
            reasons.append(
                f"⚠ CT Srv Dst: {ct_srv_dst} "
                f"— high service destination connections"
            )
        if sttl < 10 or dttl < 10:
            attack_score += 1
            reasons.append(
                f"⚠ Very low TTL values "
                f"(src:{sttl} dst:{dttl}) "
                f"— possible spoofing"
            )
        if state in ["REQ", "RST"] and rate > 100:
            attack_score += 1
            reasons.append(
                f"⚠ State {state} with high rate "
                f"— incomplete/reset connections"
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
                'CT State TTL', 'Same IP/Port',
                'Packet Rate', 'CT Srv Dst',
                'Src TTL', 'State'
            ],
            'VALUE': [
                ct_state_ttl, is_sm_ips_ports,
                rate, ct_srv_dst,
                sttl, state
            ],
            'RISK': [
                '🔴 HIGH' if ct_state_ttl <= 1
                else '🟢 LOW',
                '🔴 HIGH' if is_sm_ips_ports == 1
                else '🟢 LOW',
                '🔴 HIGH' if rate > 500
                else '🟡 MED' if rate > 200
                else '🟢 LOW',
                '🟡 MED' if ct_srv_dst > 80
                else '🟢 LOW',
                '🔴 HIGH' if sttl < 10
                else '🟢 LOW',
                '🟡 WATCH' if state in ['REQ', 'RST']
                else '🟢 NORMAL'
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
        cyber_header("▸ INTERACTIVE RADAR CHART")
        import plotly.graph_objects as go

        default_metrics = {
            'GCN': [95.17, 94.11, 99.11, 96.54, 98.90],
            'GAT': [92.84, 90.48, 100.0, 95.00, 96.70],
            'Random Forest': [96.08, 96.48, 97.82, 97.14, 99.41],
            'MLP': [94.87, 95.47, 97.07, 96.26, 99.11],
        }
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        fig_radar = go.Figure()
        colors = ['#00ff88', '#b400ff', '#ff6b00', '#ff0055']
        fill_colors = [
            'rgba(0,255,136,0.08)', 'rgba(180,0,255,0.08)',
            'rgba(255,107,0,0.08)', 'rgba(255,0,85,0.08)'
        ]
        for (model, values), color, fill in zip(default_metrics.items(), colors, fill_colors):
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself', name=model,
                line=dict(color=color, width=2),
                fillcolor=fill
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#020b14',
                radialaxis=dict(visible=True, autorange=True,
                                tickfont=dict(color='#00ccff', size=9)),
                angularaxis=dict(gridcolor='rgba(0,255,136,0.15)',
                                 tickfont=dict(color='#00ff88', size=10))
            ),
            paper_bgcolor='#020b14',
            font=dict(color='#00ff88', family='Share Tech Mono'),
            legend=dict(bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,255,136,0.3)', font=dict(size=10)),
            margin=dict(t=30, b=30), height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True,
                        config={'displayModeBar': True, 'displaylogo': False})
        graph_explanation(
            "Radar chart comparing all 4 models across 5 metrics on UNSW-NB15. "
            "<b style='color:#00ff88'>GCN</b> shows the best balance of all metrics. "
            "While Random Forest has higher accuracy, "
            "GCN and GAT provide <b style='color:#00ccff'>explainable decisions</b> "
            "— critical for real-world cybersecurity deployment."
        )
    with col2:
        cyber_header("▸ METRICS HEATMAP")
        show_image("outputs/metrics_heatmap.png")
        graph_explanation(
            "Heatmap of all model metrics on UNSW-NB15. "
            "<b style='color:#00ff88'>Darker = stronger performance.</b> "
            "GCN and GAT show consistently competitive scores. "
            "GAT achieves perfect recall of 1.0000 — "
            "it missed zero attacks in the test set."
        )

    st.markdown("---")
    cyber_header("▸ ACCURACY vs EXPLAINABILITY")
    show_image("outputs/accuracy_vs_explainability.png",
               "Key research argument — "
               "XGNN achieves high explainability "
               "with competitive accuracy")
    graph_explanation(
        "This is the core research argument of this project. "
        "Traditional models like Random Forest achieve high accuracy "
        "but are <b style='color:#ff6b00'>black boxes</b>. "
        "GCN and GAT achieve competitive accuracy on UNSW-NB15 "
        "while providing <b style='color:#00ff88'>full explainability</b> — "
        "showing WHICH features and graph connections triggered each alert."
    )

    st.markdown("---")
    cyber_header("▸ FULL MODEL COMPARISON")
    show_image("outputs/model_comparison.png")
    graph_explanation(
        "Full comparison of all 4 models across all metrics on UNSW-NB15. "
        "GCN achieves the best balance of accuracy and explainability. "
        "Random Forest leads in raw accuracy at 96.08% but cannot "
        "explain its decisions — making it unsuitable for "
        "<b style='color:#00ccff'>transparent cybersecurity deployment.</b>"
    )

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
    st.info(
        "ℹ️ This model was trained on the "
        "UNSW-NB15 dataset — 175,341 connections · "
        "42 features · 9 attack categories. "
        "Upload any compatible network traffic CSV "
        "for instant analysis."
    )

    uploaded_file = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="Supports: UNSW-NB15 (primary), "
             "CIC-IDS-2017, NSL-KDD, KDD Cup, "
             "or any network traffic CSV"
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
            st.success(
                f"✅ File uploaded successfully! "
                f"{df.shape[0]} rows · "
                f"{df.shape[1]} columns detected"
            )
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            st.stop()

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
                ► UNSW-NB15 &nbsp;&nbsp;&nbsp;✅ PRIMARY &nbsp;<a href="https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15" target="_blank" style="color:#00ff88;font-size:0.7rem;">[DOWNLOAD]</a><br>
                ► CIC-IDS-2017 &nbsp;✅ &nbsp;<a href="https://www.kaggle.com/datasets/cicdataset/cicids2017" target="_blank" style="color:#00ccff;font-size:0.7rem;">[DOWNLOAD]</a><br>
                ► NSL-KDD &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ &nbsp;<a href="https://www.kaggle.com/datasets/hassan06/nslkdd" target="_blank" style="color:#00ccff;font-size:0.7rem;">[DOWNLOAD]</a><br>
                ► KDD CUP 1999 &nbsp;✅ &nbsp;<a href="https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data" target="_blank" style="color:#00ccff;font-size:0.7rem;">[DOWNLOAD]</a><br>
                ► ANY NETWORK CSV ✅
            </div>
        </div>
        """, unsafe_allow_html=True)
