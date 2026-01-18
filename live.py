"""
L2 INDUSTRY-LEVEL MONITORING DASHBOARD - WiFi Stability Prediction
3×2 + 1×2 Grid Architecture with KPI-First Design
Pure monitoring and analytics layer consuming L1 prediction streams
Industry-grade observability for predictive WiFi stability system
"""

import socket
import json
import threading
import time
from datetime import datetime, timedelta
from collections import deque, Counter
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ======================================================
# CONFIGURATION
# ======================================================
L1_HOST = "127.0.0.1"
L1_PORT = 2022
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8052  # Changed to avoid cache
MAX_HISTORY = 500
UPDATE_MS = 1000

# ======================================================
# THEME & COLORS - Industry Standards
# ======================================================
COLORS = {
    "bg": "#050709",
    "bg_grad": "radial-gradient(ellipse at 50% -20%, rgba(30, 58, 138, 0.15) 0%, transparent 50%), linear-gradient(180deg, #0a0d14 0%, #050709 100%)",
    "surface": "#0a0d14",
    "card": "rgba(15, 23, 42, 0.45)",
    "primary": "#3b82f6",
    # Industry-standard state colors
    "stable": "#2ca02c",      # Healthy
    "degrading": "#ffbf00",   # Warning
    "unstable": "#d62728",    # Critical
    # Signal colors
    "probability": "#1f77b4", # Risk signal (Industry blue)
    "reliability": "#2ca02c", # Trust (same as stable)
    "diagnostics": "#9467bd", # Analysis (Neutral diagnostic purple)
    "insights": "#17becf",    # Forecasting (Insight cyan)
    # Legacy compatibility
    "success": "#2ca02c",
    "warning": "#ffbf00",
    "error": "#d62728",
    "text": "#f1f5f9",
    "text_dim": "#cbd5e1",
    "text_muted": "#94a3b8",
    "border": "rgba(59, 130, 246, 0.12)"
}

KPI_CARD = {
    "background": f"linear-gradient(135deg, {COLORS['card']} 0%, rgba(20, 30, 60, 0.3) 100%)",
    "backdropFilter": "blur(24px) saturate(1.3)",
    "padding": "28px 20px",
    "borderRadius": "16px",
    "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.12), 0 0 0 1px rgba(255, 255, 255, 0.05)",
    "border": f"1px solid {COLORS['border']}",
    "textAlign": "center",
    "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    "position": "relative",
    "overflow": "hidden"
}

CHART_CARD = {
    "background": f"linear-gradient(135deg, {COLORS['card']} 0%, rgba(20, 30, 60, 0.25) 100%)",
    "backdropFilter": "blur(20px) saturate(1.2)",
    "padding": "24px",
    "borderRadius": "16px",
    "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.05)",
    "border": f"1px solid {COLORS['border']}",
    "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    "minHeight": "340px"
}

# ======================================================
# THREAD-SAFE DATA STORAGE
# ======================================================
data_lock = threading.Lock()
connection_status = False
prediction_history = deque(maxlen=MAX_HISTORY)
temporal_metrics = {
    "states": deque(maxlen=MAX_HISTORY),
    "timestamps": deque(maxlen=MAX_HISTORY),
    "probabilities": deque(maxlen=MAX_HISTORY),
    "confidences": deque(maxlen=MAX_HISTORY),
    "transitions": 0,
    "flip_rate_window": deque(maxlen=60),
    "prob_variance_window": deque(maxlen=60),
    "state_durations": Counter(),
    "event_log": deque(maxlen=100),
    "lead_time_events": deque(maxlen=50)
}

def _reset_temporal_metrics():
    """Clear all metrics when L1 connects (switch from sample to live data)."""
    global connection_status
    with data_lock:
        prediction_history.clear()
        for k in temporal_metrics:
            if hasattr(temporal_metrics[k], 'clear'):
                temporal_metrics[k].clear()
            elif isinstance(temporal_metrics[k], Counter):
                temporal_metrics[k].clear()
        temporal_metrics["transitions"] = 0

def process_prediction(pred, ts=None):
    """Process one L1-style prediction (state, probability, confidence) into temporal_metrics."""
    t = ts if ts is not None else time.time()
    with data_lock:
        prediction_history.append(pred)
        temporal_metrics["states"].append(pred.get("state", "UNKNOWN"))
        temporal_metrics["timestamps"].append(t)
        temporal_metrics["probabilities"].append(pred.get("probability", 0))
        temporal_metrics["confidences"].append(pred.get("confidence", 0))
        # Transitions and flip rate
        if len(temporal_metrics["states"]) > 1:
            if temporal_metrics["states"][-1] != temporal_metrics["states"][-2]:
                temporal_metrics["transitions"] += 1
                temporal_metrics["flip_rate_window"].append(1)
            else:
                temporal_metrics["flip_rate_window"].append(0)
        if len(temporal_metrics["probabilities"]) > 1:
            temporal_metrics["prob_variance_window"].append(pred.get("probability", 0))
        # UNSTABLE events and lead time
        current_time = temporal_metrics["timestamps"][-1]
        if pred.get("state") == "UNSTABLE":
            temporal_metrics["event_log"].append({
                "timestamp": current_time,
                "state": "UNSTABLE",
                "prob": pred.get("probability", 0)
            })
            states_list = list(temporal_metrics["states"])
            timestamps_list = list(temporal_metrics["timestamps"])
            if len(states_list) > 1 and len(timestamps_list) > 1:
                for i in range(len(states_list) - 2, max(-1, len(states_list) - 20), -1):
                    if states_list[i] == "DEGRADING" and i < len(timestamps_list):
                        warning_time = timestamps_list[i]
                        lead_time = current_time - warning_time
                        if lead_time > 0 and lead_time < 300:
                            temporal_metrics["lead_time_events"].append(lead_time)
                        break

# ======================================================
# L1 CONNECTION HANDLER
# ======================================================
def receive_l1_predictions():
    """Receive live predictions from L1 node - only works with real data."""
    global connection_status
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.connect((L1_HOST, L1_PORT))
            connection_status = True
            _reset_temporal_metrics()
            print(f"[L2] Connected to L1 at {L1_HOST}:{L1_PORT}")
            while True:
                data = sock.recv(1024)
                if not data:
                    break
                try:
                    pred = json.loads(data.decode('utf-8').strip())
                    process_prediction(pred)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            connection_status = False
            print(f"[L2] Connection error: {e}. Waiting to reconnect...")
            time.sleep(2)
        finally:
            try:
                sock.close()
            except Exception:
                pass

# ======================================================
# METRIC DERIVATIONS
# ======================================================
def get_current_state():
    """Get the current WiFi state (most recent)"""
    with data_lock:
        if temporal_metrics["states"]:
            return temporal_metrics["states"][-1]
    return "UNKNOWN"

def get_current_probability():
    """Get the current instability probability (most recent)"""
    with data_lock:
        if temporal_metrics["probabilities"]:
            return temporal_metrics["probabilities"][-1]
    return 0.0

def get_current_confidence():
    """Get the current prediction confidence (most recent)"""
    with data_lock:
        if temporal_metrics["confidences"]:
            return temporal_metrics["confidences"][-1]
    return 0.0

def get_alert_level():
    with data_lock:
        recent = list(temporal_metrics["probabilities"])[-60:] if len(temporal_metrics["probabilities"]) > 0 else []
        conf_recent = list(temporal_metrics["confidences"])[-60:] if len(temporal_metrics["confidences"]) > 0 else []
        states_recent = list(temporal_metrics["states"])[-60:] if len(temporal_metrics["states"]) > 0 else []
        
        if not recent:
            return "INFO"
        
        avg_prob = np.mean(recent)
        avg_conf = np.mean(conf_recent)
        unstable_pct = (states_recent.count("UNSTABLE") / len(states_recent)) if states_recent else 0
        
        if unstable_pct > 0.5 or (avg_prob > 0.6 and avg_conf > 0.75):
            return "CRITICAL"
        elif avg_prob > 0.4 and unstable_pct > 0.2:
            return "WARNING"
        return "INFO"

def get_service_health_index():
    with data_lock:
        recent = list(temporal_metrics["probabilities"])[-300:] if len(temporal_metrics["probabilities"]) > 0 else []
        conf_recent = list(temporal_metrics["confidences"])[-300:] if len(temporal_metrics["confidences"]) > 0 else []
        states_recent = list(temporal_metrics["states"])[-300:] if len(temporal_metrics["states"]) > 0 else []
        
        if not recent:
            return 50.0
        
        avg_prob = np.mean(recent)
        avg_conf = np.mean(conf_recent)
        stability = 1.0 - (states_recent.count("UNSTABLE") / len(states_recent)) if states_recent else 1.0
        
        shi = (1.0 - avg_prob) * avg_conf * stability * 100
        return max(0, min(100, shi))

def get_prediction_reliability_index():
    with data_lock:
        recent_conf = list(temporal_metrics["confidences"])[-60:] if len(temporal_metrics["confidences"]) > 0 else []
        flip_window = list(temporal_metrics["flip_rate_window"])[-60:] if len(temporal_metrics["flip_rate_window"]) > 0 else []
        
        if not recent_conf:
            return 0.5
        
        avg_conf = np.mean(recent_conf)
        flip_rate = (sum(flip_window) / len(flip_window)) if flip_window else 0
        
        pri = avg_conf * (1.0 - min(flip_rate / 5.0, 1.0))
        return max(0, min(1, pri))

def get_volatility():
    with data_lock:
        flip_window = list(temporal_metrics["flip_rate_window"])[-60:] if len(temporal_metrics["flip_rate_window"]) > 0 else []
        prob_window = list(temporal_metrics["prob_variance_window"])[-60:] if len(temporal_metrics["prob_variance_window"]) > 0 else []
        
        flip_rate = (sum(flip_window) / len(flip_window)) if flip_window else 0
        prob_var = np.var(prob_window) if len(prob_window) > 1 else 0
        
        volatility = (flip_rate * 0.5) + (prob_var * 50)
        return volatility

def get_mean_predictive_lead_time():
    """KPI 3: Average time between early warning (DEGRADING) and instability onset (UNSTABLE)"""
    with data_lock:
        lead_times = list(temporal_metrics["lead_time_events"])
        if not lead_times:
            return 0.0
        return np.mean(lead_times)

def get_instability_exposure_ratio():
    """KPI 4: Percentage of time WiFi is in UNSTABLE state"""
    with data_lock:
        states = list(temporal_metrics["states"])[-300:] if len(temporal_metrics["states"]) > 0 else []
        if not states:
            return 0.0
        unstable_count = states.count("UNSTABLE")
        return (unstable_count / len(states)) * 100.0

def get_instability_event_count():
    """Optional: Count of UNSTABLE events in recent window (5 minutes)"""
    with data_lock:
        event_log = list(temporal_metrics["event_log"])
        if not event_log:
            return 0
        # Count events in last 5 minutes
        recent_events = [e for e in event_log if time.time() - e["timestamp"] < 300]
        return len(recent_events)

def get_average_recovery_time():
    """Optional: Average time from UNSTABLE to STABLE"""
    with data_lock:
        states = list(temporal_metrics["states"])[-200:]
        timestamps = list(temporal_metrics["timestamps"])[-200:]
    
    if len(states) < 2:
        return 0.0
    
    recovery_times = []
    unstable_start = None
    
    for i in range(len(states)):
        if states[i] == "UNSTABLE" and (i == 0 or states[i-1] != "UNSTABLE"):
            unstable_start = timestamps[i] if i < len(timestamps) else time.time()
        elif states[i] == "STABLE" and unstable_start is not None and (i == 0 or states[i-1] != "STABLE"):
            if i < len(timestamps):
                recovery_time = timestamps[i] - unstable_start
                if recovery_time > 0 and recovery_time < 300:
                    recovery_times.append(recovery_time)
            unstable_start = None
    
    return np.mean(recovery_times) if recovery_times else 0.0

def get_avg_confidence_during_unstable():
    """Optional: Average confidence during UNSTABLE states"""
    with data_lock:
        states = list(temporal_metrics["states"])[-200:]
        confidences = list(temporal_metrics["confidences"])[-200:]
    
    if not states or not confidences:
        return 0.0
    
    unstable_confs = [confidences[i] for i in range(min(len(states), len(confidences))) if states[i] == "UNSTABLE"]
    return np.mean(unstable_confs) if unstable_confs else 0.0

# ======================================================
# PLOTLY FIGURES
# ======================================================
def create_instability_probability_trend():
    """Chart 1: Instability Probability Trend - Ribbon/Stream Chart with gradient bands"""
    with data_lock:
        probs = list(temporal_metrics["probabilities"])[-100:]
    
    if not probs:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate bands: min, max, mean for ribbon effect
    ma_window = min(20, len(probs) // 3)
    if ma_window > 1:
        ma = pd.Series(probs).rolling(window=ma_window, min_periods=1).mean()
        std = pd.Series(probs).rolling(window=ma_window, min_periods=1).std().fillna(0)
        upper = ma + std
        lower = ma - std
    else:
        ma = probs
        upper = probs
        lower = probs
    
    x_vals = list(range(len(probs)))
    fig = go.Figure()
    # Ribbon: upper and lower bounds with gradient
    fig.add_trace(go.Scatter(
        x=x_vals, y=list(upper), mode='lines', name='Upper Bound',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=list(lower), mode='lines', name='Lower Bound',
        line=dict(width=0), fill='tonexty', fillcolor=f'rgba(31, 119, 180, 0.25)',
        showlegend=False, hoverinfo='skip'
    ))
    # Main trend line with gradient color
    fig.add_trace(go.Scatter(
        x=x_vals, y=probs, mode='lines', name='Probability',
        line=dict(color=COLORS['probability'], width=3, shape='spline'),
        marker=dict(size=5, color=COLORS['probability']),
        hovertemplate='Prob: %{y:.3f}<extra></extra>'
    ))
    # Rolling mean overlay
    fig.add_trace(go.Scatter(
        x=x_vals, y=list(ma), mode='lines', name='Trend',
        line=dict(color=COLORS['unstable'], width=2.5, dash='dot'),
        hovertemplate='Trend: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Instability Probability Trend",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Sample"),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Probability")
    )
    return fig

def create_confidence_trend():
    """Chart 2: Prediction Confidence Trend - Area Chart with Confidence Bands"""
    with data_lock:
        confs = list(temporal_metrics["confidences"])[-100:]
    
    if not confs:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    x_vals = list(range(len(confs)))
    mean_conf = np.mean(confs)
    std_conf = np.std(confs) if len(confs) > 1 else 0.05
    
    fig = go.Figure()
    # Confidence bands (±1σ, ±2σ)
    fig.add_trace(go.Scatter(
        x=x_vals, y=[mean_conf + 2*std_conf] * len(confs), mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=[max(0, mean_conf - 2*std_conf)] * len(confs), mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor=f'rgba(44, 160, 44, 0.08)',
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=[mean_conf + std_conf] * len(confs), mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=[max(0, mean_conf - std_conf)] * len(confs), mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor=f'rgba(44, 160, 44, 0.15)',
        showlegend=False, hoverinfo='skip'
    ))
    # Main confidence line
    fig.add_trace(go.Scatter(
        x=x_vals, y=confs, mode='lines', name='Confidence',
        line=dict(color=COLORS['reliability'], width=3.5, shape='spline'),
        fill='tozeroy', fillcolor=f'rgba(44, 160, 44, 0.3)',
        hovertemplate='Confidence: %{y:.3f}<extra></extra>'
    ))
    # Mean line
    fig.add_hline(y=mean_conf, line_dash="dot", line_color=COLORS['text_muted'], 
                  line_width=1, annotation_text=f"Mean: {mean_conf:.2f}", 
                  annotation_position="right", annotation_font_size=9)
    
    fig.update_layout(
        title="Prediction Confidence Trend",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Sample"),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Confidence")
    )
    return fig

def create_pri_chart():
    """Chart 3: Prediction Reliability Index (PRI) - Gauge/Indicator Style"""
    with data_lock:
        recent_conf = list(temporal_metrics["confidences"])[-60:] if len(temporal_metrics["confidences"]) > 0 else []
        flip_window = list(temporal_metrics["flip_rate_window"])[-60:] if len(temporal_metrics["flip_rate_window"]) > 0 else []
    
    if not recent_conf:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate current PRI
    avg_conf = np.mean(recent_conf)
    flip_rate = sum(flip_window) / len(flip_window) if flip_window else 0
    current_pri = avg_conf * (1.0 - min(flip_rate / 5.0, 1.0))
    current_pri = max(0, min(1, current_pri))
    
    # Create gauge chart (semicircle)
    fig = go.Figure()
    # Background arc (full range)
    theta_bg = np.linspace(0, 180, 100)
    fig.add_trace(go.Scatterpolar(
        r=[1]*100, theta=theta_bg, mode='lines',
        line=dict(color=COLORS['text_muted'], width=20),
        showlegend=False, hoverinfo='skip'
    ))
    # Colored zones
    zones = [
        (0, 0.4, COLORS['unstable']),
        (0.4, 0.7, COLORS['degrading']),
        (0.7, 1.0, COLORS['stable'])
    ]
    for start, end, color in zones:
        theta_zone = np.linspace(start*180, end*180, 30)
        fig.add_trace(go.Scatterpolar(
            r=[0.95]*30, theta=theta_zone, mode='lines',
            line=dict(color=color, width=18),
            showlegend=False, hoverinfo='skip'
        ))
    # Current PRI indicator
    pri_angle = current_pri * 180
    fig.add_trace(go.Scatterpolar(
        r=[0.7, 1.0], theta=[pri_angle, pri_angle], mode='lines+markers',
        line=dict(color=COLORS['reliability'], width=4),
        marker=dict(size=12, color=COLORS['reliability']),
        showlegend=False,
        hovertemplate=f'PRI: {current_pri:.3f}<extra></extra>'
    ))
    # Center value text
    fig.add_annotation(
        text=f"<b>{current_pri:.2f}</b>",
        xref="paper", yref="paper", x=0.5, y=0.35,
        showarrow=False, font=dict(size=32, color=COLORS['reliability'])
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180],
                ticktext=['0.0', '0.25', '0.5', '0.75', '1.0'],
                direction='counterclockwise',
                rotation=90,
                showgrid=True, gridcolor='rgba(255,255,255,0.1)'
            )
        ),
        title="Prediction Reliability Index",
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=40, r=40, t=50, b=40),
        height=300,
        showlegend=False
    )
    return fig

def create_state_timeline():
    """Chart 4: WiFi State Transition Timeline - Step Plot with State Zones"""
    with data_lock:
        states = list(temporal_metrics["states"])[-100:]
        timestamps = list(temporal_metrics["timestamps"])[-100:]
    
    if not states:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Convert to numeric: STABLE=0, DEGRADING=1, UNSTABLE=2
    state_map = {"STABLE": 0, "DEGRADING": 1, "UNSTABLE": 2}
    numeric_states = [state_map.get(s, 0) for s in states]
    
    x_vals = list(range(len(numeric_states)))
    colors_list = []
    for state in states:
        if state == "STABLE":
            colors_list.append(COLORS['stable'])
        elif state == "DEGRADING":
            colors_list.append(COLORS['degrading'])
        else:
            colors_list.append(COLORS['unstable'])
    
    fig = go.Figure()
    # State zone backgrounds
    fig.add_hrect(y0=1.5, y1=2.2, fillcolor=COLORS['unstable'], opacity=0.12, layer="below", line_width=0)
    fig.add_hrect(y0=0.5, y1=1.5, fillcolor=COLORS['degrading'], opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=-0.2, y1=0.5, fillcolor=COLORS['stable'], opacity=0.1, layer="below", line_width=0)
    # Step plot with distinct markers
    fig.add_trace(go.Scatter(
        x=x_vals, y=numeric_states,
        mode='lines+markers',
        line=dict(shape='hv', color=COLORS['primary'], width=3),
        marker=dict(color=colors_list, size=8, symbol=['circle', 'square', 'diamond'][:len(colors_list)], 
                   line=dict(width=1.5, color='white')),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.15)',
        hovertemplate='State: %{text}<extra></extra>',
        text=[states[i] for i in range(len(numeric_states))]
    ))
    
    fig.update_layout(
        title="WiFi State Transition Timeline",
        yaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=['STABLE', 'DEGRADING', 'UNSTABLE'],
            range=[-0.2, 2.2],
            showgrid=True, gridcolor='rgba(255,255,255,0.05)'
        ),
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        showlegend=False,
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Sample")
    )
    return fig

def create_instability_exposure_chart():
    """Chart 5: Instability Exposure Over Time (stacked area)"""
    with data_lock:
        states = list(temporal_metrics["states"])[-100:]
        timestamps = list(temporal_metrics["timestamps"])[-100:]
    
    if not states:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create time windows (e.g., 10-point windows) for aggregation
    window_size = max(5, len(states) // 20)  # Adaptive window size
    windows = []
    stable_counts = []
    degrading_counts = []
    unstable_counts = []
    
    for i in range(0, len(states), window_size):
        window_states = states[i:i+window_size]
        stable_counts.append(window_states.count("STABLE"))
        degrading_counts.append(window_states.count("DEGRADING"))
        unstable_counts.append(window_states.count("UNSTABLE"))
        windows.append(i)
    
    if not windows:
        windows = [0]
        stable_counts = [0]
        degrading_counts = [0]
        unstable_counts = [0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=windows, y=stable_counts, mode='lines', name='STABLE',
        stackgroup='one', fillcolor=COLORS['stable'],
        line=dict(width=0)
    ))
    fig.add_trace(go.Scatter(
        x=windows, y=degrading_counts, mode='lines', name='DEGRADING',
        stackgroup='one', fillcolor=COLORS['degrading'],
        line=dict(width=0)
    ))
    fig.add_trace(go.Scatter(
        x=windows, y=unstable_counts, mode='lines', name='UNSTABLE',
        stackgroup='one', fillcolor=COLORS['unstable'],
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title="Instability Exposure Over Time",
        xaxis_title="Time Window",
        yaxis_title="State Count",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_volatility_indicator():
    """Chart 6: Volatility Indicator - Simplified Average with Threshold"""
    with data_lock:
        flip_window = list(temporal_metrics["flip_rate_window"])[-60:] if len(temporal_metrics["flip_rate_window"]) > 0 else []
        prob_window = list(temporal_metrics["prob_variance_window"])[-60:] if len(temporal_metrics["prob_variance_window"]) > 0 else []
    
    if not flip_window:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate average volatility
    flip_rate = sum(flip_window) / len(flip_window) if flip_window else 0
    prob_var = np.var(prob_window) if len(prob_window) > 1 else 0
    avg_volatility = (flip_rate * 0.5) + (prob_var * 50)
    
    # Simple gauge-style visualization
    fig = go.Figure()
    # Background bar
    fig.add_trace(go.Bar(
        x=[1], y=[avg_volatility], orientation='v',
        marker=dict(
            color=COLORS['stable'] if avg_volatility < 0.2 else COLORS['degrading'] if avg_volatility < 0.4 else COLORS['unstable'],
            line=dict(width=2, color=COLORS['bg'])
        ),
        width=0.6,
        showlegend=False,
        hovertemplate=f'Avg Volatility: {avg_volatility:.3f}<extra></extra>',
        text=[f"{avg_volatility:.3f}"],
        textposition='outside'
    ))
    # Threshold reference line
    fig.add_hline(y=0.3, line_dash="dot", line_color=COLORS['degrading'], 
                  line_width=2, annotation_text="Threshold (0.3)", 
                  annotation_position="right", annotation_font_size=10)
    
    fig.update_layout(
        title="Volatility Indicator",
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, range=[0.5, 1.5]),
        yaxis=dict(range=[0, max(0.5, avg_volatility * 1.2)], showgrid=True, 
                   gridcolor='rgba(255,255,255,0.05)', title="Volatility")
    )
    return fig

def create_state_distribution():
    """Analytical Panel 1: State Distribution Summary (donut chart)"""
    with data_lock:
        states = list(temporal_metrics["states"])[-300:]
    
    if not states:
        states = ["STABLE", "DEGRADING", "UNSTABLE"]
    
    state_counts = Counter(states)
    labels = ["STABLE", "DEGRADING", "UNSTABLE"]
    values = [state_counts.get(label, 0) for label in labels]
    colors_pie = [COLORS['stable'], COLORS['degrading'], COLORS['unstable']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors_pie, line=dict(color=COLORS['bg'], width=2)),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="State Distribution Summary",
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=True
    )
    return fig

def create_lead_time_analysis():
    """Analytical Panel 2: Predictive Lead-Time Analysis - Histogram with Box Plot Overlay"""
    with data_lock:
        lead_times = list(temporal_metrics["lead_time_events"])
    
    if not lead_times:
        fig = go.Figure()
        fig.add_annotation(
            text="No lead time data yet<br>Waiting for DEGRADING → UNSTABLE transitions",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=COLORS['text_muted'])
        )
        fig.update_layout(
            title="Predictive Lead-Time Analysis",
            template='plotly_dark',
            paper_bgcolor=COLORS['bg'],
            font=dict(color=COLORS['text_dim'], family='Inter'),
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        return fig
    
    # Create histogram with gradient colors
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=lead_times,
        nbinsx=min(20, len(lead_times)),
        marker=dict(
            color=lead_times,
            colorscale=[[0, COLORS['insights']], [0.5, COLORS['reliability']], [1, COLORS['stable']]],
            line=dict(width=1.5, color=COLORS['bg']),
            showscale=False
        ),
        name='Lead Time Distribution',
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add statistical lines
    mean_lt = np.mean(lead_times)
    median_lt = np.median(lead_times)
    q25 = np.percentile(lead_times, 25)
    q75 = np.percentile(lead_times, 75)
    
    # Box plot overlay (simplified)
    fig.add_vline(x=q25, line_dash="dot", line_color=COLORS['text_muted'], line_width=1, 
                  annotation_text="Q1", annotation_position="top", annotation_font_size=9)
    fig.add_vline(x=median_lt, line_dash="dash", line_color=COLORS['reliability'], line_width=2,
                  annotation_text=f"Median: {median_lt:.1f}s", annotation_position="top", annotation_font_size=10)
    fig.add_vline(x=q75, line_dash="dot", line_color=COLORS['text_muted'], line_width=1,
                  annotation_text="Q3", annotation_position="top", annotation_font_size=9)
    fig.add_vline(x=mean_lt, line_dash="dot", line_color=COLORS['insights'], line_width=2,
                  annotation_text=f"Mean: {mean_lt:.1f}s", annotation_position="bottom", annotation_font_size=10)
    
    fig.update_layout(
        title="Predictive Lead-Time Analysis",
        xaxis_title="Lead Time (seconds)",
        yaxis_title="Frequency",
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color=COLORS['text_dim'], family='Inter'),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    return fig

# ======================================================
# DASH APP
# ======================================================
app = dash.Dash(__name__)
app.title = "L2 Monitoring Console"

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("WiFi Stability – L2 Monitoring", style={'fontSize': '38px', 'fontWeight': '800', 'color': COLORS['text'], 'margin': '0 0 6px 0', 'letterSpacing': '-0.5px'}),
            html.P("Industry-Grade Operational Intelligence Console", style={'fontSize': '13px', 'color': COLORS['text_dim'], 'margin': '0', 'letterSpacing': '0.3px'})
        ], style={'flex': '1'}),
        html.Div(id='conn-badge', style={'minWidth': '200px', 'textAlign': 'right', 'fontSize': '13px', 'fontWeight': '500'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'gap': '32px', 'padding': '40px 48px', 'borderBottom': f'1px solid {COLORS["border"]}'}),
    
    # Main Content
    html.Div([
        # ===== ALL KPIs ON TOP - POLISHED GRID =====
        html.Div([
            # Row 1: Current State Metrics (Real-time)
            html.Div([
                html.Div([
                    html.Div('CURRENT STATE', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '12px', 'letterSpacing': '1px'}),
                    html.Div(id='kpi0-state', style={'fontSize': '40px', 'fontWeight': '800', 'lineHeight': '1.1'})
                ], style={**KPI_CARD}),
                
                html.Div([
                    html.Div('CURRENT INSTABILITY PROB', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '12px', 'letterSpacing': '1px'}),
                    html.Div(id='kpi0-prob', style={'fontSize': '40px', 'fontWeight': '800', 'lineHeight': '1.1'})
                ], style={**KPI_CARD}),
                
                html.Div([
                    html.Div('CURRENT CONFIDENCE', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '12px', 'letterSpacing': '1px'}),
                    html.Div(id='kpi0-conf', style={'fontSize': '40px', 'fontWeight': '800', 'lineHeight': '1.1'})
                ], style={**KPI_CARD}),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '16px', 'marginBottom': '16px'}),
            
            # Row 2: Aggregated Health Metrics
            html.Div([
                html.Div([
                    html.Div('SERVICE HEALTH INDEX', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '12px', 'letterSpacing': '1px'}),
                    html.Div(id='kpi1-shi', style={'fontSize': '40px', 'fontWeight': '800', 'lineHeight': '1.1'})
                ], style={**KPI_CARD}),
                
                html.Div([
                    html.Div('PRED. RELIABILITY INDEX', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '12px', 'letterSpacing': '1px'}),
                    html.Div(id='kpi2-pri', style={'fontSize': '40px', 'fontWeight': '800', 'lineHeight': '1.1'})
                ], style={**KPI_CARD}),
                
                html.Div([
                    html.Div('INSTABILITY EXPOSURE', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '12px', 'letterSpacing': '1px'}),
                    html.Div(id='kpi4-exposure', style={'fontSize': '40px', 'fontWeight': '800', 'lineHeight': '1.1'})
                ], style={**KPI_CARD}),
                
                html.Div([
                    html.Div('MEAN PREDICTIVE LEAD TIME', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '12px', 'letterSpacing': '1px'}),
                    html.Div(id='kpi3-leadtime', style={'fontSize': '40px', 'fontWeight': '800', 'lineHeight': '1.1'})
                ], style={**KPI_CARD}),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '16px', 'marginBottom': '32px'}),
        ], style={'marginBottom': '40px'}),
        
        # ===== MAIN VISUALIZATION AREA - 3×2 GRID =====
        
        # Row 1: Risk, Trust, and Readiness
        html.Div([
            html.Div([dcc.Graph(id='chart-prob-trend', config={'displayModeBar': False})], style=CHART_CARD),
            html.Div([dcc.Graph(id='chart-conf-trend', config={'displayModeBar': False})], style=CHART_CARD),
            html.Div([dcc.Graph(id='chart-pri', config={'displayModeBar': False})], style=CHART_CARD),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '16px', 'marginBottom': '16px'}),
        
        # Row 2: Temporal Stability & Behavior
        html.Div([
            html.Div([dcc.Graph(id='chart-timeline', config={'displayModeBar': False})], style=CHART_CARD),
            html.Div([dcc.Graph(id='chart-exposure', config={'displayModeBar': False})], style=CHART_CARD),
            html.Div([dcc.Graph(id='chart-volatility', config={'displayModeBar': False})], style=CHART_CARD),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '16px', 'marginBottom': '16px'}),
        
        # ===== OPTIONAL SUMMARY INDICATORS =====
        html.Div([
            html.Div([
                html.Div('INSTABILITY EVENTS (5min)', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '8px'}),
                html.Div(id='summary-events', style={'fontSize': '32px', 'fontWeight': '800', 'color': COLORS['unstable']})
            ], style={**KPI_CARD, 'padding': '20px', 'minHeight': '120px'}),
            html.Div([
                html.Div('AVG RECOVERY TIME', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '8px'}),
                html.Div(id='summary-recovery', style={'fontSize': '32px', 'fontWeight': '800', 'color': COLORS['insights']})
            ], style={**KPI_CARD, 'padding': '20px', 'minHeight': '120px'}),
            html.Div([
                html.Div('AVG CONF (UNSTABLE)', style={'fontSize': '8px', 'color': COLORS['text_muted'], 'fontWeight': '700', 'textTransform': 'uppercase', 'marginBottom': '8px'}),
                html.Div(id='summary-conf-unstable', style={'fontSize': '32px', 'fontWeight': '800', 'color': COLORS['diagnostics']})
            ], style={**KPI_CARD, 'padding': '20px', 'minHeight': '120px'}),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '16px', 'marginBottom': '32px'}),
        
        # ===== ANALYTICAL SECTION - 1×2 GRID =====
        html.Div([
            html.Div([dcc.Graph(id='chart-distribution', config={'displayModeBar': False})], style=CHART_CARD),
            html.Div([dcc.Graph(id='chart-leadtime', config={'displayModeBar': False})], style=CHART_CARD),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '16px', 'marginBottom': '40px'}),
        
    ], style={'padding': '40px 48px 60px 48px'}),
    
    dcc.Interval(id='tick', interval=UPDATE_MS, n_intervals=0)
], style={'background': COLORS['bg_grad'], 'minHeight': '100vh', 'color': COLORS['text'], 'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'})

# ======================================================
# CALLBACKS
# ======================================================
@app.callback(Output('conn-badge', 'children'), Input('tick', 'n_intervals'))
def update_conn(n):
    if connection_status:
        status, color = "🟢 Connected to L1", COLORS['stable']
    else:
        status, color = "🔴 Disconnected - Waiting for L1", COLORS['unstable']
    return html.Div(status, style={'color': color, 'fontWeight': '600'})

# KPI Callbacks - Current State (Real-time)
@app.callback(
    [Output('kpi0-state', 'children'), Output('kpi0-state', 'style')],
    Input('tick', 'n_intervals')
)
def update_kpi0_state(n):
    state = get_current_state()
    color_map = {"STABLE": COLORS['stable'], "DEGRADING": COLORS['degrading'], "UNSTABLE": COLORS['unstable']}
    color = color_map.get(state, COLORS['text_dim'])
    return state, {**KPI_CARD, 'color': color}

@app.callback(
    [Output('kpi0-prob', 'children'), Output('kpi0-prob', 'style')],
    Input('tick', 'n_intervals')
)
def update_kpi0_prob(n):
    prob = get_current_probability()
    color = COLORS['stable'] if prob < 0.3 else COLORS['degrading'] if prob < 0.6 else COLORS['unstable']
    return f"{prob:.3f}", {**KPI_CARD, 'color': color}

@app.callback(
    [Output('kpi0-conf', 'children'), Output('kpi0-conf', 'style')],
    Input('tick', 'n_intervals')
)
def update_kpi0_conf(n):
    conf = get_current_confidence()
    color = COLORS['stable'] if conf > 0.7 else COLORS['degrading'] if conf > 0.5 else COLORS['unstable']
    return f"{conf:.3f}", {**KPI_CARD, 'color': color}

# KPI Callbacks - Aggregated Metrics
@app.callback(
    [Output('kpi1-shi', 'children'), Output('kpi1-shi', 'style')],
    Input('tick', 'n_intervals')
)
def update_kpi1_shi(n):
    shi = get_service_health_index()
    color = COLORS['stable'] if shi > 70 else COLORS['degrading'] if shi > 40 else COLORS['unstable']
    return f"{shi:.1f}", {**KPI_CARD, 'color': color}

@app.callback(
    [Output('kpi2-pri', 'children'), Output('kpi2-pri', 'style')],
    Input('tick', 'n_intervals')
)
def update_kpi2_pri(n):
    pri = get_prediction_reliability_index()
    color = COLORS['stable'] if pri > 0.7 else COLORS['degrading'] if pri > 0.4 else COLORS['unstable']
    return f"{pri:.2f}", {**KPI_CARD, 'color': color}

@app.callback(
    [Output('kpi3-leadtime', 'children'), Output('kpi3-leadtime', 'style')],
    Input('tick', 'n_intervals')
)
def update_kpi3_leadtime(n):
    lead_time = get_mean_predictive_lead_time()
    color = COLORS['insights']
    display_text = f"{lead_time:.1f}s" if lead_time > 0 else "N/A"
    return display_text, {**KPI_CARD, 'color': color}

@app.callback(
    [Output('kpi4-exposure', 'children'), Output('kpi4-exposure', 'style')],
    Input('tick', 'n_intervals')
)
def update_kpi4_exposure(n):
    exposure = get_instability_exposure_ratio()
    color = COLORS['stable'] if exposure < 10 else COLORS['degrading'] if exposure < 30 else COLORS['unstable']
    return f"{exposure:.1f}%", {**KPI_CARD, 'color': color}

# Chart Callbacks - Main Visualization Area
@app.callback(Output('chart-prob-trend', 'figure'), Input('tick', 'n_intervals'))
def update_chart_prob_trend(n):
    return create_instability_probability_trend()

@app.callback(Output('chart-conf-trend', 'figure'), Input('tick', 'n_intervals'))
def update_chart_conf_trend(n):
    return create_confidence_trend()

@app.callback(Output('chart-pri', 'figure'), Input('tick', 'n_intervals'))
def update_chart_pri(n):
    return create_pri_chart()

@app.callback(Output('chart-timeline', 'figure'), Input('tick', 'n_intervals'))
def update_chart_timeline(n):
    return create_state_timeline()

@app.callback(Output('chart-exposure', 'figure'), Input('tick', 'n_intervals'))
def update_chart_exposure(n):
    return create_instability_exposure_chart()

@app.callback(Output('chart-volatility', 'figure'), Input('tick', 'n_intervals'))
def update_chart_volatility(n):
    return create_volatility_indicator()

# Analytical Section Callbacks
@app.callback(Output('chart-distribution', 'figure'), Input('tick', 'n_intervals'))
def update_chart_distribution(n):
    return create_state_distribution()

@app.callback(Output('chart-leadtime', 'figure'), Input('tick', 'n_intervals'))
def update_chart_leadtime(n):
    return create_lead_time_analysis()

# Optional Summary Indicators
@app.callback(Output('summary-events', 'children'), Input('tick', 'n_intervals'))
def update_summary_events(n):
    count = get_instability_event_count()
    return str(count)

@app.callback(Output('summary-recovery', 'children'), Input('tick', 'n_intervals'))
def update_summary_recovery(n):
    avg_rt = get_average_recovery_time()
    return f"{avg_rt:.1f}s" if avg_rt > 0 else "N/A"

@app.callback(Output('summary-conf-unstable', 'children'), Input('tick', 'n_intervals'))
def update_summary_conf_unstable(n):
    avg_conf = get_avg_confidence_during_unstable()
    return f"{avg_conf:.2f}" if avg_conf > 0 else "N/A"

# ======================================================
# MAIN
# ======================================================
if __name__ == '__main__':
    # Start L1 receiver thread (only source of data - live connection required)
    threading.Thread(target=receive_l1_predictions, daemon=True).start()
    print(f"[L2] Dashboard starting on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"[L2] Waiting for L1 connection at {L1_HOST}:{L1_PORT}")
    print("[L2] Dashboard requires live L1 data connection to function.")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)

