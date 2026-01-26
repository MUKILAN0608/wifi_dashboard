import subprocess
import time
import threading
import re
import warnings
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np

# Suppress FutureWarning for DataFrame concatenation with empty/all-NA entries
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


# ================= CONFIGURATION =================
INTERFACE = "wlp1s0"
PING_TARGET = "8.8.8.8"
SAMPLE_INTERVAL_SECONDS = 0.5
MAX_DATA_POINTS = 600
DASHBOARD_PORT = 8050
DASHBOARD_HOST = "127.0.0.1"

# ================================================

# Thread-safe data storage
data_lock = threading.Lock()
connection_info = {"ssid": "N/A", "radio": "N/A", "channel": "N/A", "bssid": "N/A"}

# Initialize DataFrame with explicit dtypes to avoid FutureWarning
metrics_dataframe = pd.DataFrame({
    "timestamp": pd.Series(dtype="float64"),
    "signal_percent": pd.Series(dtype="Int64"),
    "rssi_dbm": pd.Series(dtype="Int64"),
    "rx_mbps": pd.Series(dtype="float64"),
    "tx_mbps": pd.Series(dtype="float64"),
    "link_speed_mbps": pd.Series(dtype="float64"),
    "channel": pd.Series(dtype="Int64"),
    "bandwidth_util": pd.Series(dtype="float64"),
    "rtt_ms": pd.Series(dtype="float64"),
    "rtt_jitter": pd.Series(dtype="float64"),
    "rssi_delta": pd.Series(dtype="float64"),
    "stability_score": pd.Series(dtype="float64"),
    "anomaly_flag": pd.Series(dtype="Int64")
})

# ================= DATA COLLECTION =================

def collect_wifi_metrics_linux() -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], Optional[int], Optional[str], Optional[str], Optional[str]]:
    """
    Collects Wi-Fi interface metrics from Linux system using iw and iwconfig.
    Optimized with timeout and error handling to prevent blocking.
    
    Returns:
        Tuple containing signal percentage, RSSI (dBm), RX rate (Mbps), TX rate (Mbps), Link Speed (Mbps), Channel, SSID, BSSID, Radio type.
        Returns None for any metric that cannot be retrieved.
    """
    try:
        # Get link info using iw
        iw_output = subprocess.check_output(
            ["iw", "dev", INTERFACE, "link"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2,
            stderr=subprocess.DEVNULL
        )
        
        # Get interface info using iwconfig
        iwconfig_output = subprocess.check_output(
            ["iwconfig", INTERFACE],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2,
            stderr=subprocess.DEVNULL
        )
        
        # Check if connected
        if "Not connected" in iw_output:
            return None, None, None, None, None, None, None, None, None
        
        # Parse iw output
        rssi_match = re.search(r"signal:\s+(-?\d+)", iw_output)
        rx_match = re.search(r"rx bitrate:\s+([\d\.]+)", iw_output)
        tx_match = re.search(r"tx bitrate:\s+([\d\.]+)", iw_output)
        ssid_match = re.search(r"SSID:\s+(.+)", iw_output)
        bssid_match = re.search(r"Connected to\s+([0-9a-fA-F:]{17})", iw_output)
        
        # Parse channel from both iw and iwconfig outputs for robustness
        channel_match = re.search(r"Channel:(\d+)", iwconfig_output)
        # Some iw outputs may have 'channel' in a different format
        iw_channel_match = re.search(r"channel\s+(\d+)", iw_output, re.IGNORECASE)
        signal_match = re.search(r"Signal level=(-?\d+)", iwconfig_output)
        
        # Get RSSI from iw (signal field) or iwconfig
        rssi = None
        if rssi_match:
            rssi = int(rssi_match.group(1))
        elif signal_match:
            rssi = int(signal_match.group(1))
        
        # Calculate signal percentage from RSSI (approximate: -30dBm = 100%, -90dBm = 0%)
        signal_percent = None
        if rssi is not None:
            signal_percent = max(0, min(100, int((rssi + 90) * 100 / 60)))
        
        # Get rates
        rx_rate = float(rx_match.group(1)) if rx_match else None
        tx_rate = float(tx_match.group(1)) if tx_match else None
        link_speed = (rx_rate + tx_rate) / 2 if (rx_rate and tx_rate) else (rx_rate or tx_rate)
        
        # Get channel (prefer iwconfig, fallback to iw if not found)
        if channel_match:
            channel = int(channel_match.group(1))
        elif iw_channel_match:
            channel = int(iw_channel_match.group(1))
        else:
            channel = None
        
        # Get SSID and BSSID
        ssid = ssid_match.group(1).strip() if ssid_match else None
        bssid = bssid_match.group(1).strip() if bssid_match else None
        
        # Determine radio type from iwconfig
        radio = None
        if "802.11ax" in iwconfig_output or "IEEE 802.11ax" in iwconfig_output:
            radio = "802.11ax"
        elif "802.11ac" in iwconfig_output or "IEEE 802.11ac" in iwconfig_output:
            radio = "802.11ac"
        elif "802.11n" in iwconfig_output or "IEEE 802.11n" in iwconfig_output:
            radio = "802.11n"
        elif "802.11g" in iwconfig_output or "IEEE 802.11g" in iwconfig_output:
            radio = "802.11g"
        elif "802.11a" in iwconfig_output or "IEEE 802.11a" in iwconfig_output:
            radio = "802.11a"
        else:
            radio = "802.11"
        
        return (
            signal_percent,
            rssi,
            rx_rate,
            tx_rate,
            link_speed,
            channel,
            ssid,
            bssid,
            radio
        )
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, ValueError, Exception) as e:
        # Silent error handling to prevent console spam
        return None, None, None, None, None, None, None, None, None

def collect_rtt_linux(target: str = PING_TARGET) -> Optional[float]:
    """
    Collects Round-Trip Time (RTT) latency using Linux ping command.
    
    Args:
        target: Ping target IP address or hostname
        
    Returns:
        RTT in milliseconds, or None if ping fails
    """
    try:
        # Linux ping command: ping -c 1 -W 1 target
        cmd = ["ping", "-c", "1", "-W", "1", target]
        output = subprocess.check_output(
            cmd,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=1.5,  # Reduced timeout to prevent blocking
            stderr=subprocess.DEVNULL
        )
        
        # Linux ping output pattern: "time=XX.XXX ms" or "time=XXms"
        patterns = [
            r"time=([\d\.]+)\s*ms",  # time=XX.XXX ms
            r"time=([\d\.]+)ms",  # time=XX.XXXms
            r"min/avg/max[^=]*=\s*[\d\.]+/[\d\.]+/([\d\.]+)",  # min/avg/max format
        ]
        
        for pattern in patterns:
            rtt_match = re.search(pattern, output, re.IGNORECASE)
            if rtt_match:
                try:
                    return float(rtt_match.group(1))
                except (ValueError, IndexError):
                    continue
            
        return None
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, ValueError, AttributeError, FileNotFoundError):
        return None

def collect_mac_retransmissions_linux() -> Optional[float]:
    """
    Collects MAC-layer retransmission statistics from Linux using iw station dump.
    
    Returns:
        Retry rate as a percentage, or None if not available
    """
    try:
        output = subprocess.check_output(
            ["iw", "dev", INTERFACE, "station", "dump"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2,
            stderr=subprocess.DEVNULL
        )
        
        # Parse retry statistics
        tx_retries_match = re.search(r"tx retries:\s+(\d+)", output)
        tx_failed_match = re.search(r"tx failed:\s+(\d+)", output)
        
        if tx_retries_match and tx_failed_match:
            tx_retries = int(tx_retries_match.group(1))
            tx_failed = int(tx_failed_match.group(1))
            total = tx_retries + tx_failed
            if total > 0:
                retry_rate = (tx_retries / total) * 100
                return retry_rate
        
        return None
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, ValueError, AttributeError, FileNotFoundError):
        return None

def calculate_stability_score(df: pd.DataFrame, window: int = 20) -> float:
    """
    Calculates link stability score (0-100) based on signal variance and throughput stability.
    
    Args:
        df: DataFrame with recent metrics
        window: Rolling window size for calculation
        
    Returns:
        Stability score from 0 (unstable) to 100 (perfectly stable)
    """
    if len(df) < window:
        return 50.0
    
    recent = df.tail(window)
    
    # Signal stability (0-50 points)
    signal_mean = recent['signal_percent'].mean() if not recent['signal_percent'].isna().all() else 50
    signal_std = recent['signal_percent'].std() if not recent['signal_percent'].isna().all() else 20
    signal_score = min(50, (signal_mean / 100) * 50 - (signal_std / 10) * 5)
    
    # Throughput stability (0-25 points)
    throughput_std = ((recent['rx_mbps'] + recent['tx_mbps']) / 2).std() if not (recent['rx_mbps'].isna().all() and recent['tx_mbps'].isna().all()) else 50
    throughput_score = max(0, 25 - (throughput_std / 10))
    
    # RTT stability (0-25 points)
    rtt_mean = recent['rtt_ms'].mean() if not recent['rtt_ms'].isna().all() else 50
    rtt_std = recent['rtt_ms'].std() if not recent['rtt_ms'].isna().all() else 20
    rtt_score = max(0, 25 - (rtt_mean / 100) * 5 - (rtt_std / 10) * 5)
    
    stability = max(0, min(100, signal_score + throughput_score + rtt_score))
    return stability

def detect_anomaly(df: pd.DataFrame, current_idx: int, window: int = 10) -> int:
    """
    Simple rule-based anomaly detection.
    Flags anomalies based on sudden drops in signal, high RTT, or throughput drops.
    
    Args:
        df: DataFrame with metrics
        current_idx: Current row index
        window: Window size for comparison
        
    Returns:
        1 if anomaly detected, 0 otherwise
    """
    # Validate inputs
    if df.empty or len(df) < window + 1:
        return 0
    
    # Validate index bounds
    if current_idx < 0 or current_idx >= len(df):
        return 0
    
    try:
        current = df.iloc[current_idx]
        recent = df.iloc[max(0, current_idx - window):current_idx]
    except (IndexError, KeyError):
        return 0
    
    try:
        # Check for signal drop > 20%
        if pd.notna(current['signal_percent']) and len(recent) > 0:
            signal_vals = recent['signal_percent'].dropna()
            if len(signal_vals) > 0:
                avg_signal = signal_vals.mean()
                if pd.notna(avg_signal) and avg_signal > 0 and pd.notna(current['signal_percent']) and current['signal_percent'] < avg_signal - 20:
                    return 1
        
        # Check for high RTT (> 100ms or sudden spike)
        if pd.notna(current['rtt_ms']):
            if current['rtt_ms'] > 100:
                return 1
            if len(recent) > 0:
                rtt_vals = recent['rtt_ms'].dropna()
                if len(rtt_vals) > 0:
                    avg_rtt = rtt_vals.mean()
                    if pd.notna(avg_rtt) and avg_rtt > 0 and pd.notna(current['rtt_ms']) and current['rtt_ms'] > avg_rtt * 2:
                        return 1
        
        # Check for throughput drop > 50%
        if pd.notna(current['rx_mbps']) and len(recent) > 0:
            rx_vals = recent['rx_mbps'].dropna()
            if len(rx_vals) > 0:
                avg_rx = rx_vals.mean()
                if pd.notna(avg_rx) and avg_rx > 0 and pd.notna(current['rx_mbps']) and current['rx_mbps'] < avg_rx * 0.5:
                    return 1
    except (KeyError, IndexError, ValueError, TypeError):
        # Return 0 (no anomaly) if any error occurs during detection
        return 0
    
    return 0

def calculate_link_state(df: pd.DataFrame, window: int = 20) -> Tuple[str, str]:
    """
    Calculates link state (STABLE/DEGRADING/UNSTABLE) based on recent metrics.
    
    Args:
        df: DataFrame with metric history
        window: Window size for analysis
        
    Returns:
        Tuple of (state_label, state_color)
    """
    if df.empty or len(df) < window:
        return "UNSTABLE", THEME_COLORS["error"]
    
    recent = df.tail(window)
    
    # Signal analysis
    signal_vals = recent['signal_percent'].dropna()
    signal_mean = signal_vals.mean() if len(signal_vals) > 0 else None
    signal_std = signal_vals.std() if len(signal_vals) > 0 else None
    
    # RTT analysis
    rtt_vals = recent['rtt_ms'].dropna()
    rtt_mean = rtt_vals.mean() if len(rtt_vals) > 0 else None
    rtt_std = rtt_vals.std() if len(rtt_vals) > 0 else None
    
    # PHY rate analysis
    rx_vals = recent['rx_mbps'].dropna()
    rx_std = rx_vals.std() if len(rx_vals) > 0 else None
    
    # Determine state
    if signal_mean is not None and signal_mean >= 70 and signal_std and signal_std < 5 and rtt_mean is not None and rtt_mean < 50 and rtt_std and rtt_std < 5:
        return "STABLE", THEME_COLORS["success"]
    elif signal_mean is not None and signal_mean >= 50 and (signal_std is None or signal_std < 15) and (rtt_std is None or rtt_std < 15):
        return "DEGRADING", THEME_COLORS["warning"]
    else:
        return "UNSTABLE", THEME_COLORS["error"]

def check_optimization_required(df: pd.DataFrame, latest: pd.Series) -> bool:
    """
    Determines if optimization is required based on link state and metrics.
    
    Args:
        df: DataFrame with metric history
        latest: Latest metric values
        
    Returns:
        True if optimization is required, False otherwise
    """
    if df.empty or len(df) < 10:
        return False
    
    state, _ = calculate_link_state(df)
    
    # Optimization required if unstable or degrading
    if state == "UNSTABLE":
        return True
    
    # Also check specific conditions
    if state == "DEGRADING":
        # Check if RTT is consistently high
        recent = df.tail(20)
        rtt_vals = recent['rtt_ms'].dropna()
        if len(rtt_vals) > 0:
            avg_rtt = rtt_vals.mean()
            if avg_rtt > 100:  # High latency threshold
                return True
        
        # Check if signal is consistently low
        signal_vals = recent['signal_percent'].dropna()
        if len(signal_vals) > 0:
            avg_signal = signal_vals.mean()
            if avg_signal < 40:  # Low signal threshold
                return True
    
    return False

def data_collection_worker():
    """
    Background worker thread for continuous metric collection.
    Collects Wi-Fi metrics at specified intervals with optimized performance.
    """
    global metrics_dataframe, connection_info
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metric collection worker initialized")
    print(f"Sample interval: {SAMPLE_INTERVAL_SECONDS}s")
    
    while True:
        try:
            timestamp = time.time()
            signal, rssi, rx_rate, tx_rate, link_speed, channel, ssid, bssid, radio = collect_wifi_metrics_linux()
            rtt_ms = collect_rtt_linux()
            
            # Calculate bandwidth utilization based on theoretical max for WiFi
            # Estimate max bandwidth based on radio type for better utilization calculation
            radio_lower = (radio or "").lower()
            if "802.11ax" in radio_lower or "wifi 6" in radio_lower:
                max_bandwidth = 2400
            elif "802.11ac" in radio_lower:
                max_bandwidth = 1300
            elif "802.11n" in radio_lower:
                max_bandwidth = 300
            else:
                max_bandwidth = 1000  # default
            avg_throughput = link_speed if link_speed else 0
            bw_util = (avg_throughput / max_bandwidth * 100) if avg_throughput else 0

            new_record = {
                "timestamp": timestamp,
                "signal_percent": signal,
                "rssi_dbm": rssi,
                "rx_mbps": rx_rate,
                "tx_mbps": tx_rate,
                "link_speed_mbps": link_speed,
                "channel": channel,
                "bandwidth_util": bw_util,
                "rtt_ms": rtt_ms,
                "rtt_jitter": None,  # Will be calculated after adding to dataframe
                "rssi_delta": None,  # Will be calculated after adding to dataframe
                "stability_score": 50.0,  # Will be calculated after adding to dataframe
                "anomaly_flag": 0  # Will be calculated after adding to dataframe
            }

            # Optimized data handling - minimize lock time
            with data_lock:
                # Update connection info for display
                connection_info["ssid"] = ssid or "N/A"
                connection_info["radio"] = radio or "N/A"
                connection_info["channel"] = str(channel) if channel else "N/A"
                connection_info["bssid"] = bssid or "N/A"
                
                # Use efficient append method with pd.concat (faster than loc assignment)
                new_df = pd.DataFrame([new_record])
                if len(metrics_dataframe) == 0:
                    metrics_dataframe = new_df
                else:
                    # Ensure both DataFrames have the same columns and dtypes to avoid FutureWarning
                    # Reindex new_df to match existing columns, filling missing with NaN
                    new_df = new_df.reindex(columns=metrics_dataframe.columns, fill_value=None)
                    # Ensure dtypes match
                    for col in metrics_dataframe.columns:
                        if col in new_df.columns:
                            new_df[col] = new_df[col].astype(metrics_dataframe[col].dtype, errors='ignore')
                    metrics_dataframe = pd.concat([metrics_dataframe, new_df], ignore_index=True, sort=False)
                
                # Trim data efficiently - only when needed (keep last MAX_DATA_POINTS)
                if len(metrics_dataframe) > MAX_DATA_POINTS:
                    metrics_dataframe = metrics_dataframe.iloc[-MAX_DATA_POINTS:].copy()
                    metrics_dataframe.reset_index(drop=True, inplace=True)
            
            # Calculate stability outside lock to minimize blocking - async style
            # Get current length and copy for calculations
            with data_lock:
                current_length = len(metrics_dataframe)
            if current_length >= 20:
                with data_lock:
                    df_copy = metrics_dataframe.copy()
            else:
                df_copy = None
            
            if df_copy is not None and len(df_copy) >= 20:
                try:
                    # Use the last index (which is always valid after copy)
                    current_idx = len(df_copy) - 1
                    
                    # Validate index before using
                    if current_idx >= 0 and current_idx < len(df_copy):
                        # Calculate outside lock
                        stability = calculate_stability_score(df_copy)
                        anomaly = detect_anomaly(df_copy, current_idx)
                        
                        # Calculate RSSI delta (difference from rolling average)
                        rssi_delta = None
                        if pd.notna(df_copy['rssi_dbm'].iloc[current_idx]):
                            # Use rolling window of 10 samples for average
                            window_size = min(10, current_idx + 1)
                            if window_size > 1:
                                recent_rssi = df_copy['rssi_dbm'].iloc[max(0, current_idx - window_size + 1):current_idx + 1]
                                recent_rssi_clean = recent_rssi.dropna()
                                if len(recent_rssi_clean) > 1:
                                    rssi_mean = recent_rssi_clean.iloc[:-1].mean()  # Mean of previous values
                                    current_rssi = df_copy['rssi_dbm'].iloc[current_idx]
                                    if pd.notna(rssi_mean) and pd.notna(current_rssi):
                                        rssi_delta = current_rssi - rssi_mean
                        
                        # Calculate RTT jitter (standard deviation of recent RTT values)
                        rtt_jitter = None
                        if pd.notna(df_copy['rtt_ms'].iloc[current_idx]):
                            window_size = min(10, current_idx + 1)
                            if window_size > 1:
                                recent_rtt = df_copy['rtt_ms'].iloc[max(0, current_idx - window_size + 1):current_idx + 1]
                                recent_rtt_clean = recent_rtt.dropna()
                                if len(recent_rtt_clean) > 1:
                                    rtt_jitter = recent_rtt_clean.std()
                        
                        # Update with lock - verify index is still valid
                        with data_lock:
                            # Re-check length in case DataFrame was trimmed
                            if current_idx < len(metrics_dataframe):
                                metrics_dataframe.at[current_idx, 'stability_score'] = stability
                                metrics_dataframe.at[current_idx, 'anomaly_flag'] = anomaly
                                if rssi_delta is not None:
                                    metrics_dataframe.at[current_idx, 'rssi_delta'] = rssi_delta
                                if rtt_jitter is not None:
                                    metrics_dataframe.at[current_idx, 'rtt_jitter'] = rtt_jitter
                except (IndexError, KeyError, ValueError) as calc_error:
                    # Log error for debugging but don't block
                    print(f"Warning: Calculation error (non-critical): {calc_error}")
                except Exception as calc_error:
                    # Silent error for other exceptions - don't block on calculation errors
                    pass

        except Exception as e:
            print(f"Error in data collection loop: {e}")
            # Continue even on error to prevent complete freeze
        
        time.sleep(SAMPLE_INTERVAL_SECONDS)

# Start background data collection
collection_thread = threading.Thread(target=data_collection_worker, daemon=True)
collection_thread.start()

# ================= DASHBOARD STYLING =================

# Premium industry-grade color palette with gradients
THEME_COLORS = {
    "background": "#0a0e1a",
    "background_gradient": "linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0f1419 100%)",
    "surface": "#151a24",
    "surface_glow": "rgba(88, 166, 255, 0.05)",
    "card": "#1a1f2e",
    "card_hover": "#1f2535",
    "card_glow": "rgba(88, 166, 255, 0.1)",
    "primary": "#5b9eff",
    "primary_glow": "rgba(91, 158, 255, 0.4)",
    "primary_dark": "#2563eb",
    "success": "#22c55e",
    "success_glow": "rgba(34, 197, 94, 0.3)",
    "success_dark": "#16a34a",
    "warning": "#f59e0b",
    "warning_glow": "rgba(245, 158, 11, 0.3)",
    "warning_dark": "#d97706",
    "error": "#ef4444",
    "error_glow": "rgba(239, 68, 68, 0.3)",
    "error_dark": "#dc2626",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_tertiary": "#64748b",
    "grid": "#1e293b",
    "border": "#334155",
    "border_glow": "rgba(88, 166, 255, 0.2)",
    "border_light": "#1e293b",
    "accent": "#2563eb",
    "accent_light": "#5b9eff",
    "glass": "rgba(26, 31, 46, 0.7)",
    "glass_border": "rgba(88, 166, 255, 0.2)"
}

APP_FONT_FAMILY = "'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', 'Roboto', sans-serif"

CARD_STYLE = {
    "background": THEME_COLORS["glass"],
    "padding": "18px",
    "borderRadius": "12px",
    "border": f"1px solid {THEME_COLORS['border_light']}",
    "boxShadow": "0 10px 30px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04)",
    "backdropFilter": "blur(14px)",
    "textAlign": "center",
    "transition": "all 0.3s ease"
}

GRAPH_STYLE = {
    "background": THEME_COLORS["glass"],
    "borderRadius": "12px",
    "border": f"1px solid {THEME_COLORS['border_light']}",
    "boxShadow": "0 14px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.04)",
    "backdropFilter": "blur(16px)",
    "padding": "16px",
    "minHeight": "380px"
}

# ================= DASH APPLICATION =================

app = dash.Dash(__name__)
app.title = "WiFi Network Performance Monitor"
# Force cache refresh by adding version to prevent stale layouts
app.config.suppress_callback_exceptions = False

# Premium CSS for smooth rendering
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <meta name="version" content="2.0.0">
        <title>{%title%}</title>
        <script>
            // Force reload if cached version detected
            if (window.performance && window.performance.navigation.type === 1) {
                // Page was reloaded, clear any cached callbacks
                if (window.dash_renderer) {
                    window.dash_renderer._callbacks = {};
                }
            }
        </script>
        {%favicon%}
        {%css%}
        <style>
            * { 
                -webkit-font-smoothing: antialiased; 
                -moz-osx-font-smoothing: grayscale; 
                box-sizing: border-box;
            }
            body { 
                overflow-x: hidden;
                margin: 0;
                padding: 0;
            }
            @media (max-width: 1200px) {
                .chart-grid-2 { grid-template-columns: 1fr !important; }
            }
            .card-hover-effect {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .card-hover-effect:hover {
                transform: translateY(-3px);
                box-shadow: 0 16px 44px rgba(0, 0, 0, 0.55), 0 8px 22px rgba(0, 0, 0, 0.35) !important;
            }
        </style>
    </head>
    <body>{%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

app.layout = html.Div(style={
    "background": THEME_COLORS["background_gradient"],
    "color": THEME_COLORS["text_primary"],
    "minHeight": "100vh",
    "padding": "0",
    "fontFamily": APP_FONT_FAMILY,
    "position": "relative",
    "overflowX": "hidden"
}, children=[
    # Minimal Header
    html.Header(style={
        "width": "100%",
        "background": "rgba(10, 14, 26, 0.75)",
        "backdropFilter": "blur(18px)",
        "padding": "14px 0",
        "borderBottom": f"1px solid {THEME_COLORS['border_light']}",
        "position": "sticky",
        "top": "0",
        "zIndex": "999"
    }, children=[
        html.Div(style={
            "maxWidth": "1400px",
            "margin": "0 auto",
            "padding": "0 20px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between"
        }, children=[
            html.Span("WiFi Monitor", style={
                "fontSize": "1.1rem",
                "fontWeight": "700",
                "color": THEME_COLORS["text_primary"],
                "letterSpacing": "0.2px"
            }),
            html.Div(id="connection-info", style={
                "display": "flex",
                "gap": "10px",
                "alignItems": "center",
                "flexWrap": "wrap",
                "justifyContent": "flex-end"
            })
        ])
    ]),

    html.Div(style={"maxWidth": "1400px", "margin": "0 auto", "padding": "24px 20px"}, children=[
        # Auto-refresh interval
        dcc.Interval(
            id="update-interval",
            interval=1000,
            n_intervals=0
        ),

        # KPI Section
        html.Div(style={
            "marginBottom": "32px"
        }, children=[
            html.Div(id="kpi-container", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "20px"
            })
        ]),
        
        # Metrics Cards Container
        html.Div(style={
            "marginBottom": "32px"
        }, children=[
            html.Div(id="metrics-container", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "20px"
            })
        ]),

        # Charts Section - Clean 2-Column Grid
        html.Div(className="chart-grid-2", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "18px",
            "marginBottom": "28px"
        }, children=[
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE},
                children=[
                    dcc.Graph(id="chart1-signal-strength", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE},
                children=[
                    dcc.Graph(id="chart3-rx-tx-phy-rate", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE},
                children=[
                    dcc.Graph(id="chart4-rtt-jitter", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE},
                children=[
                    dcc.Graph(id="chart5-stability-gauge", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE},
                children=[
                    dcc.Graph(id="chart6-anomaly-timeline", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE},
                children=[
                    dcc.Graph(id="chart8", config={"displayModeBar": False, "responsive": True})
                ]
            )
        ]),

        # RSSI Variation Chart - Full Width
        html.Div(style={
            "marginBottom": "32px"
        }, children=[
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE},
                children=[
                    dcc.Graph(id="chart7-rssi-delta", config={"displayModeBar": False, "responsive": True})
                ]
            )
        ]),
    ])

])


# ================= CALLBACKS =================
@app.callback(
    Output("connection-info", "children"),
    Output("kpi-container", "children"),
    Output("metrics-container", "children"),
    Output("chart1-signal-strength", "figure"),
    Output("chart3-rx-tx-phy-rate", "figure"),
    Output("chart4-rtt-jitter", "figure"),
    Output("chart5-stability-gauge", "figure"),
    Output("chart6-anomaly-timeline", "figure"),
    Output("chart7-rssi-delta", "figure"),
    Output("chart8", "figure"),
    Input("update-interval", "n_intervals")
)
def update_dashboard_components(n: int):
    """
    Updates all dashboard components with latest metrics data.
    Optimized to prevent blocking and handle errors gracefully.
    
    Args:
        n: Number of intervals elapsed (not used, required by Dash)
        
    Returns:
        Tuple containing connection info, KPIs, metrics, and charts
    """
    try:
        # Quick lock - just get data copy
        with data_lock:
            df = metrics_dataframe.copy()
            conn_info = connection_info.copy()
    except Exception as e:
        # If lock fails, return empty state
        empty_layout = create_empty_figure_layout()
        return [], [], [], empty_layout, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout

    ssid = str(conn_info.get("ssid", "")).strip()
    connected = bool(ssid and ssid.upper() not in {"N/A", "NOT CONNECTED", "DISCONNECTED"})

    chip_base = {
        "padding": "8px 12px",
        "background": THEME_COLORS["glass"],
        "borderRadius": "10px",
        "border": f"1px solid {THEME_COLORS['border_light']}",
        "backdropFilter": "blur(14px)",
        "display": "flex",
        "alignItems": "center",
        "gap": "8px",
        "lineHeight": "1",
    }

    def build_conn_badges(health_label: str, health_color: str) -> list:
        return [
            html.Div(
                style={**chip_base, "fontSize": "13px", "color": THEME_COLORS["text_primary"]},
                children=[
                    html.Span(style={
                        "display": "inline-block",
                        "width": "8px",
                        "height": "8px",
                        "borderRadius": "999px",
                        "background": health_color,
                        "boxShadow": f"0 0 0 3px {health_color}20"
                    }),
                    html.Span(health_label, style={"fontWeight": "600"})
                ]
            ),
            html.Div(
                style={**chip_base, "fontSize": "13px", "color": THEME_COLORS["text_primary"]},
                children=[html.Span(conn_info.get("ssid", "N/A"), style={"fontWeight": "500"})]
            ),
            html.Div(
                style={**chip_base, "fontSize": "13px", "color": THEME_COLORS["text_primary"]},
                children=[html.Span(conn_info.get("radio", "N/A"), style={"fontWeight": "500"})]
            ),
            html.Div(
                style={
                    **chip_base,
                    "fontSize": "11px",
                    "color": THEME_COLORS["text_secondary"],
                    "fontFamily": "'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace"
                },
                children=[html.Span(conn_info.get("bssid", "N/A"), style={"fontWeight": "400"})]
            )
        ]

    if df.empty:
        empty_layout = create_empty_figure_layout()
        if not connected:
            conn_badges = build_conn_badges("Offline", THEME_COLORS["text_tertiary"])
        else:
            conn_badges = build_conn_badges("Starting", THEME_COLORS["primary"])
        return conn_badges, [], [], empty_layout, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout

    try:
        latest_metrics = df.iloc[-1]
    except (IndexError, KeyError):
        empty_layout = create_empty_figure_layout()
        if not connected:
            conn_badges = build_conn_badges("Offline", THEME_COLORS["text_tertiary"])
        else:
            conn_badges = build_conn_badges("Starting", THEME_COLORS["primary"])
        return conn_badges, [], [], empty_layout, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout

    # Health status derived from live metrics
    try:
        stability = float(latest_metrics.get("stability_score", 50.0))
    except Exception:
        stability = 50.0
    try:
        anomaly = int(latest_metrics.get("anomaly_flag", 0) or 0)
    except Exception:
        anomaly = 0

    if not connected:
        health_label, health_color = "Offline", THEME_COLORS["text_tertiary"]
    elif anomaly == 1:
        health_label, health_color = "Issue", THEME_COLORS["error"]
    elif stability >= 80:
        health_label, health_color = "Good", THEME_COLORS["success"]
    elif stability >= 60:
        health_label, health_color = "OK", THEME_COLORS["primary"]
    elif stability >= 40:
        health_label, health_color = "Fair", THEME_COLORS["warning"]
    else:
        health_label, health_color = "Poor", THEME_COLORS["error"]

    conn_badges = build_conn_badges(health_label, health_color)

    # Generate KPI cards with error handling
    try:
        kpi_cards = create_kpi_cards(df)
    except Exception:
        kpi_cards = []

    # Generate metric cards with error handling
    try:
        metric_cards = create_metric_cards(latest_metrics)
    except Exception:
        metric_cards = []

    # Generate chart figures
    try:
        chart1 = create_chart1_signal_strength(df)
    except Exception:
        chart1 = create_empty_figure_layout()
    try:
        chart3 = create_chart3_rx_tx_phy_rate(df)
    except Exception:
        chart3 = create_empty_figure_layout()
    try:
        chart4 = create_chart4_rtt_jitter(df)
    except Exception:
        chart4 = create_empty_figure_layout()
    try:
        chart5 = create_chart5_stability_gauge(df)
    except Exception:
        chart5 = create_empty_figure_layout()
    try:
        chart6 = create_chart6_anomaly_timeline(df)
    except Exception:
        chart6 = create_empty_figure_layout()
    try:
        chart7 = create_chart7_rssi_delta(df)
    except Exception:
        chart7 = create_empty_figure_layout()
    try:
        chart8 = create_chart8_rssi_rtt_scatter(df)
    except Exception:
        chart8 = create_empty_figure_layout()

    return conn_badges, kpi_cards, metric_cards, chart1, chart3, chart4, chart5, chart6, chart7, chart8

def create_empty_figure_layout() -> dict:
    """Creates clean empty figure for when no data is available."""
    return {
        "data": [],
        "layout": {
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "font": {"color": THEME_COLORS["text_tertiary"]},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": "Collecting data...",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": THEME_COLORS["text_tertiary"]}
            }]
        }
    }

def create_metric_cards(latest: pd.Series) -> list:
    """Generates clean metric display cards."""
    metrics = [
        {
            "label": "Signal",
            "value": f"{latest['signal_percent']:.0f}%" if pd.notna(latest['signal_percent']) else "N/A",
            "color": THEME_COLORS["success"]
        },
        {
            "label": "RSSI",
            "value": f"{latest['rssi_dbm']:.0f} dBm" if pd.notna(latest['rssi_dbm']) else "N/A",
            "color": THEME_COLORS["primary"]
        },
        {
            "label": "RX Rate",
            "value": f"{latest['rx_mbps']:.1f} Mbps" if pd.notna(latest['rx_mbps']) else "N/A",
            "color": THEME_COLORS["primary"]
        },
        {
            "label": "TX Rate",
            "value": f"{latest['tx_mbps']:.1f} Mbps" if pd.notna(latest['tx_mbps']) else "N/A",
            "color": THEME_COLORS["warning"]
        },
        {
            "label": "Channel",
            "value": f"{int(latest['channel'])}" if pd.notna(latest['channel']) else "N/A",
            "color": THEME_COLORS["primary"]
        }
    ]
    
    return [
        html.Div(
            style={**CARD_STYLE},
            children=[
                html.Div(metric["label"], style={
                    "fontSize": "11px",
                    "color": THEME_COLORS["text_tertiary"],
                    "marginBottom": "8px",
                    "fontWeight": "500",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.5px"
                }),
                html.Div(metric["value"], style={
                    "fontSize": "24px",
                    "fontWeight": "600",
                    "color": metric["color"]
                })
            ]
        ) for metric in metrics
    ]

def create_kpi_cards(df: pd.DataFrame) -> list:
    """Generates clean KPI summary cards."""
    avg_signal = df['signal_percent'].dropna().mean() if len(df['signal_percent'].dropna()) > 0 else 0
    avg_rssi = df['rssi_dbm'].dropna().mean() if len(df['rssi_dbm'].dropna()) > 0 else 0
    
    if avg_signal >= 80:
        signal_color = THEME_COLORS["success"]
    elif avg_signal >= 60:
        signal_color = THEME_COLORS["primary"]
    elif avg_signal >= 40:
        signal_color = THEME_COLORS["warning"]
    else:
        signal_color = THEME_COLORS["error"]
    
    kpis = [
        {
            "label": "Avg Signal",
            "value": f"{avg_signal:.0f}%",
            "color": signal_color
        },
        {
            "label": "Avg RSSI",
            "value": f"{avg_rssi:.0f} dBm",
            "color": THEME_COLORS["primary"]
        }
    ]
    
    return [
        html.Div(
            style={
                **CARD_STYLE,
                "borderLeft": f"3px solid {kpi['color']}"
            },
            children=[
                html.Div(kpi["label"], style={
                    "fontSize": "11px",
                    "color": THEME_COLORS["text_tertiary"],
                    "marginBottom": "8px",
                    "fontWeight": "500",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.5px"
                }),
                html.Div(kpi["value"], style={
                    "fontSize": "28px",
                    "fontWeight": "600",
                    "color": kpi["color"]
                })
            ]
        ) for kpi in kpis
    ]

# --- Chart Layout Helper Stub ---
def create_base_layout(title: str = "") -> dict:
    """Clean professional chart layout."""
    layout = {
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {"color": THEME_COLORS["text_secondary"], "size": 12},
        "margin": {"l": 50, "r": 30, "t": 20, "b": 50},
        "xaxis": {
            "gridcolor": f"{THEME_COLORS['grid']}80",
            "gridwidth": 1,
            "showgrid": True,
            "zeroline": False,
            "color": THEME_COLORS["text_tertiary"],
            "title": {"font": {"size": 11, "color": THEME_COLORS["text_secondary"]}}
        },
        "yaxis": {
            "gridcolor": f"{THEME_COLORS['grid']}80",
            "gridwidth": 1,
            "showgrid": True,
            "zeroline": False,
            "color": THEME_COLORS["text_tertiary"],
            "title": {"font": {"size": 11, "color": THEME_COLORS["text_secondary"]}}
        },
        "legend": {
            "bgcolor": "transparent",
            "bordercolor": "transparent",
            "font": {"size": 11, "color": THEME_COLORS["text_secondary"]},
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top"
        },
        "autosize": True,
        "hovermode": "x unified"
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"size": 14, "color": THEME_COLORS["text_primary"], "weight": "600"},
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top"
        }
    return layout

# ================= CHART 1: SIGNAL STRENGTH VS TIME =================

def create_chart1_signal_strength(df: pd.DataFrame) -> dict:
    """
    Chart 1: Signal Strength vs Time (Line Chart)
    Shows RSSI (dBm) and Signal % over time to visualize mobility, fading, and interference.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    
    return {
        "data": [
            go.Scatter(
                x=timestamps,
                y=df["rssi_dbm"],
                mode="lines",
                name="RSSI (dBm)",
                line={"color": THEME_COLORS["primary"], "width": 2.5},
                yaxis="y",
                hovertemplate="<b>RSSI</b><br>%{y} dBm<br>%{x}<extra></extra>"
            ),
            go.Scatter(
                x=timestamps,
                y=df["signal_percent"],
                mode="lines",
                name="Signal (%)",
                line={"color": THEME_COLORS["success"], "width": 2.5},
                yaxis="y2",
                hovertemplate="<b>Signal</b><br>%{y}%<br>%{x}<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("Signal"),
            "yaxis": {
                **create_base_layout("Signal")["yaxis"],
                "title": {"text": "RSSI (dBm)", "font": {"size": 11, "color": THEME_COLORS["primary"]}},
                "side": "left"
            },
            "yaxis2": {
                "title": {"text": "Signal (%)", "font": {"size": 11, "color": THEME_COLORS["success"]}},
                "overlaying": "y",
                "side": "right",
                "range": [0, 100],
                "gridcolor": "transparent"
            }
        }
    }

# ================= CHART 3: RX VS TX PHY RATE =================

def create_chart3_rx_tx_phy_rate(df: pd.DataFrame) -> dict:
    """
    Chart 3: RX vs TX PHY Rate (Dual-Axis Line Chart)
    Shows rate locking, fallback, and asymmetry between RX and TX rates.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    
    # Calculate data ranges for proper axis separation
    rx_vals = df["rx_mbps"].dropna()
    tx_vals = df["tx_mbps"].dropna()
    
    if len(rx_vals) == 0 or len(tx_vals) == 0:
        return create_empty_figure_layout()
    
    rx_min, rx_max = rx_vals.min(), rx_vals.max()
    tx_min, tx_max = tx_vals.min(), tx_vals.max()
    
    # Calculate ranges with padding
    rx_range = rx_max - rx_min if rx_max > rx_min else max(rx_max * 0.01, 10)
    tx_range = tx_max - tx_min if tx_max > tx_min else max(tx_max * 0.01, 10)
    
    # Add padding to ranges (5% on each side)
    rx_padding = rx_range * 0.05 if rx_range > 0 else max(rx_max * 0.01, 5)
    tx_padding = tx_range * 0.05 if tx_range > 0 else max(tx_max * 0.01, 5)
    
    # Set axis ranges to prevent overlap
    rx_axis_range = [max(0, rx_min - rx_padding), rx_max + rx_padding]
    tx_axis_range = [max(0, tx_min - tx_padding), tx_max + tx_padding]
    
    return {
        "data": [
            go.Scatter(
                x=timestamps,
                y=df["rx_mbps"],
                mode="lines+markers",
                name="RX PHY Rate",
                line={"color": THEME_COLORS["primary"], "width": 3},
                marker={"size": 4, "color": THEME_COLORS["primary"], "symbol": "circle", "opacity": 0.7},
                yaxis="y",
                hovertemplate="<b>RX PHY Rate</b><br>%{y:.1f} Mbps<br>%{x}<extra></extra>"
            ),
            go.Scatter(
                x=timestamps,
                y=df["tx_mbps"],
                mode="lines+markers",
                name="TX PHY Rate",
                line={"color": THEME_COLORS["warning"], "width": 3, "dash": "dashdot"},
                marker={"size": 4, "color": THEME_COLORS["warning"], "symbol": "diamond", "opacity": 0.7},
                yaxis="y2",
                hovertemplate="<b>TX PHY Rate</b><br>%{y:.1f} Mbps<br>%{x}<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("PHY Rate"),
            "yaxis": {
                **create_base_layout("PHY Rate")["yaxis"],
                "title": {"text": "RX Rate (Mbps)", "font": {"size": 11, "color": THEME_COLORS["primary"]}},
                "side": "left",
                "range": rx_axis_range
            },
            "yaxis2": {
                "title": {"text": "TX Rate (Mbps)", "font": {"size": 11, "color": THEME_COLORS["warning"]}},
                "overlaying": "y",
                "side": "right",
                "range": tx_axis_range,
                "gridcolor": "transparent"
            }
        }
    }

# ================= CHART 4: RTT VARIABILITY =================

def create_chart4_rtt_jitter(df: pd.DataFrame) -> dict:
    """
    Chart 4: RTT Variance (Jitter) Over Time
    Shows RTT variance/jitter over time to reveal network turbulence.
    Mean RTT alone hides instability; variance shows quality issues.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    
    # Use rtt_jitter if available, otherwise calculate from RTT variance
    if 'rtt_jitter' in df.columns and df['rtt_jitter'].notna().any():
        jitter_vals = df["rtt_jitter"].dropna()
        valid_indices = df["rtt_jitter"].notna()
    else:
        # Calculate jitter as rolling standard deviation
        rtt_vals = df["rtt_ms"].dropna()
        if len(rtt_vals) == 0:
            return create_empty_figure_layout()
        
        window_size = min(10, len(rtt_vals))
        if window_size > 1:
            jitter_series = rtt_vals.rolling(window=window_size, center=True).std()
            jitter_series = jitter_series.bfill().ffill()
            jitter_vals = jitter_series.dropna()
            valid_indices = jitter_series.notna()
        else:
            jitter_vals = pd.Series([0] * len(rtt_vals))
            valid_indices = df["rtt_ms"].notna()
    
    if len(jitter_vals) == 0:
        return create_empty_figure_layout()
    
    # Get corresponding timestamps
    if isinstance(valid_indices, pd.Series):
        valid_timestamps = [timestamps[i] for i in range(len(timestamps)) if i < len(valid_indices) and valid_indices.iloc[i]]
    else:
        valid_timestamps = [timestamps[i] for i in range(len(timestamps)) if i < len(valid_indices) and valid_indices[i]]
    
    jitter_list = jitter_vals.tolist()
    
    # Calculate mean jitter for reference
    mean_jitter = jitter_vals.mean() if len(jitter_vals) > 0 else 0.0
    
    return {
        "data": [
            # Jitter values
            go.Scatter(
                x=valid_timestamps,
                y=jitter_list,
                mode="lines+markers",
                name="RTT Jitter",
                line={"color": THEME_COLORS["warning"], "width": 2.5},
                marker={"size": 4, "color": THEME_COLORS["warning"], "opacity": 0.7},
                fill="tozeroy",
                fillcolor=f"rgba(248, 81, 73, 0.2)",
                hovertemplate="<b>RTT Jitter</b><br>%{y:.2f} ms<br>%{x}<extra></extra>"
            ),
            # Mean jitter reference line
            go.Scatter(
                x=[valid_timestamps[0], valid_timestamps[-1]] if len(valid_timestamps) > 0 else [],
                y=[mean_jitter, mean_jitter],
                mode="lines",
                name="Mean Jitter",
                line={"color": THEME_COLORS["success"], "width": 1.5, "dash": "dot"},
                hovertemplate=f"<b>Mean Jitter</b><br>{mean_jitter:.2f} ms<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("RTT Jitter"),
            "yaxis": {
                **create_base_layout("RTT Jitter")["yaxis"],
                "title": {"text": "Jitter (ms)", "font": {"size": 11, "color": THEME_COLORS["warning"]}}
            }
        }
    }

# ================= CHART 5: LINK STABILITY SCORE =================

def create_chart5_stability_gauge(df: pd.DataFrame) -> dict:
    """
    Chart 5: Link Stability Score (Gauge/Indicator)
    ML-compressed representation showing stability score (0-100).
    """
    if df.empty:
        return create_empty_figure_layout()
    
    current_stability = df["stability_score"].iloc[-1] if pd.notna(df["stability_score"].iloc[-1]) else 50.0
    
    # Determine color based on stability
    if current_stability >= 80:
        gauge_color = THEME_COLORS["success"]
    elif current_stability >= 60:
        gauge_color = THEME_COLORS["primary"]
    elif current_stability >= 40:
        gauge_color = THEME_COLORS["warning"]
    else:
        gauge_color = THEME_COLORS["error"]
    
    return {
        "data": [
            go.Indicator(
                mode="gauge+number+delta",
                value=current_stability,
                domain={"x": [0, 1], "y": [0, 1]},
                delta={"reference": 50, "position": "top"},
                gauge={
                    "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": THEME_COLORS["text_secondary"]},
                    "bar": {"color": gauge_color},
                    "bgcolor": THEME_COLORS["card"],
                    "borderwidth": 2,
                    "bordercolor": THEME_COLORS["border"],
                    "steps": [
                        {"range": [0, 40], "color": THEME_COLORS["error"]},
                        {"range": [40, 60], "color": THEME_COLORS["warning"]},
                        {"range": [60, 80], "color": THEME_COLORS["primary"]},
                        {"range": [80, 100], "color": THEME_COLORS["success"]}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            )
        ],
        "layout": {
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "font": {"color": THEME_COLORS["text_secondary"], "size": 12},
            "margin": {"l": 50, "r": 50, "t": 20, "b": 50},
            "title": {
                "text": "Stability",
                "font": {"size": 14, "color": THEME_COLORS["text_primary"], "weight": "600"},
                "x": 0.5,
                "xanchor": "center",
                "y": 0.98,
                "yanchor": "top"
            },
            "autosize": True
        }
    }

# ================= CHART 6: ANOMALY TIMELINE =================

def create_chart6_anomaly_timeline(df: pd.DataFrame) -> dict:
    """
    Chart 6: Anomaly Timeline (Step Plot - Binary Timeline)
    Shows when abnormal behavior occurred, correlates with RTT spikes.
    """
    if df.empty:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    anomaly_flags = df["anomaly_flag"].fillna(0).astype(int)
    
    return {
        "data": [
            go.Scatter(
                x=timestamps,
                y=anomaly_flags,
                mode="lines+markers",
                name="Anomaly",
                line={"shape": "hv", "color": THEME_COLORS["error"], "width": 2},
                marker={"color": THEME_COLORS["error"], "size": 6},
                fill="tozeroy",
                fillcolor=f"rgba(239, 83, 80, 0.3)",
                hovertemplate="<b>Anomaly</b><br>%{y}<br>%{x}<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("Anomalies"),
            "yaxis": {
                **create_base_layout("Anomalies")["yaxis"],
                "title": {"text": "Anomaly", "font": {"size": 11, "color": THEME_COLORS["error"]}},
                "range": [-0.1, 1.1],
                "tickmode": "linear",
                "tick0": 0,
                "dtick": 1,
                "tickvals": [0, 1],
                "ticktext": ["Normal", "Anomaly"]
            }
        }
    }

# ================= CHART 7: RSSI DELTA (ΔRSSI) =================

def create_chart7_rssi_delta(df: pd.DataFrame) -> dict:
    """
    Chart 7: RSSI Variation Relative to Rolling Mean (ΔRSSI)
    Shows difference between current RSSI and rolling average.

    Captures signal dynamics, mobility, and fading effects.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    rssi_delta = df["rssi_delta"].dropna()
    
    if len(rssi_delta) == 0:
        return create_empty_figure_layout()
    
    valid_timestamps = [timestamps[i] for i in range(len(timestamps)) if pd.notna(df["rssi_delta"].iloc[i])]
    
    return {
        "data": [
            go.Scatter(
                x=valid_timestamps,
                y=rssi_delta,
                mode="lines+markers",
                name="ΔRSSI",
                line={"color": THEME_COLORS["primary"], "width": 2.5},
                marker={"size": 4, "color": THEME_COLORS["primary"]},
                fill="tozeroy",
                fillcolor=f"rgba(66, 153, 225, 0.2)",
                hovertemplate="<b>ΔRSSI</b><br>%{y:.1f} dB<br>%{x}<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("RSSI Δ"),
            "yaxis": {
                **create_base_layout("RSSI Δ")["yaxis"],
                "title": {"text": "ΔRSSI (dB)", "font": {"size": 11, "color": THEME_COLORS["primary"]}}
            }
        }
    }

# ================= CHART 8: RSSI VS RTT SCATTER =================

def create_chart8_rssi_rtt_scatter(df: pd.DataFrame) -> dict:
    """
    Chart 8: Relationship Between Signal Strength and Network Latency (Scatter Plot)
    Shows correlation (or lack thereof) between RSSI and RTT.
    Color-coded by link state.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    # Get valid data points
    valid_data = df[df['rssi_dbm'].notna() & df['rtt_ms'].notna()].copy()
    
    if len(valid_data) == 0:
        return create_empty_figure_layout()
    
    # Calculate link state for each point (using rolling window)
    states = []
    colors = []
    
    for idx in valid_data.index:
        window_start = max(0, idx - 10)
        window_df = df.iloc[window_start:idx+1]
        if len(window_df) >= 5:
            state, state_color = calculate_link_state(window_df, window=min(10, len(window_df)))
        else:
            state, state_color = "UNSTABLE", THEME_COLORS["error"]
        states.append(state)
        colors.append(state_color)
    
    # Group by state for separate traces
    scatter_data = []
    for state in ["STABLE", "DEGRADING", "UNSTABLE"]:
        state_mask = [s == state for s in states]
        if any(state_mask):
            state_rssi = valid_data['rssi_dbm'].iloc[state_mask].tolist()
            state_rtt = valid_data['rtt_ms'].iloc[state_mask].tolist()
            state_color = THEME_COLORS["success"] if state == "STABLE" else THEME_COLORS["warning"] if state == "DEGRADING" else THEME_COLORS["error"]
            
            scatter_data.append(go.Scatter(
                x=state_rssi,
                y=state_rtt,
                mode="markers",
                name=state,
                marker={
                    "size": 6,
                    "color": state_color,
                    "opacity": 0.7,
                    "line": {"width": 1, "color": state_color}
                },
                hovertemplate="<b>%{fullData.name}</b><br>RSSI: %{x} dBm<br>RTT: %{y:.1f} ms<extra></extra>"
            ))
    
    return {
        "data": scatter_data,
        "layout": {
            **create_base_layout("RSSI vs RTT"),
            "xaxis": {
                **create_base_layout("RSSI vs RTT")["xaxis"],
                "title": {"text": "RSSI (dBm)", "font": {"size": 11, "color": THEME_COLORS["primary"]}}
            },
            "yaxis": {
                **create_base_layout("RSSI vs RTT")["yaxis"],
                "title": {"text": "RTT (ms)", "font": {"size": 11, "color": THEME_COLORS["warning"]}}
            }
        }
    }

# ================= APPLICATION ENTRY POINT =================

def main():
    """Main entry point for the dashboard application."""
    print("=" * 60)
    print("Wireless Network Performance Monitor")
    print("=" * 60)
    print(f"Dashboard URL: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        app.run(
            host=DASHBOARD_HOST,
            port=DASHBOARD_PORT,
            debug=False,
            dev_tools_hot_reload=False
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
