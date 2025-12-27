import subprocess
import time
import threading
import re
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np

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
        # Get connection info using 'iw dev <interface> link'
        link_output = subprocess.check_output(
            ["iw", "dev", INTERFACE, "link"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2,
            stderr=subprocess.DEVNULL
        )
        
        # Get detailed info using 'iw dev <interface> info'
        info_output = subprocess.check_output(
            ["iw", "dev", INTERFACE, "info"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2,
            stderr=subprocess.DEVNULL
        )
        
        # Get signal strength using iwconfig (more reliable for signal percentage)
        try:
            iwconfig_output = subprocess.check_output(
                ["iwconfig", INTERFACE],
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=2,
                stderr=subprocess.DEVNULL
            )
        except:
            iwconfig_output = ""
        
        # Parse SSID
        ssid_match = re.search(r"SSID:\s*(.+)", link_output)
        if not ssid_match:
            ssid_match = re.search(r'ESSID:"([^"]+)"', iwconfig_output)
        ssid = ssid_match.group(1).strip() if ssid_match else None
        
        # Parse BSSID (MAC address)
        bssid_match = re.search(r"Connected to\s+([0-9a-fA-F:]{17})", link_output)
        if not bssid_match:
            bssid_match = re.search(r"Access Point:\s+([0-9a-fA-F:]{17})", link_output)
        bssid = bssid_match.group(1).strip() if bssid_match else None
        
        # Parse signal strength (dBm)
        rssi_match = re.search(r"signal:\s*(-?\d+)\s*dBm", link_output)
        if not rssi_match:
            rssi_match = re.search(r"Signal level=(-?\d+)\s*dBm", iwconfig_output)
        rssi = int(rssi_match.group(1)) if rssi_match else None
        
        # Convert RSSI to signal percentage (typical range: -30 to -90 dBm)
        # -30 dBm = 100%, -90 dBm = 0%
        if rssi is not None:
            signal_percent = max(0, min(100, int((rssi + 90) / 60 * 100)))
        else:
            # Try to get signal percentage from iwconfig
            signal_match = re.search(r"Signal level=(-?\d+)/(\d+)", iwconfig_output)
            if signal_match:
                signal_val = int(signal_match.group(1))
                signal_max = int(signal_match.group(2))
                signal_percent = int((signal_val / signal_max) * 100) if signal_max > 0 else None
            else:
                signal_percent = None
        
        # Parse channel
        channel_match = re.search(r"freq:\s*(\d+)", link_output)
        if not channel_match:
            channel_match = re.search(r"Channel:\s*(\d+)", iwconfig_output)
        channel = None
        if channel_match:
            freq = int(channel_match.group(1))
            # Convert frequency to channel (2.4 GHz: 2412-2484, 5 GHz: 5180-5825)
            if 2412 <= freq <= 2484:
                channel = (freq - 2412) // 5 + 1
            elif 5180 <= freq <= 5825:
                channel = (freq - 5180) // 5 + 36
        
        # Parse radio type/standard
        radio_match = re.search(r"type\s+(\w+)", info_output)
        if not radio_match:
            radio_match = re.search(r"IEEE\s+802\.11(\w+)", iwconfig_output)
        radio = None
        if radio_match:
            radio_type = radio_match.group(1).upper()
            if "A" in radio_type or "AC" in radio_type:
                radio = "802.11ac"
            elif "N" in radio_type:
                radio = "802.11n"
            elif "AX" in radio_type or "6" in radio_type:
                radio = "802.11ax"
            else:
                radio = f"802.11{radio_type}"
        
        # Get TX bitrate (Mbps)
        tx_match = re.search(r"tx bitrate:\s*([\d\.]+)\s*MBit/s", link_output)
        if not tx_match:
            tx_match = re.search(r"Bit Rate[:=]\s*([\d\.]+)\s*Mb/s", iwconfig_output)
        tx_rate = float(tx_match.group(1)) if tx_match else None
        
        # RX rate is typically same as TX for most interfaces
        rx_rate = tx_rate
        
        # Calculate link speed
        link_speed = (rx_rate + tx_rate) / 2 if (rx_rate and tx_rate) else (rx_rate or tx_rate or None)
        
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
            timeout=1.5,
            stderr=subprocess.DEVNULL
        )
        
        # Linux ping output pattern: "time=XX.XXX ms" or "time=XXms"
        patterns = [
            r"time=([\d\.]+)\s*ms",  # time=XX.XXX ms
            r"time<(\d+)\s*ms",  # time<1ms
            r"(\d+\.?\d*)\s*ms",  # Just XX.XXX ms (fallback)
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
                    metrics_dataframe = pd.concat([metrics_dataframe, new_df], ignore_index=True)
                
                # Trim data efficiently - only when needed (keep last MAX_DATA_POINTS)
                if len(metrics_dataframe) > MAX_DATA_POINTS:
                    metrics_dataframe = metrics_dataframe.iloc[-MAX_DATA_POINTS:].copy()
                    metrics_dataframe.reset_index(drop=True, inplace=True)
            
            # Calculate stability outside lock to minimize blocking - async style
            # Get current length and copy for calculations
            with data_lock:
                current_length = len(metrics_dataframe)
                if current_length >= 20:
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
                        
                        # Update with lock - verify index is still valid
                        with data_lock:
                            # Re-check length in case DataFrame was trimmed
                            if current_idx < len(metrics_dataframe):
                                metrics_dataframe.at[current_idx, 'stability_score'] = stability
                                metrics_dataframe.at[current_idx, 'anomaly_flag'] = anomaly
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

CARD_STYLE = {
    "background": f"linear-gradient(135deg, {THEME_COLORS['card']} 0%, {THEME_COLORS['card_hover']} 100%)",
    "padding": "28px",
    "borderRadius": "16px",
    "boxShadow": "0 6px 24px rgba(0, 0, 0, 0.35), 0 3px 12px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
    "border": f"1px solid {THEME_COLORS['border']}",
    "textAlign": "center",
    "minWidth": "160px",
    "transition": "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
    "position": "relative",
    "overflow": "hidden",
    "backdropFilter": "blur(12px)",
    "className": "card-hover-effect"
}

GRAPH_STYLE = {
    "borderRadius": "16px",
    "overflow": "hidden",
    "background": f"linear-gradient(135deg, {THEME_COLORS['surface']} 0%, {THEME_COLORS['card']} 100%)",
    "border": f"1px solid {THEME_COLORS['border']}",
    "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.4), 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
    "transition": "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
    "padding": "24px",
    "minHeight": "420px",
    "position": "relative",
    "className": "card-hover-effect"
}

# ================= DASH APPLICATION =================

app = dash.Dash(__name__)
app.title = "WiFi Network Performance Monitor"

# Premium CSS for smooth rendering
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
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
            @media (max-width: 1400px) {
                .chart-grid-3 { grid-template-columns: repeat(2, 1fr) !important; }
            }
            @media (max-width: 900px) {
                .chart-grid-3 { grid-template-columns: 1fr !important; }
                .chart-grid-2 { grid-template-columns: 1fr !important; }
            }
            .card-hover-effect {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .card-hover-effect:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), 0 4px 12px rgba(0, 0, 0, 0.4) !important;
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
    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
    "position": "relative",
    "overflowX": "hidden"
}, children=[
    
    html.Div(style={"maxWidth": "1800px", "margin": "0 auto", "padding": "40px 36px", "position": "relative"}, children=[
        
        # Title Section - Professional Header
        html.Div(style={
            "textAlign": "center", 
            "marginBottom": "56px", 
            "paddingTop": "40px", 
            "position": "relative",
            "borderBottom": f"2px solid {THEME_COLORS['border']}",
            "paddingBottom": "32px"
        }, children=[
            html.H1("WiFi Network Performance Monitor", style={
                "margin": "0 0 12px 0",
                "fontSize": "48px",
                "fontWeight": "800",
                "letterSpacing": "-1.5px",
                "background": f"linear-gradient(135deg, {THEME_COLORS['primary']} 0%, {THEME_COLORS['accent_light']} 100%)",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
                "backgroundClip": "text",
                "lineHeight": "1.2",
                "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
                "textShadow": f"0 4px 20px {THEME_COLORS['primary_glow']}"
            }),
            html.P("Real-time Network Analytics & Performance Metrics", style={
                "margin": "0",
                "fontSize": "16px",
                "color": THEME_COLORS["text_secondary"],
                "fontWeight": "400",
                "letterSpacing": "0.3px",
                "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
            })
        ]),
        
        # Connection Info Section
        html.Div(style={
            "background": f"linear-gradient(135deg, {THEME_COLORS['surface']} 0%, {THEME_COLORS['card']} 100%)",
            "padding": "36px 44px",
            "borderRadius": "16px",
            "marginBottom": "40px",
            "border": f"1px solid {THEME_COLORS['border']}",
            "boxShadow": f"0 10px 40px rgba(0, 0, 0, 0.45), 0 5px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
            "backdropFilter": "blur(24px)",
            "position": "relative"
        }, children=[
            html.Div(id="connection-info", style={
                "display": "flex",
                "justifyContent": "center",
                "flexWrap": "wrap",
                "gap": "16px",
                "alignItems": "center"
            })
        ]),

        # Auto-refresh interval (500ms for faster updates)
        dcc.Interval(
            id="update-interval",
            interval=1000,  # Optimized to 1s to prevent blocking and freezing
            n_intervals=0
        ),

        # KPI Section
        html.Div(style={
            "background": f"linear-gradient(135deg, {THEME_COLORS['surface']} 0%, {THEME_COLORS['card']} 100%)",
            "padding": "40px",
            "borderRadius": "16px",
            "marginBottom": "40px",
            "border": f"1px solid {THEME_COLORS['border']}",
            "boxShadow": f"0 10px 40px rgba(0, 0, 0, 0.45), 0 5px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
            "backdropFilter": "blur(24px)",
            "position": "relative"
        }, children=[
            html.H3("Key Performance Indicators", style={
                "margin": "0 0 28px 0",
                "fontSize": "22px",
                "color": THEME_COLORS["text_primary"],
                "fontWeight": "700",
                "letterSpacing": "0.5px",
                "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
                "textAlign": "center",
                "textTransform": "uppercase",
                "letterSpacing": "1px"
            }),
            html.Div(id="kpi-container", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                "gap": "20px"
            })
        ]),
        
        # Metrics Cards Container
        html.Div(style={
            "background": f"linear-gradient(135deg, {THEME_COLORS['surface']} 0%, {THEME_COLORS['card']} 100%)",
            "padding": "40px",
            "borderRadius": "16px",
            "marginBottom": "40px",
            "border": f"1px solid {THEME_COLORS['border']}",
            "boxShadow": f"0 10px 40px rgba(0, 0, 0, 0.45), 0 5px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
            "backdropFilter": "blur(24px)",
            "position": "relative"
        }, children=[
            html.H3("Current Metrics", style={
                "margin": "0 0 28px 0",
                "fontSize": "22px",
                "color": THEME_COLORS["text_primary"],
                "fontWeight": "700",
                "letterSpacing": "0.5px",
                "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
                "textAlign": "center",
                "textTransform": "uppercase",
                "letterSpacing": "1px"
            }),
            html.Div(id="metrics-container", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                "gap": "20px"
            })
        ]),

        # Research Insights Section
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "32px",
            "marginBottom": "40px"
        }, children=[
            # Link State Explanation Panel
            html.Div(style={
                "background": f"linear-gradient(135deg, {THEME_COLORS['surface']} 0%, {THEME_COLORS['card']} 100%)",
                "padding": "32px",
                "borderRadius": "16px",
                "border": f"1px solid {THEME_COLORS['border']}",
                "boxShadow": f"0 10px 40px rgba(0, 0, 0, 0.45), 0 5px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
                "backdropFilter": "blur(24px)"
            }, children=[
                html.H4("Link State Analysis", style={
                    "margin": "0 0 20px 0",
                    "fontSize": "18px",
                    "color": THEME_COLORS["text_primary"],
                    "fontWeight": "700",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
                }),
                html.Div(id="link-state-panel", style={
                    "color": THEME_COLORS["text_secondary"],
                    "fontSize": "14px",
                    "lineHeight": "1.8",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
                })
            ]),
            # Cross-Layer Insight Box
            html.Div(style={
                "background": f"linear-gradient(135deg, {THEME_COLORS['surface']} 0%, {THEME_COLORS['card']} 100%)",
                "padding": "32px",
                "borderRadius": "16px",
                "border": f"1px solid {THEME_COLORS['border']}",
                "boxShadow": f"0 10px 40px rgba(0, 0, 0, 0.45), 0 5px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
                "backdropFilter": "blur(24px)"
            }, children=[
                html.H4("Cross-Layer Insights", style={
                    "margin": "0 0 20px 0",
                    "fontSize": "18px",
                    "color": THEME_COLORS["text_primary"],
                    "fontWeight": "700",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
                }),
                html.Div(id="cross-layer-insight", style={
                    "color": THEME_COLORS["text_secondary"],
                    "fontSize": "14px",
                    "lineHeight": "1.8",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
                })
            ])
        ]),

        # Metric Freshness and Sampling Information
        html.Div(style={
            "background": f"linear-gradient(135deg, {THEME_COLORS['surface']} 0%, {THEME_COLORS['card']} 100%)",
            "padding": "24px 32px",
            "borderRadius": "16px",
            "marginBottom": "40px",
            "border": f"1px solid {THEME_COLORS['border']}",
            "boxShadow": f"0 10px 40px rgba(0, 0, 0, 0.45), 0 5px 20px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
            "backdropFilter": "blur(24px)"
        }, children=[
            html.Div(id="sampling-info", style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "16px",
                "color": THEME_COLORS["text_secondary"],
                "fontSize": "13px",
                "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
            })
        ]),

        # Charts Section - Refined Research-Oriented Charts
        html.Div(className="chart-grid-3", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "32px",
            "marginBottom": "48px"
        }, children=[
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE, "cursor": "help"},
                title="Signal Strength vs Time: Displays RSSI (dBm) and Signal percentage over time. Shows signal quality trends, mobility effects, and interference patterns.",
                children=[
                    dcc.Graph(id="chart1-signal-strength", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE, "cursor": "help"},
                title="RX vs TX PHY Rate: Dual-axis chart comparing receive and transmit physical data rates. Shows rate locking, fallback behavior, and asymmetry between RX/TX.",
                children=[
                    dcc.Graph(id="chart3-rx-tx-phy-rate", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE, "cursor": "help"},
                title="RTT Variability Analysis: Shows latency over time with moving average and confidence bands. Identifies latency spikes, jitter, and network stability issues.",
                children=[
                    dcc.Graph(id="chart4-rtt-variability", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE, "cursor": "help"},
                title="Link Stability Score: Gauge showing overall connection stability (0-100). Calculated from signal variance, throughput stability, and RTT consistency.",
                children=[
                    dcc.Graph(id="chart5-stability-gauge", config={"displayModeBar": False, "responsive": True})
                ]
            ),
            html.Div(
                className="card-hover-effect",
                style={**GRAPH_STYLE, "cursor": "help"},
                title="Anomaly Timeline: Binary timeline showing when abnormal network behavior occurred. Flags signal drops, high latency, or throughput degradation.",
                children=[
                    dcc.Graph(id="chart6-anomaly-timeline", config={"displayModeBar": False, "responsive": True})
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
    Output("link-state-panel", "children"),
    Output("cross-layer-insight", "children"),
    Output("sampling-info", "children"),
    Output("chart1-signal-strength", "figure"),
    Output("chart3-rx-tx-phy-rate", "figure"),
    Output("chart4-rtt-variability", "figure"),
    Output("chart5-stability-gauge", "figure"),
    Output("chart6-anomaly-timeline", "figure"),
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
        return [], [], [], "Collecting data...", "Collecting data...", "Initializing...", empty_layout, empty_layout, empty_layout, empty_layout, empty_layout
    
    # Enterprise connection info badges - Premium Design
    conn_badges = [
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
                "padding": "12px 20px",
                "background": f"linear-gradient(135deg, {THEME_COLORS['card']} 0%, {THEME_COLORS['card_hover']} 100%)",
                "borderRadius": "10px",
                "border": f"1px solid {THEME_COLORS['border']}",
                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
                "transition": "all 0.3s ease",
                "cursor": "help"
            },
            title="SSID (Service Set Identifier): The name of the WiFi network you are connected to.",
            children=[
                html.Span("SSID", style={"fontWeight": "600", "color": THEME_COLORS["text_secondary"], "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.5px", "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"}),
                html.Span("•", style={"color": THEME_COLORS["border"], "fontSize": "10px", "margin": "0 4px"}),
                html.Span(conn_info["ssid"], style={"color": THEME_COLORS["text_primary"], "fontWeight": "500", "fontSize": "13px", "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"})
            ]
        ),
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
                "padding": "12px 20px",
                "background": f"linear-gradient(135deg, {THEME_COLORS['card']} 0%, {THEME_COLORS['card_hover']} 100%)",
                "borderRadius": "10px",
                "border": f"1px solid {THEME_COLORS['border']}",
                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
                "transition": "all 0.3s ease",
                "cursor": "help"
            },
            title="Radio Type: The WiFi standard being used (e.g., 802.11ac, 802.11ax/WiFi 6). Determines maximum speed and features.",
            children=[
                html.Span("Radio", style={"fontWeight": "600", "color": THEME_COLORS["text_secondary"], "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.5px", "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"}),
                html.Span("•", style={"color": THEME_COLORS["border"], "fontSize": "10px", "margin": "0 4px"}),
                html.Span(conn_info["radio"], style={"color": THEME_COLORS["text_primary"], "fontWeight": "500", "fontSize": "13px", "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"})
            ]
        ),
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
                "padding": "12px 20px",
                "background": f"linear-gradient(135deg, {THEME_COLORS['card']} 0%, {THEME_COLORS['card_hover']} 100%)",
                "borderRadius": "10px",
                "border": f"1px solid {THEME_COLORS['border']}",
                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
                "transition": "all 0.3s ease",
                "cursor": "help"
            },
            title="Channel: The WiFi frequency channel number. Different channels help avoid interference from other networks.",
            children=[
                html.Span("Channel", style={"fontWeight": "600", "color": THEME_COLORS["text_secondary"], "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.5px", "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"}),
                html.Span("•", style={"color": THEME_COLORS["border"], "fontSize": "10px", "margin": "0 4px"}),
                html.Span(conn_info["channel"], style={"color": THEME_COLORS["text_primary"], "fontWeight": "500", "fontSize": "13px", "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"})
            ]
        ),
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
                "padding": "12px 20px",
                "background": f"linear-gradient(135deg, {THEME_COLORS['card']} 0%, {THEME_COLORS['card_hover']} 100%)",
                "borderRadius": "10px",
                "border": f"1px solid {THEME_COLORS['border']}",
                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
                "transition": "all 0.3s ease",
                "cursor": "help"
            },
            title="BSSID (Basic Service Set Identifier): The MAC address of the access point you are connected to. Unique identifier for the access point.",
            children=[
                html.Span("BSSID", style={"fontWeight": "600", "color": THEME_COLORS["text_secondary"], "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.5px", "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"}),
                html.Span("•", style={"color": THEME_COLORS["border"], "fontSize": "10px", "margin": "0 4px"}),
                html.Span(conn_info.get("bssid", "N/A"), style={"color": THEME_COLORS["text_primary"], "fontWeight": "500", "fontSize": "11px", "fontFamily": "'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace"})
            ]
        )
    ]
    
    if df.empty:
        empty_layout = create_empty_figure_layout()
        sampling_text = f"Sampling: {SAMPLE_INTERVAL_SECONDS}s interval • Data Sources: Linux iw, iwconfig, ping • Interface: {INTERFACE}"
        return conn_badges, [], [], html.Div("Collecting data..."), html.Div("Collecting data..."), sampling_text, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout

    try:
        latest_metrics = df.iloc[-1]
    except (IndexError, KeyError):
        empty_layout = create_empty_figure_layout()
        sampling_text = f"Sampling: {SAMPLE_INTERVAL_SECONDS}s interval • Data Sources: Linux iw, iwconfig, ping • Interface: {INTERFACE}"
        return conn_badges, [], [], html.Div("No data available"), html.Div("No data available"), sampling_text, empty_layout, empty_layout, empty_layout, empty_layout, empty_layout
    
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
    
    # Generate research insights
    link_state_text = generate_link_state_explanation(df, latest_metrics)
    cross_layer_text = generate_cross_layer_insight(df, latest_metrics)
    
    # Generate sampling information
    total_samples = len(df)
    time_span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) > 1 else 0
    sampling_text = f"Sampling: {SAMPLE_INTERVAL_SECONDS}s interval • Data Sources: Linux iw, iwconfig, ping • Interface: {INTERFACE} • Samples: {total_samples} • Duration: {time_span:.1f}s"
    
    # Generate refined charts - with error handling
    try:
        chart1 = create_chart1_signal_strength(df)
        chart3 = create_chart3_rx_tx_phy_rate(df)
        chart4 = create_chart4_rtt_variability(df)
        chart5 = create_chart5_stability_gauge(df)
        chart6 = create_chart6_anomaly_timeline(df)
    except Exception as chart_error:
        # If chart generation fails, return empty layouts
        empty_layout = create_empty_figure_layout()
        chart1 = chart3 = chart4 = chart5 = chart6 = empty_layout
    
    return conn_badges, kpi_cards, metric_cards, link_state_text, cross_layer_text, sampling_text, chart1, chart3, chart4, chart5, chart6

# ================= CHART GENERATORS =================

def create_base_layout(title: str) -> dict:
    """Creates enterprise-grade base layout configuration for charts."""
    return {
        "title": {
            "text": title,
            "font": {"size": 17, "color": THEME_COLORS["text_primary"], "family": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif", "weight": "700"},
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
            "yanchor": "top",
            "pad": {"t": 12, "b": 12}
        },
        "paper_bgcolor": "transparent",
        "plot_bgcolor": THEME_COLORS["card"],
        "font": {"color": THEME_COLORS["text_primary"], "family": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif", "size": 13},
        "xaxis": {
            "gridcolor": THEME_COLORS["grid"],
            "showgrid": True,
            "zeroline": False,
            "color": THEME_COLORS["text_secondary"],
            "tickformat": "%H:%M:%S",
            "linecolor": THEME_COLORS["border"],
            "linewidth": 1
        },
        "yaxis": {
            "gridcolor": THEME_COLORS["grid"],
            "showgrid": True,
            "zeroline": False,
            "color": THEME_COLORS["text_secondary"],
            "linecolor": THEME_COLORS["border"],
            "linewidth": 1
        },
        "margin": {"l": 70, "r": 50, "t": 90, "b": 70},
        "hovermode": "x unified",
        "showlegend": True,
        "legend": {
            "bgcolor": "transparent",
            "bordercolor": THEME_COLORS["border"],
            "borderwidth": 1,
            "font": {"size": 11, "color": THEME_COLORS["text_secondary"]},
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top"
        },
        "autosize": True
    }

def create_empty_figure_layout() -> dict:
    """Creates empty figure for when no data is available."""
    return {
        "data": [],
        "layout": {
            "paper_bgcolor": THEME_COLORS["surface"],
            "plot_bgcolor": THEME_COLORS["card"],
            "font": {"color": THEME_COLORS["text_primary"]},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": "Collecting data...",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16, "color": THEME_COLORS["text_secondary"]}
            }]
        }
    }

def create_metric_cards(latest: pd.Series) -> list:
    """
    Generates metric display cards from latest measurements.
    
    Args:
        latest: Series containing the most recent metric values
        
    Returns:
        List of HTML Div elements representing metric cards
    """
    metrics = [
        {
            "label": "Signal",
            "value": f"{latest['signal_percent']:.0f}%" if pd.notna(latest['signal_percent']) else "N/A",
            "color": THEME_COLORS["success"],
            "tooltip": "Signal Strength: Percentage of WiFi signal quality (0-100%). Higher values indicate better connection quality."
        },
        {
            "label": "RSSI",
            "value": f"{latest['rssi_dbm']:.0f} dBm" if pd.notna(latest['rssi_dbm']) else "N/A",
            "color": THEME_COLORS["primary"],
            "tooltip": "RSSI (Received Signal Strength Indicator): Signal power in decibels. Values closer to 0 indicate stronger signal (typically -30 to -90 dBm)."
        },
        {
            "label": "RX Rate",
            "value": f"{latest['rx_mbps']:.1f} Mbps" if pd.notna(latest['rx_mbps']) else "N/A",
            "color": THEME_COLORS["primary"],
            "tooltip": "Receive Rate: Physical data rate for receiving data from the access point, measured in Megabits per second (Mbps)."
        },
        {
            "label": "TX Rate",
            "value": f"{latest['tx_mbps']:.1f} Mbps" if pd.notna(latest['tx_mbps']) else "N/A",
            "color": THEME_COLORS["warning"],
            "tooltip": "Transmit Rate: Physical data rate for sending data to the access point, measured in Megabits per second (Mbps)."
        },
        {
            "label": "Channel",
            "value": f"{int(latest['channel'])}" if pd.notna(latest['channel']) else "N/A",
            "color": THEME_COLORS["primary"],
            "tooltip": "WiFi Channel: The radio frequency channel number your WiFi connection is using. Different channels help avoid interference."
        }
    ]
    
    return [
        html.Div(
            style={**CARD_STYLE, "cursor": "help"},
            title=metric.get("tooltip", ""),
            children=[
                html.Div(metric["label"], style={
                    "fontSize": "11px",
                    "color": THEME_COLORS["text_secondary"],
                    "marginBottom": "12px",
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
                }),
                html.Div(metric["value"], style={
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "color": metric["color"],
                    "letterSpacing": "-0.5px",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
                    "textShadow": f"0 2px 8px {metric['color']}40"
                })
            ]
        ) for metric in metrics
    ]

def generate_link_state_explanation(df: pd.DataFrame, latest: pd.Series) -> str:
    """
    Generates rule-based link state explanation.
    
    Args:
        df: DataFrame with metric history
        latest: Latest metric values
        
    Returns:
        Text explanation of current link state
    """
    if df.empty or len(df) < 10:
        return "Insufficient data for state analysis. Collecting samples..."
    
    # Analyze recent window (last 20 samples)
    recent = df.tail(20)
    
    # Signal analysis
    signal_vals = recent['signal_percent'].dropna()
    signal_mean = signal_vals.mean() if len(signal_vals) > 0 else None
    signal_std = signal_vals.std() if len(signal_vals) > 0 else None
    signal_trend = "stable" if signal_std and signal_std < 5 else "variable" if signal_std and signal_std < 15 else "unstable"
    
    # RTT analysis
    rtt_vals = recent['rtt_ms'].dropna()
    rtt_mean = rtt_vals.mean() if len(rtt_vals) > 0 else None
    rtt_std = rtt_vals.std() if len(rtt_vals) > 0 else None
    rtt_trend = "stable" if rtt_std and rtt_std < 5 else "variable" if rtt_std and rtt_std < 15 else "unstable"
    
    # PHY rate analysis
    rx_vals = recent['rx_mbps'].dropna()
    rx_std = rx_vals.std() if len(rx_vals) > 0 else None
    phy_trend = "stable" if rx_std and rx_std < 10 else "variable" if rx_std and rx_std < 50 else "unstable"
    
    # Determine overall state with proper None handling
    signal_str = f"{signal_mean:.0f}%" if signal_mean is not None else "N/A"
    rtt_str = f"{rtt_mean:.1f}ms" if rtt_mean is not None else "N/A"
    
    if signal_mean is not None and signal_mean >= 70 and signal_trend == "stable" and rtt_mean is not None and rtt_mean < 50 and rtt_trend == "stable":
        state = "STABLE"
        state_color = THEME_COLORS["success"]
        explanation = f"Link is stable. Signal: {signal_str} ({signal_trend}), RTT: {rtt_str} ({rtt_trend}), PHY rates stable."
    elif signal_mean is not None and signal_mean >= 50 and (signal_trend in ["stable", "variable"] or rtt_trend in ["stable", "variable"]):
        state = "DEGRADING"
        state_color = THEME_COLORS["warning"]
        explanation = f"Link shows degradation signs. Signal: {signal_str} ({signal_trend}), RTT: {rtt_str} ({rtt_trend}), PHY rates {phy_trend}."
    else:
        state = "UNSTABLE"
        state_color = THEME_COLORS["error"]
        explanation = f"Link is unstable. Signal: {signal_str} ({signal_trend}) if available, RTT: {rtt_str} ({rtt_trend}) if available, high variability detected."
    
    return [
        html.Strong(f"State: {state}", style={"color": state_color}),
        html.Br(),
        explanation
    ]

def generate_cross_layer_insight(df: pd.DataFrame, latest: pd.Series):
    """
    Generates cross-layer insight highlighting PHY-network layer relationships.
    
    Args:
        df: DataFrame with metric history
        latest: Latest metric values
        
    Returns:
        HTML elements or text insight about cross-layer behavior
    """
    if df.empty or len(df) < 10:
        return html.Div("Insufficient data for cross-layer analysis. Collecting samples...")
    
    recent = df.tail(20)
    
    # PHY layer stability
    rx_vals = recent['rx_mbps'].dropna()
    tx_vals = recent['tx_mbps'].dropna()
    phy_stable = False
    if len(rx_vals) > 0 and len(tx_vals) > 0:
        rx_std = rx_vals.std()
        tx_std = tx_vals.std()
        phy_stable = (rx_std < 20 and tx_std < 20)
    
    # Network layer (RTT) behavior
    rtt_vals = recent['rtt_ms'].dropna()
    rtt_mean = rtt_vals.mean() if len(rtt_vals) > 0 else None
    rtt_std = rtt_vals.std() if len(rtt_vals) > 0 else None
    rtt_high = rtt_mean and rtt_mean > 50
    rtt_variable = rtt_std and rtt_std > 10
    
    # Generate insight with proper None handling
    rtt_mean_str = f"{rtt_mean:.1f}ms" if rtt_mean is not None else "N/A"
    rtt_std_str = f"{rtt_std:.1f}ms" if rtt_std is not None else "N/A"
    
    if phy_stable and rtt_high:
        return [
            html.Strong("PHY-Network Mismatch: "),
            f"PHY rates are stable (RX/TX variance < 20 Mbps), but RTT is elevated ({rtt_mean_str}). This suggests network-layer congestion or upstream issues, not PHY-layer problems."
        ]
    elif phy_stable and rtt_variable:
        return [
            html.Strong("PHY-Network Mismatch: "),
            f"PHY rates are stable, but RTT shows high variability (std: {rtt_std_str}). This indicates network-layer jitter independent of PHY stability."
        ]
    elif not phy_stable and rtt_high:
        return [
            html.Strong("Cross-Layer Correlation: "),
            f"Both PHY rates and RTT ({rtt_mean_str}) show degradation. This suggests PHY-layer issues (signal quality, interference) are affecting network performance."
        ]
    else:
        return [
            html.Strong("Normal Operation: "),
            f"PHY and network layers are consistent. RTT: {rtt_mean_str} if available, PHY rates within expected variance."
        ]

def create_kpi_cards(df: pd.DataFrame) -> list:
    """
    Generates KPI summary cards from dataframe statistics.
    
    Args:
        df: DataFrame containing metric history
        
    Returns:
        List of HTML Div elements representing KPI cards
    """
    # Calculate KPIs from available data
    avg_signal = df['signal_percent'].dropna().mean() if len(df['signal_percent'].dropna()) > 0 else 0
    avg_rssi = df['rssi_dbm'].dropna().mean() if len(df['rssi_dbm'].dropna()) > 0 else 0
    
    rx_dropped = df['rx_mbps'].dropna()
    tx_dropped = df['tx_mbps'].dropna()
    max_throughput = max(rx_dropped.max() if len(rx_dropped) > 0 else 0, tx_dropped.max() if len(tx_dropped) > 0 else 0)
    
    avg_link_speed = df['link_speed_mbps'].dropna().mean() if len(df['link_speed_mbps'].dropna()) > 0 else 0
    
    # Signal quality interpretation
    if avg_signal >= 80:
        signal_status = "Excellent"
        signal_color = THEME_COLORS["success"]
    elif avg_signal >= 60:
        signal_status = "Good"
        signal_color = THEME_COLORS["primary"]
    elif avg_signal >= 40:
        signal_status = "Fair"
        signal_color = THEME_COLORS["warning"]
    else:
        signal_status = "Poor"
        signal_color = THEME_COLORS["error"]
    
    kpis = [
        {
            "label": "Avg Signal",
            "value": f"{avg_signal:.0f}%",
            "status": signal_status,
            "color": signal_color,
            "tooltip": "Average Signal Strength: Mean signal quality percentage over the monitoring period. Higher values indicate more stable connection."
        },
        {
            "label": "Avg RSSI",
            "value": f"{avg_rssi:.0f} dBm",
            "status": "Signal Power",
            "color": THEME_COLORS["primary"],
            "tooltip": "Average RSSI: Mean received signal strength indicator in decibels. Shows overall signal power level over time."
        }
    ]
    
    return [
        html.Div(
            style={
                **CARD_STYLE,
                "background": f"linear-gradient(135deg, {THEME_COLORS['card']} 0%, #1e2a3a 100%)",
                "borderLeft": f"4px solid {kpi['color']}",
                "boxShadow": f"0 4px 16px rgba(0, 0, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 0 20px {kpi['color']}20",
                "cursor": "help"
            },
            title=kpi.get("tooltip", ""),
            children=[
                html.Div(kpi["label"], style={
                    "fontSize": "11px",
                    "color": THEME_COLORS["text_secondary"],
                    "marginBottom": "12px",
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
                }),
                html.Div(kpi["value"], style={
                    "fontSize": "32px",
                    "fontWeight": "700",
                    "color": kpi["color"],
                    "marginBottom": "8px",
                    "letterSpacing": "-0.5px",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
                    "textShadow": f"0 2px 8px {kpi['color']}40"
                }),
                html.Div(kpi["status"], style={
                    "fontSize": "13px",
                    "color": kpi["color"],
                    "fontWeight": "500",
                    "letterSpacing": "0.5px",
                    "fontFamily": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"
                })
            ]
        ) for kpi in kpis
    ]

# ================= EVENT DETECTION =================

def detect_events(df: pd.DataFrame) -> List[Dict]:
    """
    Detects significant events for annotation on charts.
    
    Args:
        df: DataFrame with metrics
        
    Returns:
        List of event dictionaries with timestamp, type, and description
    """
    events = []
    if df.empty or len(df) < 5:
        return events
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    
    # Detect signal drops (>20% drop)
    signal_vals = df["signal_percent"].dropna()
    if len(signal_vals) > 5:
        for i in range(5, len(df)):
            if pd.notna(df["signal_percent"].iloc[i]) and pd.notna(df["signal_percent"].iloc[i-5]):
                prev_avg = signal_vals.iloc[max(0, i-5):i].mean()
                current = df["signal_percent"].iloc[i]
                if prev_avg > 0 and current < prev_avg - 20:
                    events.append({
                        "timestamp": timestamps[i],
                        "type": "signal_drop",
                        "description": f"Signal drop: {prev_avg:.0f}% → {current:.0f}%"
                    })
    
    # Detect RTT spikes (>100ms or >2x average)
    rtt_vals = df["rtt_ms"].dropna()
    if len(rtt_vals) > 5:
        rtt_mean = rtt_vals.mean()
        for i in range(len(df)):
            if pd.notna(df["rtt_ms"].iloc[i]):
                rtt = df["rtt_ms"].iloc[i]
                if rtt > 100 or (rtt_mean > 0 and rtt > rtt_mean * 2):
                    events.append({
                        "timestamp": timestamps[i],
                        "type": "rtt_spike",
                        "description": f"RTT spike: {rtt:.1f}ms"
                    })
    
    # Detect PHY rate fallback (>50% drop)
    rx_vals = df["rx_mbps"].dropna()
    if len(rx_vals) > 5:
        for i in range(5, len(df)):
            if pd.notna(df["rx_mbps"].iloc[i]) and pd.notna(df["rx_mbps"].iloc[i-5]):
                prev_avg = rx_vals.iloc[max(0, i-5):i].mean()
                current = df["rx_mbps"].iloc[i]
                if prev_avg > 0 and current < prev_avg * 0.5:
                    events.append({
                        "timestamp": timestamps[i],
                        "type": "phy_fallback",
                        "description": f"PHY fallback: {prev_avg:.1f} → {current:.1f} Mbps"
                    })
    
    return events

# ================= CHART 1: SIGNAL STRENGTH VS TIME =================

def create_chart1_signal_strength(df: pd.DataFrame) -> dict:
    """
    Chart 1: Signal Strength vs Time (Line Chart)
    Shows RSSI (dBm) and Signal % over time to visualize mobility, fading, and interference.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    
    # Detect events for annotation
    events = detect_events(df)
    signal_events = [e for e in events if e["type"] == "signal_drop"]
    
    # Create annotations for signal drop events
    annotations = []
    for event in signal_events:
        # Find closest timestamp index
        event_ts = event["timestamp"].timestamp()
        closest_idx = min(range(len(df)), key=lambda i: abs(df["timestamp"].iloc[i] - event_ts))
        if closest_idx < len(df) and pd.notna(df["signal_percent"].iloc[closest_idx]):
            signal_val = df["signal_percent"].iloc[closest_idx]
            annotations.append({
                "x": event["timestamp"],
                "y": signal_val,
                "text": "⚠ Signal Drop",
                "showarrow": True,
                "arrowhead": 2,
                "arrowcolor": THEME_COLORS["error"],
                "arrowsize": 1.5,
                "ax": 0,
                "ay": -30,
                "bgcolor": THEME_COLORS["error"],
                "bordercolor": THEME_COLORS["error"],
                "font": {"color": "white", "size": 10}
            })
    
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
            **create_base_layout("Signal Strength vs Time"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "RSSI (dBm)",
                "side": "left",
                "color": THEME_COLORS["primary"]
            },
            "yaxis2": {
                "title": "Signal (%)",
                "overlaying": "y",
                "side": "right",
                "range": [0, 100],
                "color": THEME_COLORS["success"]
            },
            "annotations": annotations
        }
    }

# ================= CHART 2: RTT DISTRIBUTION =================

def create_chart2_rtt_distribution(df: pd.DataFrame) -> dict:
    """
    Chart 2: RTT (Latency) Distribution (Histogram)
    Shows latency spread to identify tail latency and jitter.
    """
    if df.empty:
        return create_empty_figure_layout()
    
    rtt_vals = df["rtt_ms"].dropna()
    
    if len(rtt_vals) == 0:
        return create_empty_figure_layout()
    
    return {
        "data": [
            go.Histogram(
                x=rtt_vals,
                name="RTT Distribution",
                marker={"color": THEME_COLORS["warning"], "opacity": 0.8, "line": {"color": THEME_COLORS["border"], "width": 1}},
                nbinsx=30,
                hovertemplate="<b>RTT</b><br>%{x:.1f} ms<br>Count: %{y}<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("RTT (Latency) Distribution"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "Frequency"
            },
            "xaxis": {
                **create_base_layout("")["xaxis"],
                "title": "RTT (ms)",
                "tickformat": ""
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
    
    # Detect events for annotation
    events = detect_events(df)
    phy_events = [e for e in events if e["type"] == "phy_fallback"]
    
    # Create annotations for PHY fallback events
    annotations = []
    for event in phy_events:
        # Find closest timestamp index
        event_ts = event["timestamp"].timestamp()
        closest_idx = min(range(len(df)), key=lambda i: abs(df["timestamp"].iloc[i] - event_ts))
        if closest_idx < len(df) and pd.notna(df["rx_mbps"].iloc[closest_idx]):
            rx_val = df["rx_mbps"].iloc[closest_idx]
            annotations.append({
                "x": event["timestamp"],
                "y": rx_val,
                "text": "⚠ PHY Fallback",
                "showarrow": True,
                "arrowhead": 2,
                "arrowcolor": THEME_COLORS["warning"],
                "arrowsize": 1.5,
                "ax": 0,
                "ay": -30,
                "bgcolor": THEME_COLORS["warning"],
                "bordercolor": THEME_COLORS["warning"],
                "font": {"color": "white", "size": 10}
            })
    
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
            **create_base_layout("RX vs TX PHY Rate"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "RX Rate (Mbps)",
                "side": "left",
                "color": THEME_COLORS["primary"],
                "range": rx_axis_range,
                "showgrid": True,
                "gridcolor": f"{THEME_COLORS['primary']}40"
            },
            "yaxis2": {
                "title": "TX Rate (Mbps)",
                "overlaying": "y",
                "side": "right",
                "color": THEME_COLORS["warning"],
                "range": tx_axis_range,
                "showgrid": False  # Disable grid for right axis to reduce visual clutter
            },
            "annotations": annotations
        }
    }

# ================= CHART 4: RTT VARIABILITY =================

def create_chart4_rtt_variability(df: pd.DataFrame) -> dict:
    """
    Chart 4: RTT Variability with Moving Average
    Professional visualization showing RTT over time with moving average and statistical bands.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    rtt_vals = df["rtt_ms"].dropna()
    
    if len(rtt_vals) == 0:
        return create_empty_figure_layout()
    
    # Get corresponding timestamps for non-null RTT values
    valid_indices = df["rtt_ms"].notna()
    valid_timestamps = [timestamps[i] for i in range(len(timestamps)) if valid_indices.iloc[i]]
    
    # Convert rtt_vals to list for consistency
    rtt_list = rtt_vals.tolist()
    
    # Calculate moving average (rolling window of 10 samples)
    window_size = min(10, len(rtt_vals))
    if window_size > 1:
        moving_avg_series = rtt_vals.rolling(window=window_size, center=True).mean()
        moving_std_series = rtt_vals.rolling(window=window_size, center=True).std()
        
        # Fill NaN values at edges with the first/last valid value
        moving_avg_series = moving_avg_series.bfill().ffill()
        moving_std_series = moving_std_series.fillna(0)
        
        # Convert to lists
        moving_avg = moving_avg_series.tolist()
        moving_std = moving_std_series.tolist()
        
        # Create upper and lower bounds (mean ± 1.5 * std)
        upper_bound = [avg + 1.5 * std if pd.notna(avg) and pd.notna(std) else (avg if pd.notna(avg) else 0) for avg, std in zip(moving_avg, moving_std)]
        lower_bound = [avg - 1.5 * std if pd.notna(avg) and pd.notna(std) else (avg if pd.notna(avg) else 0) for avg, std in zip(moving_avg, moving_std)]
    else:
        moving_avg = rtt_list
        upper_bound = rtt_list
        lower_bound = rtt_list
    
    # Calculate overall statistics for reference lines
    if len(rtt_vals) > 0:
        mean_rtt = rtt_vals.mean()
        if pd.isna(mean_rtt):
            mean_rtt = 0.0
    else:
        mean_rtt = 0.0
    
    # Detect events for annotation
    events = detect_events(df)
    rtt_events = [e for e in events if e["type"] == "rtt_spike"]
    
    # Create annotations for RTT spike events
    annotations = []
    for event in rtt_events:
        # Find closest timestamp index in valid_timestamps
        event_ts = event["timestamp"]
        closest_idx = min(range(len(valid_timestamps)), key=lambda i: abs((valid_timestamps[i] - event_ts).total_seconds())) if len(valid_timestamps) > 0 else None
        if closest_idx is not None and closest_idx < len(rtt_list):
            rtt_val = rtt_list[closest_idx]
            if abs((valid_timestamps[closest_idx] - event_ts).total_seconds()) < SAMPLE_INTERVAL_SECONDS * 2:
                annotations.append({
                    "x": event["timestamp"],
                    "y": rtt_val,
                    "text": "⚠ RTT Spike",
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowcolor": THEME_COLORS["error"],
                    "arrowsize": 1.5,
                    "ax": 0,
                    "ay": -30,
                    "bgcolor": THEME_COLORS["error"],
                    "bordercolor": THEME_COLORS["error"],
                    "font": {"color": "white", "size": 10}
                })
    
    return {
        "data": [
            # Confidence band (upper bound)
            go.Scatter(
                x=valid_timestamps,
                y=upper_bound,
                mode="lines",
                name="Upper Bound",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip"
            ),
            # Confidence band (lower bound)
            go.Scatter(
                x=valid_timestamps,
                y=lower_bound,
                mode="lines",
                name="Confidence Band",
                fill="tonexty",
                fillcolor=f"rgba(248, 81, 73, 0.15)",
                line={"width": 0},
                showlegend=True,
                hovertemplate="<b>Upper Bound</b><br>%{y:.1f} ms<extra></extra>"
            ),
            # Moving average line
            go.Scatter(
                x=valid_timestamps,
                y=moving_avg,
                mode="lines",
                name="Moving Average",
                line={"color": THEME_COLORS["primary"], "width": 2.5, "dash": "dash"},
                hovertemplate="<b>Moving Avg</b><br>%{y:.1f} ms<br>%{x}<extra></extra>"
            ),
            # Actual RTT values
            go.Scatter(
                x=valid_timestamps,
                y=rtt_list,
                mode="lines+markers",
                name="RTT",
                line={"color": THEME_COLORS["warning"], "width": 2},
                marker={"size": 3, "color": THEME_COLORS["warning"], "opacity": 0.6},
                hovertemplate="<b>RTT</b><br>%{y:.1f} ms<br>%{x}<extra></extra>"
            ),
            # Mean reference line
            go.Scatter(
                x=[valid_timestamps[0], valid_timestamps[-1]] if len(valid_timestamps) > 0 else [],
                y=[mean_rtt, mean_rtt],
                mode="lines",
                name="Mean",
                line={"color": THEME_COLORS["success"], "width": 1.5, "dash": "dot"},
                hovertemplate=f"<b>Mean</b><br>{mean_rtt:.1f} ms<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("RTT Variability Analysis"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "RTT (ms)"
            },
            "xaxis": {
                **create_base_layout("")["xaxis"],
                "title": "Time"
            },
            "annotations": annotations
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
            "title": {
                "text": "Link Stability Score",
                "font": {"size": 17, "color": THEME_COLORS["text_primary"], "family": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif", "weight": "700"},
                "x": 0.5,
                "xanchor": "center",
                "y": 0.97,
                "yanchor": "top",
                "pad": {"t": 12, "b": 12}
            },
            "paper_bgcolor": THEME_COLORS["surface"],
            "plot_bgcolor": THEME_COLORS["card"],
            "font": {"color": THEME_COLORS["text_primary"], "family": "'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif"},
            "margin": {"l": 30, "r": 30, "t": 90, "b": 30},
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
            **create_base_layout("Anomaly Timeline"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "Anomaly Flag",
                "range": [-0.1, 1.1],
                "tickmode": "linear",
                "tick0": 0,
                "dtick": 1,
                "tickvals": [0, 1],
                "ticktext": ["Normal", "Anomaly"]
            }
        }
    }

# ================= CHART 7: BANDWIDTH UTILIZATION =================

def create_chart7_bandwidth_utilization(df: pd.DataFrame) -> dict:
    """
    Chart 7: Bandwidth Utilization Over Time
    Shows bandwidth utilization percentage over time to track network capacity usage.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    bw_util = df["bandwidth_util"].fillna(0)
    
    return {
        "data": [
            go.Scatter(
                x=timestamps,
                y=bw_util,
                mode="lines+markers",
                name="Bandwidth Utilization",
                line={"color": THEME_COLORS["warning"], "width": 2.5},
                marker={"size": 4, "color": THEME_COLORS["warning"]},
                fill="tozeroy",
                fillcolor=f"rgba(210, 153, 34, 0.2)",
                hovertemplate="<b>Bandwidth Util</b><br>%{y:.1f}%<br>%{x}<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("Bandwidth Utilization Over Time"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "Utilization (%)",
                "range": [0, 100]
            }
        }
    }

# ================= CHART 8: LINK SPEED TREND =================

def create_chart8_link_speed_trend(df: pd.DataFrame) -> dict:
    """
    Chart 8: Link Speed Trend Over Time
    Shows link speed (Mbps) trend to visualize connection quality changes.
    """
    if df.empty or len(df) == 0:
        return create_empty_figure_layout()
    
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]
    link_speed = df["link_speed_mbps"].dropna()
    
    if len(link_speed) == 0:
        return create_empty_figure_layout()
    
    # Get corresponding timestamps for non-null values
    valid_timestamps = [timestamps[i] for i in range(len(timestamps)) if pd.notna(df["link_speed_mbps"].iloc[i])]
    
    return {
        "data": [
            go.Scatter(
                x=valid_timestamps,
                y=link_speed,
                mode="lines+markers",
                name="Link Speed",
                line={"color": THEME_COLORS["success"], "width": 2.5},
                marker={"size": 4, "color": THEME_COLORS["success"]},
                hovertemplate="<b>Link Speed</b><br>%{y:.1f} Mbps<br>%{x}<extra></extra>"
            )
        ],
        "layout": {
            **create_base_layout("Link Speed Trend"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "Link Speed (Mbps)"
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
            debug=False
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting dashboard: {e}")

if __name__ == "__main__":
    main()