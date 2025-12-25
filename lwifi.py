import subprocess
import time
import threading
import re
from datetime import datetime
from typing import Tuple, Optional
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ================= CONFIGURATION =================
PING_TARGET = "8.8.8.8"
SAMPLE_INTERVAL_SECONDS = 1
MAX_DATA_POINTS = 300
DASHBOARD_PORT = 8050
DASHBOARD_HOST = "127.0.0.1"
# ================================================

# Thread-safe data storage
data_lock = threading.Lock()
# Initialize DataFrame with explicit dtypes to avoid FutureWarning
metrics_dataframe = pd.DataFrame({
    "timestamp": pd.Series(dtype="float64"),
    "signal_percent": pd.Series(dtype="Int64"),
    "rssi_dbm": pd.Series(dtype="Int64"),
    "rx_mbps": pd.Series(dtype="float64"),
    "tx_mbps": pd.Series(dtype="float64"),
    "snr_db": pd.Series(dtype="float64"),
    "link_speed_mbps": pd.Series(dtype="float64"),
    "channel": pd.Series(dtype="Int64"),
    "bandwidth_util": pd.Series(dtype="float64")
})

# ================= DATA COLLECTION =================

def collect_wifi_metrics_windows() -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], Optional[float], Optional[int], Optional[str], Optional[str], Optional[str]]:
    """
    Collects Wi-Fi interface metrics from Windows system using netsh.
    
    Returns:
        Tuple containing signal percentage, RSSI (dBm), RX rate (Mbps), TX rate (Mbps), SNR (dB), Link Speed (Mbps), Channel, SSID, BSSID, Radio type.
        Returns None for any metric that cannot be retrieved.
    """
    try:
        output = subprocess.check_output(
            ["netsh", "wlan", "show", "interfaces"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5
        )

        signal_match = re.search(r"Signal\s+:\s+(\d+)%", output)
        rssi_match = re.search(r"Rssi\s+:\s+(-?\d+)", output)
        rx_match = re.search(r"Receive rate \(Mbps\)\s+:\s+([\d\.]+)", output)
        tx_match = re.search(r"Transmit rate \(Mbps\)\s+:\s+([\d\.]+)", output)
        snr_match = re.search(r"Signal/Noise Ratio\s+:\s+(\d+)", output)
        channel_match = re.search(r"Channel\s+:\s+(\d+)", output)
        ssid_match = re.search(r"SSID\s+:\s+(.+)", output)
        bssid_match = re.search(r"BSSID\s+:\s+([0-9A-Fa-f:]+)", output)
        radio_match = re.search(r"Radio type\s+:\s+(.+)", output)

        # Calculate average link speed
        rx_val = float(rx_match.group(1)) if rx_match else 0
        tx_val = float(tx_match.group(1)) if tx_match else 0
        link_speed = (rx_val + tx_val) / 2 if (rx_val > 0 and tx_val > 0) else (rx_val or tx_val)

        return (
            int(signal_match.group(1)) if signal_match else None,
            int(rssi_match.group(1)) if rssi_match else None,
            float(rx_match.group(1)) if rx_match else None,
            float(tx_match.group(1)) if tx_match else None,
            int(snr_match.group(1)) if snr_match else 0,
            link_speed,
            int(channel_match.group(1)) if channel_match else None,
            ssid_match.group(1).strip() if ssid_match else None,
            bssid_match.group(1).strip() if bssid_match else None,
            radio_match.group(1).strip() if radio_match else None
        )
    except (subprocess.SubprocessError, subprocess.TimeoutExpired, ValueError) as e:
        print(f"Error collecting Wi-Fi metrics: {e}")
        return None, None, None, None, 0, None, None, None, None, None



def data_collection_worker():
    """
    Background worker thread for continuous metric collection.
    Collects Wi-Fi metrics at specified intervals.
    """
    global metrics_dataframe
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metric collection worker initialized")
    print(f"Sample interval: {SAMPLE_INTERVAL_SECONDS}s")
    
    while True:
        try:
            timestamp = time.time()
            signal, rssi, rx_rate, tx_rate, snr, link_speed, channel, ssid, bssid, radio = collect_wifi_metrics_windows()
            
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
                "snr_db": snr,
                "link_speed_mbps": link_speed,
                "channel": channel,
                "bandwidth_util": bw_util
            }

            with data_lock:
                new_df = pd.DataFrame([new_record])
                metrics_dataframe = pd.concat(
                    [metrics_dataframe, new_df],
                    ignore_index=True,
                    sort=False
                )
                
                if len(metrics_dataframe) > MAX_DATA_POINTS:
                    metrics_dataframe = metrics_dataframe.iloc[-MAX_DATA_POINTS:].reset_index(drop=True)

        except Exception as e:
            print(f"Error in data collection loop: {e}")
        
        time.sleep(SAMPLE_INTERVAL_SECONDS)

# Start background data collection
collection_thread = threading.Thread(target=data_collection_worker, daemon=True)
collection_thread.start()

# ================= DASHBOARD STYLING =================

THEME_COLORS = {
    "background": "#0a0e27",
    "surface": "#1a1f3a",
    "card": "#252b4a",
    "primary": "#4a9eff",
    "success": "#00c853",
    "warning": "#ffa726",
    "error": "#ef5350",
    "text_primary": "#ffffff",
    "text_secondary": "#b0b8d4",
    "grid": "#2d3350",
    "border": "#3d4466"
}

CARD_STYLE = {
    "backgroundColor": THEME_COLORS["card"],
    "padding": "24px",
    "borderRadius": "12px",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
    "border": f"1px solid {THEME_COLORS['border']}",
    "textAlign": "center",
    "minWidth": "160px",
    "transition": "transform 0.2s"
}

GRAPH_STYLE = {
    "marginBottom": "24px",
    "borderRadius": "12px",
    "overflow": "hidden"
}

# ================= DASH APPLICATION =================

app = dash.Dash(__name__)
app.title = "WiFi Network Performance Dashboard"

app.layout = html.Div(style={
    "backgroundColor": THEME_COLORS["background"],
    "color": THEME_COLORS["text_primary"],
    "minHeight": "100vh",
    "padding": "32px",
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
}, children=[
    
    html.Div(style={"maxWidth": "1600px", "margin": "0 auto"}, children=[
        
        # Header Section
        html.Div(style={
            "backgroundColor": THEME_COLORS["surface"],
            "padding": "32px",
            "borderRadius": "12px",
            "marginBottom": "32px",
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
            "border": f"1px solid {THEME_COLORS['border']}"
        }, children=[
            html.H1("WiFi Network Performance Dashboard", style={
                "margin": "0 0 8px 0",
                "fontSize": "32px",
                "fontWeight": "600",
                "letterSpacing": "-0.5px"
            }),
            html.P("Real-Time Layer 2 WiFi Metrics and Network Analytics", style={
                "margin": "0",
                "fontSize": "16px",
                "color": THEME_COLORS["text_secondary"],
                "fontWeight": "400"
            })
        ]),

        # Auto-refresh interval
        dcc.Interval(
            id="update-interval",
            interval=1000,
            n_intervals=0
        ),

        # KPI Section
        html.Div(style={
            "backgroundColor": THEME_COLORS["surface"],
            "padding": "24px",
            "borderRadius": "12px",
            "marginBottom": "32px",
            "border": f"1px solid {THEME_COLORS['border']}"
        }, children=[
            html.H3("Key Performance Indicators", style={
                "margin": "0 0 20px 0",
                "fontSize": "18px",
                "color": THEME_COLORS["text_primary"],
                "fontWeight": "600"
            }),
            html.Div(id="kpi-container", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
                "gap": "16px"
            })
        ]),

        # Metrics Cards Container
        html.Div(style={
            "backgroundColor": THEME_COLORS["surface"],
            "padding": "24px",
            "borderRadius": "12px",
            "marginBottom": "32px",
            "border": f"1px solid {THEME_COLORS['border']}"
        }, children=[
            html.H3("Current Metrics", style={
                "margin": "0 0 20px 0",
                "fontSize": "18px",
                "color": THEME_COLORS["text_primary"],
                "fontWeight": "600"
            }),
            html.Div(id="metrics-container", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
                "gap": "16px"
            })
        ]),

        # Charts Section
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(500px, 1fr))",
            "gap": "24px",
            "marginBottom": "32px"
        }, children=[
            html.Div(style=GRAPH_STYLE, children=[
                dcc.Graph(id="signal-strength-chart", config={"displayModeBar": False})
            ]),
            
            html.Div(style=GRAPH_STYLE, children=[
                dcc.Graph(id="throughput-chart", config={"displayModeBar": False})
            ])
        ]),

        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(500px, 1fr))",
            "gap": "24px",
            "marginBottom": "32px"
        }, children=[
            html.Div(style=GRAPH_STYLE, children=[
                dcc.Graph(id="snr-rssi-chart", config={"displayModeBar": False})
            ]),
            
            html.Div(style=GRAPH_STYLE, children=[
                dcc.Graph(id="channel-util-chart", config={"displayModeBar": False})
            ])
        ]),

        # Footer
        html.Div(style={
            "textAlign": "center",
            "paddingTop": "32px",
            "color": THEME_COLORS["text_secondary"],
            "fontSize": "13px",
            "borderTop": f"1px solid {THEME_COLORS['border']}"
        }, children=[
            html.P(f"Update Interval: {SAMPLE_INTERVAL_SECONDS}s | Data Points: {MAX_DATA_POINTS} | Auto-Refresh Enabled")
        ])
    ])
])

# ================= CALLBACKS =================

@app.callback(
    Output("kpi-container", "children"),
    Output("metrics-container", "children"),
    Output("signal-strength-chart", "figure"),
    Output("throughput-chart", "figure"),
    Output("snr-rssi-chart", "figure"),
    Output("channel-util-chart", "figure"),
    Input("update-interval", "n_intervals")
)
def update_dashboard_components(n: int):
    """
    Updates all dashboard components with latest metrics data.
    
    Args:
        n: Number of intervals elapsed (not used, required by Dash)
        
    Returns:
        Tuple containing KPIs, metrics cards, and four chart figures
    """
    with data_lock:
        df = metrics_dataframe.copy()
    
    if df.empty:
        empty_layout = create_empty_figure_layout()
        return [], [], empty_layout, empty_layout, empty_layout, empty_layout

    latest_metrics = df.iloc[-1]
    
    # Generate KPI cards
    kpi_cards = create_kpi_cards(df)
    
    # Generate metric cards
    metric_cards = create_metric_cards(latest_metrics)
    
    # Generate charts
    signal_chart = create_signal_strength_chart(df)
    throughput_chart = create_throughput_chart(df)
    snr_chart = create_snr_rssi_chart(df)
    channel_chart = create_channel_util_chart(df)
    
    return kpi_cards, metric_cards, signal_chart, throughput_chart, snr_chart, channel_chart

# ================= CHART GENERATORS =================

def create_base_layout(title: str) -> dict:
    """Creates base layout configuration for charts."""
    return {
        "title": {
            "text": title,
            "font": {"size": 20, "color": THEME_COLORS["text_primary"], "family": "'Segoe UI'"},
            "x": 0.02
        },
        "paper_bgcolor": THEME_COLORS["surface"],
        "plot_bgcolor": THEME_COLORS["card"],
        "font": {"color": THEME_COLORS["text_primary"], "family": "'Segoe UI'"},
        "xaxis": {
            "gridcolor": THEME_COLORS["grid"],
            "showgrid": True,
            "zeroline": False,
            "color": THEME_COLORS["text_secondary"],
            "tickformat": "%H:%M:%S"
        },
        "yaxis": {
            "gridcolor": THEME_COLORS["grid"],
            "showgrid": True,
            "zeroline": False,
            "color": THEME_COLORS["text_secondary"]
        },
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "hovermode": "x unified",
        "showlegend": True,
        "legend": {
            "bgcolor": THEME_COLORS["card"],
            "bordercolor": THEME_COLORS["border"],
            "borderwidth": 1
        }
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
            "color": THEME_COLORS["success"]
        },
        {
            "label": "RSSI",
            "value": f"{latest['rssi_dbm']:.0f} dBm" if pd.notna(latest['rssi_dbm']) else "N/A",
            "color": THEME_COLORS["primary"]
        },
        {
            "label": "SNR",
            "value": f"{latest['snr_db']:.0f} dB" if pd.notna(latest['snr_db']) else "N/A",
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
            "label": "Link Speed",
            "value": f"{latest['link_speed_mbps']:.1f} Mbps" if pd.notna(latest['link_speed_mbps']) else "N/A",
            "color": THEME_COLORS["success"]
        },
        {
            "label": "Channel",
            "value": f"{int(latest['channel'])}" if pd.notna(latest['channel']) else "N/A",
            "color": THEME_COLORS["primary"]
        },
        {
            "label": "BW Util",
            "value": f"{latest['bandwidth_util']:.1f}%" if pd.notna(latest['bandwidth_util']) else "N/A",
            "color": THEME_COLORS["warning"] if (pd.notna(latest['bandwidth_util']) and latest['bandwidth_util'] > 50) else THEME_COLORS["success"]
        }
    ]
    
    return [
        html.Div(style=CARD_STYLE, children=[
            html.Div(metric["label"], style={
                "fontSize": "11px",
                "color": THEME_COLORS["text_secondary"],
                "marginBottom": "8px",
                "fontWeight": "500",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px"
            }),
            html.Div(metric["value"], style={
                "fontSize": "20px",
                "fontWeight": "700",
                "color": metric["color"]
            })
        ]) for metric in metrics
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
    avg_signal = df['signal_percent'].dropna().mean() if not df['signal_percent'].empty else 0
    avg_rssi = df['rssi_dbm'].dropna().mean() if not df['rssi_dbm'].empty else 0
    avg_snr = df['snr_db'].dropna().mean() if not df['snr_db'].empty else 0
    max_throughput = max(df['rx_mbps'].dropna().max(), df['tx_mbps'].dropna().max()) if not df['rx_mbps'].empty else 0
    avg_link_speed = df['link_speed_mbps'].dropna().mean() if not df['link_speed_mbps'].empty else 0
    
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
    
    # SNR quality interpretation
    if avg_snr >= 40:
        snr_status = "Excellent"
        snr_color = THEME_COLORS["success"]
    elif avg_snr >= 25:
        snr_status = "Good"
        snr_color = THEME_COLORS["primary"]
    elif avg_snr >= 15:
        snr_status = "Fair"
        snr_color = THEME_COLORS["warning"]
    else:
        snr_status = "Poor"
        snr_color = THEME_COLORS["error"]
    
    kpis = [
        {
            "label": "Avg Signal",
            "value": f"{avg_signal:.0f}%",
            "status": signal_status,
            "color": signal_color
        },
        {
            "label": "Avg RSSI",
            "value": f"{avg_rssi:.0f} dBm",
            "status": "Signal Power",
            "color": THEME_COLORS["primary"]
        },
        {
            "label": "Avg SNR",
            "value": f"{avg_snr:.0f} dB",
            "status": snr_status,
            "color": snr_color
        },
        {
            "label": "Peak Speed",
            "value": f"{max_throughput:.0f} Mbps",
            "status": "Current Peak",
            "color": THEME_COLORS["success"]
        },
        {
            "label": "Avg Link Speed",
            "value": f"{avg_link_speed:.0f} Mbps",
            "status": "Average Speed",
            "color": THEME_COLORS["warning"]
        }
    ]
    
    return [
        html.Div(style={
            **CARD_STYLE,
            "backgroundColor": THEME_COLORS["card"],
            "borderLeft": f"4px solid {kpi['color']}"
        }, children=[
            html.Div(kpi["label"], style={
                "fontSize": "11px",
                "color": THEME_COLORS["text_secondary"],
                "marginBottom": "8px",
                "fontWeight": "500",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px"
            }),
            html.Div(kpi["value"], style={
                "fontSize": "20px",
                "fontWeight": "700",
                "color": kpi["color"],
                "marginBottom": "4px"
            }),
            html.Div(kpi["status"], style={
                "fontSize": "12px",
                "color": kpi["color"],
                "fontWeight": "600"
            })
        ]) for kpi in kpis
    ]

def create_signal_strength_chart(df: pd.DataFrame) -> dict:
    """Generates a simple signal strength area chart (single axis)."""
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]

    return {
        "data": [
            go.Scatter(
                x=timestamps,
                y=df["signal_percent"],
                mode="lines",
                name="Signal Strength",
                line={"color": THEME_COLORS["success"], "width": 3},
                fill="tozeroy",
                fillcolor=f"rgba(0, 200, 83, 0.2)"
            )
        ],
        "layout": {
            **create_base_layout("Signal Strength"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "range": [0, 100],
                "title": "Signal (%)"
            }
        }
    }

def create_throughput_chart(df: pd.DataFrame) -> dict:
    """Generates a simple grouped bar chart for RX/TX rates (single axis)."""
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]

    return {
        "data": [
            go.Bar(
                x=timestamps,
                y=df["rx_mbps"],
                name="RX Rate",
                marker={"color": THEME_COLORS["primary"], "opacity": 0.8}
            ),
            go.Bar(
                x=timestamps,
                y=df["tx_mbps"],
                name="TX Rate",
                marker={"color": THEME_COLORS["warning"], "opacity": 0.8}
            )
        ],
        "layout": {
            **create_base_layout("Throughput (RX/TX)"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "Bitrate (Mbps)"
            },
            "barmode": "group"
        }
    }

def create_snr_rssi_chart(df: pd.DataFrame) -> dict:
    """Generates a simple SNR scatter plot (single axis)."""
    timestamps = [datetime.fromtimestamp(ts) for ts in df["timestamp"]]

    return {
        "data": [
            go.Scatter(
                x=timestamps,
                y=df["snr_db"],
                mode="markers",
                name="SNR",
                marker={
                    "color": THEME_COLORS["success"],
                    "size": 7
                }
            )
        ],
        "layout": {
            **create_base_layout("SNR (Scatter)"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "SNR (dB)"
            }
        }
    }

def create_channel_util_chart(df: pd.DataFrame) -> dict:
    """Generates a simple histogram for bandwidth utilization (single axis)."""
    bw_util_vals = df["bandwidth_util"].dropna()

    return {
        "data": [
            go.Histogram(
                x=bw_util_vals,
                name="BW Util Distribution",
                marker={"color": THEME_COLORS["warning"], "opacity": 0.7},
                nbinsx=20
            )
        ],
        "layout": {
            **create_base_layout("Bandwidth Utilization (Histogram)"),
            "yaxis": {
                **create_base_layout("")["yaxis"],
                "title": "Frequency"
            },
            "xaxis": {
                **create_base_layout("")["xaxis"],
                "title": "Utilization (%)",
                "tickformat": ""
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