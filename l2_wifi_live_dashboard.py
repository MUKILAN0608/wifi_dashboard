import subprocess
import time
import threading
import re
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# ================== CONFIG ==================
INTERFACE = "wlp1s0"
PING_TARGET = "8.8.8.8"
MAX_ROWS = 300   # keep last 5 minutes (1 sec sampling)
# ============================================

# Shared dataframe (thread-safe by design for append-only)
df = pd.DataFrame(columns=[
    "timestamp",
    "rssi_dbm",
    "rx_bitrate_mbps",
    "tx_bitrate_mbps",
    "rtt_ms"
])

# ----------- METRIC COLLECTION ---------------

def get_iw_metrics():
    out = subprocess.getoutput(f"iw dev {INTERFACE} link")
    if "Not connected" in out:
        return None, None, None

    rssi = re.search(r"signal:\s+(-\d+)", out)
    rx = re.search(r"rx bitrate:\s+([\d\.]+)", out)
    tx = re.search(r"tx bitrate:\s+([\d\.]+)", out)

    return (
        int(rssi.group(1)) if rssi else None,
        float(rx.group(1)) if rx else None,
        float(tx.group(1)) if tx else None
    )

def get_rtt():
    out = subprocess.getoutput(f"ping -c 1 -W 1 {PING_TARGET}")
    rtt = re.search(r"time=([\d\.]+)", out)
    return float(rtt.group(1)) if rtt else None

def logger_loop():
    global df
    print("Starting Wi-Fi Performance Evaluation and Monitoring Interface Logger")

    while True:
        ts = time.time()
        rssi, rx, tx = get_iw_metrics()
        rtt = get_rtt()

        new_row = {
            "timestamp": ts,
            "rssi_dbm": rssi,
            "rx_bitrate_mbps": rx,
            "tx_bitrate_mbps": tx,
            "rtt_ms": rtt
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        if len(df) > MAX_ROWS:
            df = df.iloc[-MAX_ROWS:]

        time.sleep(1)

# Start logger thread
threading.Thread(target=logger_loop, daemon=True).start()

# ---------------- DASH APP -------------------

# Initialize the app with a dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Wi-Fi Performance Evaluation and Monitoring Interface"

app.layout = html.Div([
    html.H1("Wi-Fi Performance Evaluation and Monitoring Interface", style={"textAlign": "center", "color": "#FFFFFF"}),

    dcc.Interval(
        id="interval",
        interval=1000,
        n_intervals=0
    ),

    html.Div(id="metrics", style={
        "display": "flex",
        "gap": "30px",
        "marginBottom": "20px",
        "justifyContent": "center",
        "color": "#FFFFFF"
    }),

    dcc.Graph(id="rssi_graph", style={"backgroundColor": "#1E1E1E"}),
    dcc.Graph(id="rtt_graph", style={"backgroundColor": "#1E1E1E"}),
    dcc.Graph(id="bitrate_graph", style={"backgroundColor": "#1E1E1E"})
], style={"backgroundColor": "#121212", "padding": "20px"})

@app.callback(
    Output("metrics", "children"),
    Output("rssi_graph", "figure"),
    Output("rtt_graph", "figure"),
    Output("bitrate_graph", "figure"),
    Input("interval", "n_intervals")
)
def update_dashboard(_):
    if df.empty:
        return [], {}, {}, {}

    latest = df.iloc[-1]

    metrics = [
        html.Div([html.H4("RSSI (dBm)", style={"color": "#FFFFFF"}), html.H2(latest["rssi_dbm"], style={"color": "#FFFFFF"})]),
        html.Div([html.H4("RX PHY (Mbps)", style={"color": "#FFFFFF"}), html.H2(latest["rx_bitrate_mbps"], style={"color": "#FFFFFF"})]),
        html.Div([html.H4("TX PHY (Mbps)", style={"color": "#FFFFFF"}), html.H2(latest["tx_bitrate_mbps"], style={"color": "#FFFFFF"})]),
        html.Div([html.H4("RTT (ms)", style={"color": "#FFFFFF"}), html.H2(latest["rtt_ms"], style={"color": "#FFFFFF"})])
    ]

    rssi_fig = {
        "data": [{
            "x": df["timestamp"],
            "y": df["rssi_dbm"],
            "type": "line",
            "name": "RSSI",
            "line": {"color": "#FF5733"}
        }],
        "layout": {"title": "RSSI Over Time", "plot_bgcolor": "#121212", "paper_bgcolor": "#121212", "font": {"color": "#FFFFFF"}}
    }

    rtt_fig = {
        "data": [{
            "x": df["timestamp"],
            "y": df["rtt_ms"],
            "type": "line",
            "name": "RTT",
            "line": {"color": "#33FF57"}
        }],
        "layout": {"title": "RTT Over Time", "plot_bgcolor": "#121212", "paper_bgcolor": "#121212", "font": {"color": "#FFFFFF"}}
    }

    bitrate_fig = {
        "data": [
            {
                "x": df["timestamp"],
                "y": df["rx_bitrate_mbps"],
                "type": "line",
                "name": "RX PHY",
                "line": {"color": "#3357FF"}
            },
            {
                "x": df["timestamp"],
                "y": df["tx_bitrate_mbps"],
                "type": "line",
                "name": "TX PHY",
                "line": {"color": "#FF33A1"}
            }
        ],
        "layout": {"title": "PHY Bitrate Over Time", "plot_bgcolor": "#121212", "paper_bgcolor": "#121212", "font": {"color": "#FFFFFF"}}
    }

    return metrics, rssi_fig, rtt_fig, bitrate_fig

# ---------------- RUN ------------------------

if __name__ == "__main__":
    print("Starting Dash server at http://127.0.0.1:8050")
    app.run(debug=False)
