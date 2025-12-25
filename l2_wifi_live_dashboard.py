import subprocess
import time
import threading
import re
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

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
    print("📡 L2 logger started")

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

app = dash.Dash(__name__)
app.title = "Live Wi-Fi Dashboard (L2)"

app.layout = html.Div([
    html.H1("📡 Live Wi-Fi Performance Dashboard (L2)"),

    dcc.Interval(
        id="interval",
        interval=1000,
        n_intervals=0
    ),

    html.Div(id="metrics", style={
        "display": "flex",
        "gap": "30px",
        "marginBottom": "20px"
    }),

    dcc.Graph(id="rssi_graph"),
    dcc.Graph(id="rtt_graph"),
    dcc.Graph(id="bitrate_graph")
])

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
        html.Div([html.H4("RSSI (dBm)"), html.H2(latest["rssi_dbm"])]),
        html.Div([html.H4("RX PHY (Mbps)"), html.H2(latest["rx_bitrate_mbps"])]),
        html.Div([html.H4("TX PHY (Mbps)"), html.H2(latest["tx_bitrate_mbps"])]),
        html.Div([html.H4("RTT (ms)"), html.H2(latest["rtt_ms"])])
    ]

    rssi_fig = {
        "data": [{
            "x": df["timestamp"],
            "y": df["rssi_dbm"],
            "type": "line",
            "name": "RSSI"
        }],
        "layout": {"title": "RSSI Over Time"}
    }

    rtt_fig = {
        "data": [{
            "x": df["timestamp"],
            "y": df["rtt_ms"],
            "type": "line",
            "name": "RTT"
        }],
        "layout": {"title": "RTT Over Time"}
    }

    bitrate_fig = {
        "data": [
            {
                "x": df["timestamp"],
                "y": df["rx_bitrate_mbps"],
                "type": "line",
                "name": "RX PHY"
            },
            {
                "x": df["timestamp"],
                "y": df["tx_bitrate_mbps"],
                "type": "line",
                "name": "TX PHY"
            }
        ],
        "layout": {"title": "PHY Bitrate Over Time"}
    }

    return metrics, rssi_fig, rtt_fig, bitrate_fig

# ---------------- RUN ------------------------

if __name__ == "__main__":
    print("🚀 Starting Dash server at http://127.0.0.1:8050")
    app.run(debug=False)
