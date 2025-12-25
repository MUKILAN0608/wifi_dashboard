import time
import subprocess
import csv
import re

INTERFACE = "wlp1s0"
PING_TARGET = "8.8.8.8"
LOG_FILE = "l2_wifi_metrics.csv"

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

print("📡 L2 Wi-Fi Logger Started")
print(f"Logging to {LOG_FILE} (Ctrl+C to stop)")

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "rssi_dbm",
        "rx_bitrate_mbps",
        "tx_bitrate_mbps",
        "rtt_ms"
    ])

    try:
        while True:
            ts = time.time()
            rssi, rx, tx = get_iw_metrics()
            rtt = get_rtt()

            writer.writerow([ts, rssi, rx, tx, rtt])
            f.flush()
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Logging stopped by user")
