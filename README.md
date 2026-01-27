# WiFi Network Performance Monitor

A real -time, cross-platform WiFi network performance monitoring dashboard with advanced analytics and visualization capabilities. This tool provides comprehensive insights into WiFi link quality, signal strength, latency, throughput, and network stability metrics.

## Features

### 📊 Real-Time Monitoring
- **Signal Strength Tracking**: RSSI (dBm) and signal percentage over time
- **PHY Rate Monitoring**: RX/TX physical data rates with dual-axis visualization
- **Latency Analysis**: Round-Trip Time (RTT) measurements with jitter/variance tracking
- **Link Stability Scoring**: 0-100 stability score based on signal variance, throughput, and RTT consistency
- **Anomaly Detection**: Rule-based detection of signal drops, latency spikes, and throughput degradation

### 🔍 Advanced Analytics
- **Network Link Analysis**: Comprehensive link state assessment (STABLE/DEGRADING/UNSTABLE)
- **Cross-Layer Insights**: PHY-Network layer interaction analysis
- **RSSI Variation Tracking**: ΔRSSI (deviation from rolling mean) to capture signal dynamics
- **RTT Jitter Analysis**: Network turbulence detection through variance measurements
- **MAC-Layer Metrics** (Linux only): Retransmission rate monitoring via `iw station dump`

### 📈 Interactive Visualizations
- Signal Strength vs Time (dual-axis: RSSI + Signal %)
- RX vs TX PHY Rate comparison
- RTT Variance (Jitter) Over Time
- Link Stability Gauge
- Anomaly Timeline (binary event markers)
- RSSI vs RTT Scatter Plot (color-coded by link state)
- RSSI Variation Relative to Rolling Mean

### 🎨 Professional Dashboard
- Minimal, polished UI design
- Real-time auto-refresh (1-second intervals)
- Responsive grid layouts
- Hover tooltips for all metrics
- Dark theme with professional styling

## Requirements

### System Requirements
- **Python 3.8+**
- **Operating System**: 
  - Windows 10/11 (for `lwifi.py` or `wifi_windows.py`)
  - Linux with `iw` and `iwconfig` tools (for `wifi_linux.py`)

### Python Dependencies
```bash
pip install dash plotly pandas numpy
```

### Linux-Specific Requirements
```bash
# Install iw and wireless-tools
sudo apt-get install iw wireless-tools  # Debian/Ubuntu
sudo yum install iw wireless-tools      # RHEL/CentOS
```

### Windows-Specific Requirements
- Built-in `netsh` command (available by default)
- Administrator privileges may be required for some metrics

## Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd wifi_dashboard
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install dash plotly pandas numpy
   ```

## Configuration

### Linux Configuration (`wifi_linux.py`)

Edit the configuration section at the top of `wifi_linux.py`:

```python
INTERFACE = "wlp1s0"          # Your WiFi interface name
PING_TARGET = "8.8.8.8"       # Ping target for latency measurement
SAMPLE_INTERVAL_SECONDS = 0.5 # Data collection interval
MAX_DATA_POINTS = 600         # Maximum data points to store
DASHBOARD_PORT = 8050        # Dashboard web server port
DASHBOARD_HOST = "127.0.0.1" # Dashboard host address
```

**Finding your WiFi interface:**
```bash
iwconfig                    # List all wireless interfaces
ip link show           # List all network interfaces
```

### Windows Configuration (`lwifi.py` or `wifi_windows.py`)

Edit the configuration section:

```python
PING_TARGET = "8.8.8.8"       # Ping target for latency measurement
SAMPLE_INTERVAL_SECONDS = 0.5 # Data collection interval
MAX_DATA_POINTS = 600         # Maximum data points to store
DASHBOARD_PORT = 8050        # Dashboard web server port
DASHBOARD_HOST = "127.0.0.1" # Dashboard host address
```

## Usage

### Linux

```bash
python wifi_linux.py
```

The dashboard will be available at: `http://127.0.0.1:8050`

### Windows

```bash
python lwifi.py
# or
python wifi_windows.py
```

The dashboard will be available at: `http://127.0.0.1:8050`

### Simple Logger (Linux)

For basic CSV logging without the dashboard:

```bash
python l2_wifi_logger.py
```

This will create `l2_wifi_metrics.csv` with timestamp, RSSI, RX/TX bitrates, and RTT.

## Dashboard Components

### Connection Information
- **SSID**: Network name
- **Radio Type**: WiFi standard (802.11ac, 802.11ax, etc.)
- **Channel**: WiFi frequency channel
- **BSSID**: Access point MAC address

### Key Performance Indicators (KPIs)
- Average Signal Strength
- Average RSSI

### Current Metrics
- Signal Strength (%)
- RSSI (dBm)
- RX Rate (Mbps)
- TX Rate (Mbps)
- Channel

### Network Link Analysis
- **Link State Indicator**: Current link state (STABLE/DEGRADING/UNSTABLE)
- **Link State Explanation**: Detailed breakdown of signal, RTT, and PHY rate trends
- **Cross-Layer Analysis**: Insights into PHY-Network layer interactions
- **MAC Retransmissions** (Linux only): Retry rate and status

### Charts
1. **Signal Strength vs Time**: Dual-axis chart showing RSSI and signal percentage
2. **RX vs TX PHY Rate**: Comparison of receive and transmit rates
3. **RTT Variance (Jitter) Over Time)**: Network stability indicator
4. **Link Stability Score**: Gauge showing overall connection quality (0-100)
5. **Anomaly Timeline**: Binary timeline of detected anomalies
6. **RSSI vs RTT Scatter Plot**: Relationship between signal strength and latency
7. **RSSI Variation Relative to Rolling Mean**: Signal dynamics and mobility effects

## Metrics Explained

### Signal Strength
- **Signal %**: 0-100% quality indicator (higher is better)
- **RSSI (dBm)**: Received Signal Strength Indicator in decibels
  - Excellent: -30 to -50 dBm
  - Good: -50 to -60 dBm
  - Fair: -60 to -70 dBm
  - Poor: -70 to -90 dBm

### PHY Rates
- **RX Rate**: Physical data rate for receiving data (Mbps)
- **TX Rate**: Physical data rate for transmitting data (Mbps)
- Rates depend on WiFi standard, signal strength, and interference

### Latency
- **RTT (ms)**: Round-Trip Time to ping target
- **RTT Jitter**: Standard deviation of RTT values (lower is better)
- High jitter indicates network instability

### Link State
- **STABLE**: Consistent performance, low variance
- **DEGRADING**: Showing signs of degradation
- **UNSTABLE**: High variability, poor performance

## Troubleshooting

### Linux Issues

**"iw: command not found"**
```bash
sudo apt-get install iw
```

**"Permission denied" for iw commands**
```bash
# Run with sudo or add user to netdev group
sudo usermod -aG netdev $USER
# Log out and back in
```

**Interface not found**
```bash
# List available interfaces
iwconfig
# Update INTERFACE in configuration
```

### Windows Issues

**"netsh: access denied"**
- Run Python script as Administrator

**No WiFi metrics**
- Ensure you're connected to a WiFi network
- Check that WiFi adapter is enabled

### General Issues

**Dashboard not loading**
- Check if port 8050 is already in use
- Verify firewall settings
- Try changing `DASHBOARD_PORT` in configuration

**No data appearing**
- Wait a few seconds for initial data collection
- Check console for error messages
- Verify network connectivity

## Performance Notes

- Data collection runs in a background thread to prevent UI blocking
- Dashboard updates every 1 second
- Maximum 600 data points stored (configurable)
- Sample interval: 0.5 seconds (configurable)

## File Structure

```
wifi_dashboard/
├── README.md              # This file
├── wifi_linux.py         # Linux dashboard (uses iw/iwconfig)
├── lwifi.py              # Windows dashboard (uses netsh)
├── wifi_windows.py       # Alternative Windows dashboard
└── l2_wifi_logger.py     # Simple CSV logger (Linux)
```

## Technical Details

### Data Collection
- **Thread-safe**: Uses threading locks for concurrent access
- **Non-blocking**: Optimized timeouts prevent UI freezing
- **Error handling**: Graceful degradation on command failures

### Metrics Calculation
- **Stability Score**: Weighted combination of signal variance, throughput stability, and RTT consistency
- **Anomaly Detection**: Rule-based detection of sudden drops/spikes
- **RSSI Delta**: Difference from rolling average (10-sample window)
- **RTT Jitter**: Standard deviation of recent RTT values (10-sample window)

### Linux-Specific Features
- MAC-layer retransmission monitoring via `iw dev <interface> station dump`
- More detailed link statistics than Windows

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

Built with:
- [Dash](https://dash.plotly.com/) - Web framework
- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing

## Support

For issues, questions, or contributions, please [open an issue](link-to-issues) or [create a pull request](link-to-prs).

