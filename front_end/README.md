# AI Trading Bot Dashboard

A minimalist Streamlit dashboard for monitoring AI-powered crypto trading activity.

## Features

- **Real-time Monitoring**: Auto-refreshes every 5 minutes
- **Key Metrics**: Account value, available cash, unrealized PnL, open positions
- **Position Tracking**: View all active positions with entry prices, targets, and stop losses
- **Trade History**: Timeline of recent trades with AI reasoning
- **AI Decisions**: Latest signals and confidence scores for each asset
- **Portfolio Performance**: Visual chart tracking account value over time (optimized scale)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or if using UV:
```bash
uv add streamlit pandas plotly
```

## Usage

### Option 1: Local Access Only
```bash
./run_dashboard.sh
```

### Option 2: Network Access (Same WiFi)
```bash
./run_dashboard_network.sh
```

### Option 3: Internet Access (via ngrok)
```bash
./run_dashboard_ngrok.sh
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Data Sources

The dashboard reads from the following CSV files in the `data/` directory:

- `ai_decisions.csv` - AI model decisions with timestamps and confidence scores
- `ai_messages.csv` - Full conversation history between system and AI
- `portfolio_state.csv` - Portfolio snapshots with positions and PnL
- `trade_history.csv` - Executed trades with entry/exit details

## Features in Detail

### Header Metrics
- **Total Account Value**: Current equity with percentage change
- **Available Cash**: Uninvested balance
- **Unrealized PnL**: Profit/loss on open positions (color-coded)
- **Open Positions**: Number of active trades

### Current Positions Table
Each position shows:
- Coin symbol and side (LONG/SHORT)
- Entry price and quantity
- Leverage multiplier
- Profit target and stop loss levels
- Margin used and confidence score
- Visual progress bar showing price position
- Expandable AI trade justification

### Recent Trades Timeline
Displays the last 10 trades with:
- Relative timestamp (e.g., "5 mins ago")
- Action type (ENTRY/EXIT)
- Trade details (coin, side, quantity, price)
- PnL for closed positions
- Confidence score
- Expandable AI reasoning

### AI Decision Summary
Latest signal for each coin showing:
- Current recommendation (ENTRY/HOLD/CLOSE)
- Confidence score with color-coded progress bar
- Timestamp of decision

### Portfolio Performance Chart
- Line chart of account value over time
- **Optimized scale** - Dynamically adjusts to show variations clearly
- Markers on data points
- Interactive hover tooltips

## Design

The dashboard follows a minimalist design philosophy:
- **Colors**: Clean white/gray background with blue, green, and red accents
- **Typography**: Sans-serif fonts with monospace for numbers
- **Spacing**: Generous whitespace for readability
- **Layout**: Responsive two-column design (60/40 split)

## Auto-Refresh

The dashboard automatically refreshes every 5 minutes. A countdown timer shows time until next refresh, and a manual refresh button is available for immediate updates.

## Customization

You can adjust the refresh interval by modifying the `refresh_interval` value in the `main()` function (default: 300 seconds).

## External Access

See `SHARING_GUIDE.md` for instructions on making the dashboard accessible to external users.

## Files

- `dashboard.py` - Main application
- `requirements.txt` - Python dependencies
- `run_dashboard.sh` - Local launcher
- `run_dashboard_network.sh` - Network access launcher
- `run_dashboard_ngrok.sh` - Internet access launcher (requires ngrok)
- `test_network_access.sh` - Test network accessibility
- `.streamlit/config.toml` - Streamlit theme configuration

