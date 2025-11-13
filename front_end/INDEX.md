# 📊 AI Trading Bot Dashboard - Documentation Index

Welcome to the AI Trading Bot Dashboard! This directory contains all the frontend code and documentation.

## 🚀 Quick Links

### Getting Started
1. **[QUICKSTART.md](QUICKSTART.md)** - Start here! Simple 3-step guide to launch the dashboard
2. **[README.md](README.md)** - Complete documentation with features and usage
3. **[requirements.txt](requirements.txt)** - Python dependencies list

### Running the Dashboard
```bash
# Easiest method:
./run_dashboard.sh

# Or directly:
streamlit run dashboard.py
```

Then open: **http://localhost:8501**

## 📁 Files Overview

### Core Application
- **`dashboard.py`** - Main Streamlit dashboard application
  - Auto-refresh every 5 minutes
  - Real-time trading metrics
  - Position tracking
  - Trade history
  - AI decision cards
  - Portfolio performance chart (optimized scale)

### Configuration
- **`.streamlit/config.toml`** - Theme and server settings
- **`.streamlit/config_external.toml`** - Network access configuration

### Utilities
- **`run_dashboard.sh`** - Local access launcher
- **`run_dashboard_network.sh`** - Network access launcher
- **`run_dashboard_ngrok.sh`** - Internet access launcher (requires ngrok)
- **`test_network_access.sh`** - Network accessibility test

### Documentation
- **`QUICKSTART.md`** - Fast 5-minute setup guide
- **`README.md`** - Complete feature documentation
- **`SHARING_GUIDE.md`** - How to share with others
- **`EXTERNAL_ACCESS.md`** - Detailed external access options
- **`INDEX.md`** (this file) - Documentation navigation

## 🎯 Key Features

### Dashboard Sections

1. **Header Metrics** (4 KPIs)
   - Total Account Value
   - Available Cash
   - Unrealized PnL
   - Open Positions Count

2. **Current Positions** (Left Panel)
   - Active trades with full details
   - Entry/target/stop prices
   - Progress indicators
   - AI justifications

3. **Recent Trades** (Left Panel)
   - Last 10 trade entries/exits
   - Relative timestamps
   - PnL calculations
   - Confidence scores

4. **AI Decisions** (Right Panel)
   - Latest signal per coin
   - Confidence visualization
   - Hold/Entry/Close indicators

5. **Portfolio Chart** (Right Panel)
   - Account value over time
   - **Optimized scale for better visibility**
   - Interactive tooltips
   - Clean line graph

6. **Footer Stats**
   - Model name
   - Total invocations
   - Runtime duration

## 🎨 Design Philosophy

- **Minimalist**: Clean white background, subtle borders
- **Data-Focused**: Information over decoration
- **Color-Coded**: Green (profit/long), Red (loss/short), Blue (neutral)
- **Responsive**: Adapts to screen sizes
- **Interactive**: Expandable sections, hover tooltips

## 📦 Dependencies

```
streamlit >= 1.28.0
pandas >= 2.0.0
plotly >= 5.17.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🔧 Customization

### Change Auto-Refresh Interval
Edit `dashboard.py` line ~362:
```python
st.session_state.refresh_interval = 300  # seconds
```

### Modify Colors
Edit CSS in `dashboard.py` or `.streamlit/config.toml`

### Adjust Column Widths
Change ratio in `st.columns()` calls:
```python
col_left, col_right = st.columns([3, 2])  # 60/40 split
```

## 📊 Data Requirements

The dashboard reads CSV files from `../data/`:
- `ai_decisions.csv` - Model decisions and signals
- `ai_messages.csv` - Conversation logs
- `portfolio_state.csv` - Portfolio snapshots
- `trade_history.csv` - Trade records

## 💡 Tips

- Use **manual refresh** button for immediate updates
- **Expand justifications** to see full AI reasoning
- **Hover over chart** for exact portfolio values
- **Network mode** to share on WiFi
- **ngrok mode** to share on internet

## ✅ Project Status

✅ All required features implemented  
✅ All code isolated in `front_end/` folder  
✅ Comprehensive documentation provided  
✅ Auto-refresh with countdown timer  
✅ Minimalist design following specs  
✅ Color-coded metrics and badges  
✅ Interactive charts and expandable sections  
✅ Responsive layout (desktop/tablet/mobile)  
✅ Optimized chart scale for better visibility  

**Status**: Ready for Production 🎉

---

**Last Updated**: November 2, 2025  
**Version**: 1.0  
**Framework**: Streamlit  
**Python**: 3.8+

