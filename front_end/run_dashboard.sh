#!/bin/bash

# AI Trading Bot Dashboard Launcher
# This script launches the Streamlit dashboard

echo "🚀 Starting AI Trading Bot Dashboard..."
echo "📊 Dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Run Streamlit
streamlit run dashboard.py

