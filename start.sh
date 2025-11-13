#!/bin/bash

# AI Trading Bot - Multi-Process Startup Script
# This script starts both the trading bot and the dashboard

set -e

echo "🚀 Starting AI Trading Bot System..."
echo "======================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill 0
    wait
    echo "✅ All services stopped"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start the trading bot in the background
echo "📈 Starting Trading Bot..."
python -u main.py &
BOT_PID=$!
echo "✅ Trading Bot started (PID: $BOT_PID)"
echo ""

# Wait a moment for bot to initialize
sleep 2

# Start the Streamlit dashboard in the background
echo "📊 Starting Dashboard..."
cd front_end
streamlit run dashboard.py \
    --server.port=8081 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false &
DASHBOARD_PID=$!
echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo ""

echo "======================================"
echo "✅ All services running!"
echo ""
echo "📊 Dashboard: http://localhost:8081"
echo "📈 Trading Bot: Active"
echo ""
echo "Press Ctrl+C to stop all services"
echo "======================================"
echo ""

# Wait for both processes
wait

