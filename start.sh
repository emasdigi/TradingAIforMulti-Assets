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

# Function to refresh news cache periodically
refresh_news_loop() {
    # Run initial refresh immediately
    echo "📰 Running initial news cache refresh..."
    python -m bot.news_cache
    echo "✅ Initial news refresh completed"
    
    # Then refresh every 3 hours (10800 seconds)
    while true; do
        sleep 10800  # 3 hours
        echo "📰 Refreshing news cache (scheduled refresh)..."
        python -m bot.news_cache
        echo "✅ News refresh completed at $(date)"
    done
}

# Start the news cache refresh loop in the background
refresh_news_loop &
NEWS_PID=$!
echo "✅ News Cache Scheduler started (PID: $NEWS_PID, refreshes every 3 hours)"
echo ""

# Wait a moment for initial news cache to complete
sleep 5

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
echo "📰 News Scheduler: Active (refreshes every 3 hours)"
echo ""
echo "Press Ctrl+C to stop all services"
echo "======================================"
echo ""

# Wait for both processes
wait