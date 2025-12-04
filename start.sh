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

# Start news cache refresher loop
NEWS_REFRESH_INTERVAL_SECONDS=${NEWS_REFRESH_INTERVAL_SECONDS:-10800}
start_news_cache_refresher() {
    echo "📰 Starting News Cache refresher (interval: ${NEWS_REFRESH_INTERVAL_SECONDS}s)..."
    while true; do
        echo "📰 Refreshing news cache at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        if python -m bot.news_cache; then
            echo "✅ News cache refresh completed"
        else
            echo "⚠️ News cache refresh failed" >&2
        fi
        sleep "${NEWS_REFRESH_INTERVAL_SECONDS}"
    done
}
start_news_cache_refresher &
NEWS_PID=$!
echo "✅ News cache refresher started (PID: $NEWS_PID)"
echo ""

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
echo "📰 News Cache: Auto-refreshing every ${NEWS_REFRESH_INTERVAL_SECONDS}s"
echo ""
echo "Press Ctrl+C to stop all services"
echo "======================================"
echo ""

# Wait for both processes
wait
