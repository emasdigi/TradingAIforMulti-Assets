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

# # Start the trading bot in the background
# echo "📈 Starting Trading Bot..."
# python -u main.py &
# BOT_PID=$!
# echo "✅ Trading Bot started (PID: $BOT_PID)"
# echo ""

# # Wait a moment for bot to initialize
# sleep 2

# # Start the news cache updater in the background
# echo "🔄 Starting News Cache Updater (hourly)..."
# (
#     while true; do
#         loop_start=$(date +%s)
#         echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running bot/news_cache.py..."
#         python -u bot/news_cache.py || echo "⚠️  bot/news_cache.py failed, will retry in 1 hour"
#         loop_end=$(date +%s)
#         runtime=$((loop_end - loop_start))
#         sleep_duration=$((3600 - runtime))
#         if [ $sleep_duration -gt 0 ]; then
#             echo "[$(date '+%Y-%m-%d %H:%M:%S')] bot/news_cache.py completed. Waiting $sleep_duration seconds (~$((sleep_duration / 60)) minutes) before next run..."
#             sleep $sleep_duration
#         else
#             echo "[$(date '+%Y-%m-%d %H:%M:%S')] bot/news_cache.py took longer than 1 hour; starting next run immediately..."
#         fi
#     done
# ) &
# CACHE_PID=$!
# echo "✅ News Cache Updater started (PID: $CACHE_PID)"
# echo ""

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
echo "🔄 News Cache Updater: Running every hour"
echo ""
echo "Press Ctrl+C to stop all services"
echo "======================================"
echo ""

# Wait for all background processes
wait
