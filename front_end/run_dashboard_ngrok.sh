#!/bin/bash

# Launch dashboard and create ngrok tunnel for internet access
# Requires ngrok to be installed and configured

echo "🌍 Starting AI Trading Bot Dashboard with Internet Access (ngrok)"
echo "=================================================================="
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed"
    echo ""
    echo "To install ngrok:"
    echo "  Mac:   brew install ngrok"
    echo "  Other: Download from https://ngrok.com/download"
    echo ""
    echo "After installing:"
    echo "  1. Sign up at https://ngrok.com"
    echo "  2. Get your auth token"
    echo "  3. Run: ngrok config add-authtoken YOUR_TOKEN"
    echo ""
    exit 1
fi

# Navigate to script directory
cd "$(dirname "$0")"

# Start Streamlit in background
echo "🚀 Starting Streamlit dashboard..."
streamlit run dashboard.py --server.port 8501 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 5

# Start ngrok tunnel
echo ""
echo "🌐 Creating secure tunnel with ngrok..."
echo ""

# Run ngrok
ngrok http 8501

# Cleanup when script exits
trap "kill $STREAMLIT_PID" EXIT
