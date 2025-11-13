#!/bin/bash

# Launch Streamlit dashboard with network access
# Accessible to anyone on the same network

echo "🌐 Starting AI Trading Bot Dashboard with Network Access"
echo "=================================================="
echo ""

# Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ipconfig getifaddr en0)
    if [ -z "$LOCAL_IP" ]; then
        LOCAL_IP=$(ipconfig getifaddr en1)
    fi
else
    # Linux
    LOCAL_IP=$(hostname -I | awk '{print $1}')
fi

echo "📍 Your computer's IP address: $LOCAL_IP"
echo ""
echo "📊 Dashboard will be accessible at:"
echo "   Local:    http://localhost:8501"
echo "   Network:  http://$LOCAL_IP:8501"
echo ""
echo "Share the Network URL with others on your WiFi/LAN"
echo "Press Ctrl+C to stop the dashboard"
echo ""
echo "=================================================="

# Navigate to script directory
cd "$(dirname "$0")"

# Run Streamlit with network access
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
