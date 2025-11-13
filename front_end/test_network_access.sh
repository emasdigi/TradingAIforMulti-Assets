#!/bin/bash

# Test if dashboard is accessible from network

echo "🧪 Testing Dashboard Network Accessibility"
echo "=========================================="
echo ""

# Get local IP
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ipconfig getifaddr en0)
    if [ -z "$LOCAL_IP" ]; then
        LOCAL_IP=$(ipconfig getifaddr en1)
    fi
else
    LOCAL_IP=$(hostname -I | awk '{print $1}')
fi

echo "🔍 Your IP Address: $LOCAL_IP"
echo ""

# Check if port 8501 is in use
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✅ Port 8501 is open and listening"
    echo ""
    
    # Test local access
    echo "🧪 Testing localhost access..."
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "✅ Dashboard accessible on localhost"
    else
        echo "❌ Dashboard NOT accessible on localhost"
    fi
    echo ""
    
    # Test network access
    echo "🧪 Testing network access..."
    if curl -s http://$LOCAL_IP:8501 > /dev/null 2>&1; then
        echo "✅ Dashboard accessible on network"
        echo ""
        echo "🎉 SUCCESS! Dashboard is accessible at:"
        echo "   http://$LOCAL_IP:8501"
    else
        echo "❌ Dashboard NOT accessible on network"
        echo ""
        echo "💡 Try running with network access:"
        echo "   ./run_dashboard_network.sh"
    fi
else
    echo "❌ Port 8501 is not in use"
    echo ""
    echo "💡 Dashboard is not running. Start it with:"
    echo "   ./run_dashboard.sh"
    echo "   or"
    echo "   ./run_dashboard_network.sh"
fi

echo ""
echo "=========================================="
