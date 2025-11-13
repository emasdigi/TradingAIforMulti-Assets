#!/bin/bash

# Stop any running Streamlit processes and restart clean

echo "🔄 Restarting Streamlit Dashboard..."
echo ""

# Kill any existing streamlit processes
pkill -f "streamlit run dashboard.py" 2>/dev/null
sleep 2

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "✅ Cache cleared"
echo "🚀 Starting fresh dashboard..."
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Run Streamlit
streamlit run dashboard.py

