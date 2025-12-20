#!/bin/bash
# Quick launcher for the NeoSkidRL Reward Dashboard

cd "$(dirname "$0")"

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing UI dependencies..."
    pip install -e ".[ui]"
fi

echo "Launching NeoSkidRL Reward Dashboard..."
echo "Dashboard will open in your browser at http://localhost:8501"
echo ""

streamlit run src/neoskidrl/ui/reward_dashboard.py

