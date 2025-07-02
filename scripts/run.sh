#!/bin/bash
# Quick start script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "í´§ Running setup first..."
    chmod +x scripts/setup.sh
    ./scripts/setup.sh
fi

echo "íº€ Starting RAG Agentic AI Internal Audit System..."

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Start application
streamlit run main_application.py --server.port=8501 --server.address=0.0.0.0 --browser.gatherUsageStats=false
