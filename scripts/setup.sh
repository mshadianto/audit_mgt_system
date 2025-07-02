#!/bin/bash
# RAG Agentic AI Internal Audit System - Setup Script

set -e

echo "Ì¥ç Setting up RAG Agentic AI Internal Audit System..."

# Check Python version
if ! python3 --version | grep -q "3.1[1-9]"; then
    echo "‚ùå Python 3.11+ required"
    exit 1
fi

# Create virtual environment
echo "Ì≥¶ Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Ì≥¶ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template
if [ ! -f .env ]; then
    cp .env.template .env
    echo "‚öôÔ∏è Please edit .env file with your API keys"
fi

echo "‚úÖ Setup completed!"
echo "Ì∫Ä Run: ./scripts/run.sh"
