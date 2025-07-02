#!/bin/bash
# RAG Agentic AI - One-Click Launcher

echo "� RAG Agentic AI Internal Audit System - Quick Start"
echo "=" * 60

# Check if setup is needed
if [ ! -d "venv" ]; then
    echo "� First time setup..."
    ./scripts/setup.sh
fi

echo "� Starting application..."
./scripts/run.sh
