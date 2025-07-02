#!/bin/bash
# RAG Agentic AI - One-Click Launcher

echo "Ì¥ç RAG Agentic AI Internal Audit System - Quick Start"
echo "=" * 60

# Check if setup is needed
if [ ! -d "venv" ]; then
    echo "Ì¥ß First time setup..."
    ./scripts/setup.sh
fi

echo "Ì∫Ä Starting application..."
./scripts/run.sh
