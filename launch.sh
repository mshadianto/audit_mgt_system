#!/bin/bash
echo "Starting RAG Agentic AI Internal Audit System..."
echo "Setting environment variables..."

export ANONYMIZED_TELEMETRY=False
export CHROMA_SERVER_NOFILE=65536
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "Launching application..."
streamlit run audit_app_error_fixed.py
