@echo off
echo Starting RAG Agentic AI Internal Audit System...
echo Setting environment variables...

set ANONYMIZED_TELEMETRY=False
set CHROMA_SERVER_NOFILE=65536
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo Launching application...
streamlit run audit_app_error_fixed.py

pause
