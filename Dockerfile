# ======================================================
# Dockerfile for RAG Agentic AI Audit System
# Optimized for Python 3.11-slim | Streamlit 1.39+
# ======================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python -m pip install --upgrade pip==25.1.1

# Set working directory
WORKDIR /app

# Copy only requirement files first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "main_application.py", "--server.address=0.0.0.0"]
