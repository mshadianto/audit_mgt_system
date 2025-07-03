# Dockerfile
FROM python:3.11-slim

# Update pip dalam container
RUN python -m pip install --upgrade pip==25.1.1

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dengan updated pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main_application.py", "--server.address", "0.0.0.0"]